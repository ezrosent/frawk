//! Batched splitting of input.
//!
//! Under some circumstances, we can scan the entire input for field and record separators in
//! batches.
//!
//! CSV/TSV parsing is adapted from geofflangdale/simdcsv.
//!
//! As that repo name implies, it uses SIMD instructions to accelerate CSV and TSV parsing. It does
//! this by performing a fast pass over the input that produces a list of "relevant indices" into
//! the underlying input buffer, these play the same role as "control characters" in the SimdJSON
//! paper (by the same authors). The higher-level state machine parser then operates on these
//! control characters rather than on a byte-by-byte basis.
//!
//! Single-byte splitting is a simpler algorithm adapting the CSV/TSV parsing methods, but ignoring
//! escaping issues.

/// TODO: on uncommon/adversarial inputs, these parsers can exhibit quadratic behavior. They will
/// rescan all of the input every time a line exceeds the input chunk size (a few KB by default).
/// We can fix this easily enough by reusing the computed offsets at the end of the buffer into an
/// auxiliary vector at the cost of 2x steady-state memory usage, or more complex offset management
/// in the `Offsets` type.
/// NB the changes to fix this issue will now be in the chunk module.
use std::borrow::Borrow;
use std::io::Read;
use std::mem;
use std::str;

use lazy_static::lazy_static;
use regex::{bytes, Regex};

use crate::common::{ExecutionStrategy, Result};
use crate::pushdown::FieldSet;
use crate::runtime::{
    str_impl::{Buf, Str, UniqueBuf},
    Int, LazyVec, RegexCache,
};

use super::{
    chunk::{self, Chunk, ChunkProducer, OffsetChunk, ParallelChunkProducer},
    normalize_join_indexes, DefaultLine, LineReader, ReaderState,
};

pub struct CSVReader<P> {
    prod: P,
    cur_chunk: OffsetChunk,
    cur_buf: Buf,
    buf_len: usize,
    prev_ix: usize,
    last_len: usize,
    // Used to trigger updating FILENAME on the first read.
    ifmt: InputFormat,
    field_set: FieldSet,
}

impl LineReader for CSVReader<Box<dyn ChunkProducer<Chunk = OffsetChunk>>> {
    type Line = Line;
    fn filename(&self) -> Str<'static> {
        Str::from(self.cur_chunk.get_name()).unmoor()
    }
    fn request_handles(&self, size: usize) -> Vec<Box<dyn FnOnce() -> Self + Send>> {
        let producers = self.prod.try_dyn_resize(size);
        let mut res = Vec::with_capacity(producers.len());
        let ifmt = self.ifmt;
        for p_factory in producers.into_iter() {
            let field_set = self.field_set.clone();
            res.push(Box::new(move || CSVReader {
                prod: p_factory(),
                cur_chunk: OffsetChunk::default(),
                cur_buf: UniqueBuf::new(0).into_buf(),
                buf_len: 0,
                prev_ix: 0,
                last_len: 0,
                ifmt,
                field_set,
            }) as _)
        }
        res
    }
    fn read_line(&mut self, _pat: &Str, _rc: &mut RegexCache) -> Result<(bool, Line)> {
        let mut line = Line::default();
        let changed = self.read_line_reuse(_pat, _rc, &mut line)?;
        Ok((changed, line))
    }
    fn read_line_reuse<'a, 'b: 'a>(
        &'b mut self,
        _pat: &Str,
        _rc: &mut RegexCache,
        old: &'a mut Line,
    ) -> Result<bool> {
        Ok(self.read_line_inner(old)?)
    }
    fn read_state(&self) -> i64 {
        if self.cur_chunk.version != 0 && self.last_len == 0 {
            ReaderState::EOF as i64
        } else {
            ReaderState::OK as i64
        }
    }
    fn next_file(&mut self) -> Result<bool> {
        self.prod.next_file()
    }
    fn set_used_fields(&mut self, field_set: &FieldSet) {
        self.field_set = field_set.clone();
    }
}

// TODO: Pass in an ExecutioNStrategy to CSVReader and ByteReader.

impl CSVReader<Box<dyn ChunkProducer<Chunk = OffsetChunk>>> {
    // TODO: make a new enum called ParallelStrategy and pass that in here?
    pub fn new(
        r: impl Read + Send + 'static,
        ifmt: InputFormat,
        chunk_size: usize,
        name: impl Borrow<str>,
        exec_strategy: ExecutionStrategy,
    ) -> Self {
        let prod: Box<dyn ChunkProducer<Chunk = OffsetChunk>> = match exec_strategy {
            ExecutionStrategy::Serial => Box::new(chunk::new_offset_chunk_producer_csv(
                r,
                chunk_size,
                name.borrow(),
                ifmt,
                /*version=*/ 1,
            )),
            x @ ExecutionStrategy::ShardPerRecord => {
                let s = String::from(name.borrow());
                Box::new(ParallelChunkProducer::new(
                    move || {
                        chunk::new_offset_chunk_producer_csv(
                            r,
                            chunk_size,
                            s.as_str(),
                            ifmt,
                            /*version=*/ 1,
                        )
                    },
                    x.num_workers() * 2,
                ))
            }
            ExecutionStrategy::ShardPerFile => unimplemented!(),
        };
        CSVReader {
            prod,
            cur_buf: UniqueBuf::new(0).into_buf(),
            buf_len: 0,
            cur_chunk: OffsetChunk::default(),
            prev_ix: 0,
            last_len: 0,
            field_set: FieldSet::all(),
            ifmt,
        }
    }
}

// TODO rename as it handles CSV and TSV
impl<P: ChunkProducer<Chunk = OffsetChunk>> CSVReader<P> {
    fn refresh_buf(&mut self) -> Result<(/*is eof*/ bool, /* file changed */ bool)> {
        let prev_version = self.cur_chunk.version;
        if self.prod.get_chunk(&mut self.cur_chunk)? {
            return Ok((true, false));
        }
        self.cur_buf = self.cur_chunk.buf.take().unwrap().into_buf();
        self.buf_len = self.cur_chunk.len;
        self.prev_ix = 0;
        Ok((false, prev_version != self.cur_chunk.version))
    }

    fn stepper<'a, 'b: 'a>(&'b mut self, st: State, line: &'a mut Line) -> Stepper<'a> {
        Stepper {
            buf: &self.cur_buf,
            buf_len: self.buf_len,
            off: &mut self.cur_chunk.off,
            prev_ix: self.prev_ix,
            ifmt: self.ifmt,
            field_set: self.field_set.clone(),
            line,
            st,
        }
    }
    pub fn read_line_inner<'a, 'b: 'a>(
        &'b mut self,
        line: &'a mut Line,
    ) -> Result</*file changed*/ bool> {
        line.clear();
        let mut changed = false;
        if self.cur_chunk.off.start == self.cur_chunk.off.fields.len() {
            // NB: see comment on corresponding condition in ByteReader.
            let (is_eof, has_changed) = self.refresh_buf()?;
            changed = has_changed;
            // NB: >= because the `push_past` logic in stepper can result in prev_ix pointing two
            // past the end of the buffer.
            if is_eof && self.prev_ix >= self.buf_len {
                self.last_len = 0;
                debug_assert!(!changed);
                return Ok(false);
            }
        }

        let (prev_ix, st) = {
            let mut stepper = self.stepper(State::Init, line);
            (unsafe { stepper.step() }, stepper.st)
        };
        let consumed = prev_ix - self.prev_ix;
        self.prev_ix = prev_ix;
        self.last_len = consumed;
        if st != State::Done {
            line.promote();
        }
        Ok(changed)
    }
}

#[derive(Default, Debug)]
pub struct Offsets {
    pub start: usize,
    // NB We're using u64s to potentially handle huge input streams.
    // An alternative option would be to save lines and fields in separate vectors, but this is
    // more efficient for iteration
    pub fields: Vec<u64>,
}

#[derive(Default, Clone, Debug)]
pub struct Line {
    raw: Str<'static>,
    // Why is len a separate value from raw? We don't always populate raw, but we need to report
    // line lengths to maintain some invariants from the Reader type in splitter.rs.
    len: usize,
    fields: Vec<Str<'static>>,
    partial: Str<'static>,
}

impl Line {
    pub fn len(&self) -> usize {
        self.len
    }
}

impl<'a> super::Line<'a> for Line {
    fn join_cols<F>(
        &mut self,
        start: Int,
        end: Int,
        sep: &Str<'a>,
        nf: usize,
        trans: F,
    ) -> Result<Str<'a>>
    where
        F: FnMut(Str<'static>) -> Str<'static>,
    {
        debug_assert_eq!(self.fields.len(), nf);
        let (start, end) = normalize_join_indexes(start, end, nf)?;
        let sep = sep.clone().unmoor();
        Ok(sep
            .join(self.fields[start..end].iter().cloned().map(trans))
            .upcast())
    }
    fn nf(&mut self, _pat: &Str, _rc: &mut super::RegexCache) -> Result<usize> {
        Ok(self.fields.len())
    }

    fn get_col(
        &mut self,
        col: super::Int,
        _pat: &Str,
        _ofs: &Str,
        _rc: &mut super::RegexCache,
    ) -> Result<Str<'a>> {
        if col == 0 {
            return Ok(self.raw.clone().upcast());
        }
        if col < 0 {
            return err!("attempt to access negative index {}", col);
        }
        Ok(self
            .fields
            .get(col as usize - 1)
            .cloned()
            .unwrap_or_else(Str::default)
            .upcast())
    }

    // Setting columns for CSV doesn't work. We refuse it outright.
    fn set_col(
        &mut self,
        _col: super::Int,
        _s: &Str<'a>,
        _pat: &Str,
        _rc: &mut super::RegexCache,
    ) -> Result<()> {
        Ok(())
    }
}

impl Line {
    pub fn promote(&mut self) {
        let partial = mem::replace(&mut self.partial, Str::default());
        self.fields.push(partial);
    }
    pub fn promote_null(&mut self) {
        debug_assert_eq!(self.partial, Str::default());
        self.fields.push(Str::default());
    }
    pub fn clear(&mut self) {
        self.fields.clear();
        self.partial = Str::default();
        self.raw = Str::default();
        self.len = 0;
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum State {
    Init,
    BS,
    Quote,
    QuoteInQuote,
    Done,
}

// Stepper implements the core splitting algorithm given a "chunk" of offsets computed by a
// CSVReader. The [step] method implements a basic state machine going through the control
// characters extracted by initial pass.
pub struct Stepper<'a> {
    pub ifmt: InputFormat,
    pub buf: &'a Buf,
    pub buf_len: usize,
    pub off: &'a mut Offsets,
    pub prev_ix: usize,
    pub st: State,
    pub line: &'a mut Line,
    pub field_set: FieldSet,
}

impl<'a> Stepper<'a> {
    fn append(&mut self, s: Str<'static>) {
        let partial = mem::replace(&mut self.line.partial, Str::default());
        self.line.partial = Str::concat(partial, s);
    }

    unsafe fn append_slice(&mut self, i: usize, j: usize) {
        self.append(self.buf.slice_to_str(i, j));
    }

    unsafe fn push_past(&mut self, i: usize) {
        self.append_slice(self.prev_ix, i);
        self.prev_ix = i + 1;
    }

    pub fn promote_null(&mut self) {
        self.line.promote_null();
    }
    pub fn promote(&mut self) {
        self.line.promote();
    }

    fn get(&mut self, line_start: usize, j: usize, cur: usize) -> usize {
        self.off.start = cur;
        if self.field_set.get(0) {
            self.line.raw = unsafe { self.buf.slice_to_str(line_start, j) };
        }
        self.line.len += j - line_start;
        self.prev_ix
    }

    pub unsafe fn step(&mut self) -> usize {
        let sep = self.ifmt.sep();
        let line_start = self.prev_ix;
        let bs = &self.buf.as_bytes()[0..self.buf_len];
        let mut cur = self.off.start;
        let bs_transition = match self.ifmt {
            // Escape sequences only occur within quotes for CSV-formatted data.
            InputFormat::CSV => State::Quote,
            // There are no "quoted fields" in TSV, and escape sequences simply occur at any point
            // in a field.
            InputFormat::TSV => State::Init,
        };
        macro_rules! get_next {
            () => {
                if cur == self.off.fields.len() {
                    self.push_past(bs.len());
                    return self.get(line_start, bs.len(), cur);
                } else {
                    let res = *self.off.fields.get_unchecked(cur) as usize;
                    cur += 1;
                    res
                }
            };
        }
        'outer: loop {
            match self.st {
                State::Init => 'init: loop {
                    // First, skip over any unused fields.
                    let cur_field = self.line.fields.len() + 1;
                    if !self.field_set.get(cur_field) {
                        loop {
                            if cur == self.off.fields.len() {
                                self.prev_ix = bs.len() + 1;
                                return self.get(line_start, bs.len(), cur);
                            }
                            let ix = *self.off.fields.get_unchecked(cur) as usize;
                            cur += 1;
                            match *bs.get_unchecked(ix) {
                                b'\r' | b'"' | b'\\' => {}
                                b'\n' => {
                                    self.prev_ix = ix + 1;
                                    self.promote_null();
                                    self.st = State::Done;
                                    return self.get(line_start, ix, cur);
                                }
                                _x => {
                                    debug_assert_eq!(_x, sep);
                                    self.prev_ix = ix + 1;
                                    self.promote_null();
                                    continue 'init;
                                }
                            }
                        }
                    }
                    // Common case: Loop through records until the end of the line.
                    let ix = get_next!();
                    match *bs.get_unchecked(ix) {
                        b'\r' => {
                            self.push_past(ix);
                            continue;
                        }
                        b'\n' => {
                            self.push_past(ix);
                            self.promote();
                            self.st = State::Done;
                            return self.get(line_start, ix, cur);
                        }
                        b'"' => {
                            self.push_past(ix);
                            self.st = State::Quote;
                            continue 'outer;
                        }
                        // Only happens in TSV mode
                        b'\\' => {
                            self.push_past(ix);
                            self.st = State::BS;
                            continue 'outer;
                        }
                        _x => {
                            debug_assert_eq!(_x, sep);
                            self.push_past(ix);
                            self.promote();
                            continue;
                        }
                    }
                },
                State::Quote => {
                    // Parse a quoted field; this will only happen in CSV mode.
                    let ix = get_next!();
                    match *bs.get_unchecked(ix) {
                        b'"' => {
                            // We have found a quote, time to figure out if the next character is a
                            // quote, or if it is the end of the quoted portion of the field.
                            //
                            // One interesting thing to note here is that this allows for a mixture
                            // of quoted and unquoted portions of a single CSV field, which is
                            // technically more than is supported by the standard IIUC.
                            self.push_past(ix);
                            self.st = State::QuoteInQuote;
                            continue;
                        }
                        b'\\' => {
                            // A similar lookahead case: handling escaped sequences.
                            self.push_past(ix);
                            self.st = State::BS;
                            continue;
                        }
                        _ => unreachable!(),
                    }
                }
                State::QuoteInQuote => {
                    // We've just seen a " inside a ", it could be the end of the quote, or it
                    // could be an escaped quote. We peek ahead one character and check.
                    if bs.len() == self.prev_ix {
                        // We are past the end! Let's pick this up later.
                        // We had better not have any more offsets in the stream!
                        debug_assert_eq!(self.off.fields.len(), cur);
                        return self.get(line_start, bs.len(), cur);
                    }
                    if *bs.get_unchecked(self.prev_ix) == b'"' {
                        self.append("\"".into());
                        self.st = State::Quote;
                        // burn the next entry. It should be a quote. Using get_next here is a
                        // convenience: if we hit the branch that returns early within the macro,
                        // there's a bug somewhere! But there shouldn't be one, because all quotes
                        // should appear in the offsets vector, and we know that there is more
                        // space in `bs`.
                        let _q = get_next!();
                        debug_assert_eq!(bs[_q], b'"');
                        self.prev_ix += 1;
                    } else {
                        self.st = State::Init;
                    }
                }
                State::BS => {
                    if bs.len() == self.prev_ix {
                        debug_assert_eq!(self.off.fields.len(), cur);
                        return self.get(line_start, bs.len(), cur);
                    }
                    match *bs.get_unchecked(self.prev_ix) {
                        b'n' => self.append("\n".into()),
                        b't' => self.append("\t".into()),
                        b'\\' => self.append("\\".into()),
                        x => {
                            let buf = &[x];
                            let s: Str<'static> = Str::concat(
                                "\\".into(),
                                Str::from(str::from_utf8_unchecked(buf)).unmoor(),
                            );
                            self.append(s);
                        }
                    }
                    self.prev_ix += 1;
                    self.st = bs_transition;
                }
                State::Done => panic!("cannot start in Done state"),
            }
        }
    }
}

#[derive(Copy, Clone)]
pub enum InputFormat {
    CSV,
    TSV,
}

impl InputFormat {
    fn sep(self) -> u8 {
        match self {
            InputFormat::CSV => ',' as u8,
            InputFormat::TSV => '\t' as u8,
        }
    }
}

// get_find_indexes{_bytes}, what's that all about?
//
// These functions use vector instructions that, while commonly supported on x86, are occasionally
// missing. The safest way to handle this fact is to query _at runtime_ whether or not a given
// feature-set is supported. To avoid querying this on every function call, the calling library
// will instead store a function pointer that is computed at startup based on the dynamically
// available CPU features.

pub fn get_find_indexes(
    ifmt: InputFormat,
) -> unsafe fn(&[u8], &mut Offsets, u64, u64) -> (u64, u64) {
    #[cfg(target_arch = "x86_64")]
    const IS_X64: bool = true;
    #[cfg(not(target_arch = "x86_64"))]
    const IS_X64: bool = false;
    #[cfg(feature = "allow_avx2")]
    const ALLOW_AVX2: bool = true;
    #[cfg(not(feature = "allow_avx2"))]
    const ALLOW_AVX2: bool = false;
    assert!(IS_X64, "CSV is only supported on x86_64 machines");

    if ALLOW_AVX2 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("pclmulqdq") {
        match ifmt {
            InputFormat::CSV => generic::find_indexes_csv::<avx2::Impl>,
            InputFormat::TSV => generic::find_indexes_tsv::<avx2::Impl>,
        }
    } else if is_x86_feature_detected!("sse2") && is_x86_feature_detected!("pclmulqdq") {
        match ifmt {
            InputFormat::CSV => generic::find_indexes_csv::<sse2::Impl>,
            InputFormat::TSV => generic::find_indexes_tsv::<sse2::Impl>,
        }
    } else {
        // TODO write a simple fallback implementation of Vector for non-x86
        panic!("CSV requires at least SSE2 and PCLMULQDQ support");
    }
}

pub fn get_find_indexes_bytes() -> Option<unsafe fn(&[u8], &mut Offsets, u8, u8)> {
    #[cfg(target_arch = "x86_64")]
    const IS_X64: bool = true;
    #[cfg(not(target_arch = "x86_64"))]
    const IS_X64: bool = false;
    #[cfg(feature = "allow_avx2")]
    const ALLOW_AVX2: bool = true;
    #[cfg(not(feature = "allow_avx2"))]
    const ALLOW_AVX2: bool = false;
    assert!(IS_X64, "CSV is only supported on x86_64 machines");

    if ALLOW_AVX2 && is_x86_feature_detected!("avx2") {
        Some(generic::find_indexes_byte::<avx2::Impl>)
    } else if is_x86_feature_detected!("sse2") {
        Some(generic::find_indexes_byte::<sse2::Impl>)
    } else {
        // TODO writing a fallback implementation of this function would be pretty easy.
        None
    }
}

// TODO: consider putting these into the runtime struct to avoid the extra indirection.
lazy_static! {
    static ref QUOTE: Regex = Regex::new(r#"""#).unwrap();
    static ref TAB: Regex = Regex::new(r#"\t"#).unwrap();
    static ref NEWLINE: Regex = Regex::new(r#"\n"#).unwrap();
    static ref NEEDS_ESCAPE_TSV: bytes::RegexSet =
        bytes::RegexSet::new(&[r#"\t"#, r#"\n"#]).unwrap();
    static ref NEEDS_ESCAPE_CSV: bytes::RegexSet =
        bytes::RegexSet::new(&[r#"""#, r#"\t"#, r#"\n"#, ","]).unwrap();
}

pub fn escape_csv<'a>(s: &Str<'a>) -> Str<'a> {
    let bs = unsafe { &*s.get_bytes() };
    let matches = NEEDS_ESCAPE_CSV.matches(bs);
    if !matches.matched_any() {
        return s.clone();
    }
    let mut cur = s.clone();
    for m in matches.into_iter() {
        let (pat, subst_for) = match m {
            0 => (&*QUOTE, r#""""#),
            1 => (&*TAB, r#"\t"#),
            2 => (&*NEWLINE, r#"\n"#),
            // This just necessitates the ""s
            3 => continue,
            _ => unreachable!(),
        };
        cur = cur.subst_all(pat, &Str::from(subst_for).upcast()).0;
    }
    let quote = Str::from("\"");
    Str::concat(Str::concat(quote.clone(), cur), quote)
}

pub fn escape_tsv<'a>(s: &Str<'a>) -> Str<'a> {
    let bs = unsafe { &*s.get_bytes() };
    let matches = NEEDS_ESCAPE_TSV.matches(bs);
    if !matches.matched_any() {
        return s.clone();
    }
    let mut cur = s.clone();
    for m in matches.into_iter() {
        let (pat, subst_for) = match m {
            0 => (&*TAB, r#"\t"#),
            1 => (&*NEWLINE, r#"\n"#),
            _ => unreachable!(),
        };
        cur = cur.subst_all(pat, &Str::from(subst_for).upcast()).0;
    }
    cur
}

#[cfg(test)]
mod escape_tests {
    use super::*;

    #[test]
    fn csv_escaping() {
        let s1 = Str::from("no escaping");
        let s2 = Str::from("This ought to be escaped, for two\treasons");
        assert_eq!(escape_csv(&s1), s1);
        assert_eq!(
            escape_csv(&s2),
            Str::from(r#""This ought to be escaped, for two\treasons""#)
        );
    }

    #[test]
    fn tsv_escaping() {
        let s1 = Str::from("no, escaping");
        let s2 = Str::from("This ought to be escaped, for one\treason");
        assert_eq!(escape_tsv(&s1), s1);
        assert_eq!(
            escape_tsv(&s2),
            Str::from(r#"This ought to be escaped, for one\treason"#)
        );
    }
}

mod generic {
    use super::Offsets;
    const MAX_INPUT_SIZE: usize = 64;

    pub trait Vector {
        const VEC_BYTES: usize;
        const INPUT_SIZE: usize = Self::VEC_BYTES * 2;
        const _ASSERT_LTE_MAX: usize = MAX_INPUT_SIZE - Self::INPUT_SIZE;
        type Input: Copy;

        // Precondition: bptr points to at least INPUT_SIZE bytes.
        unsafe fn fill_input(btr: *const u8) -> Self::Input;

        // Compute a mask of which bits in input match (bytewise) `m`.
        unsafe fn cmp_mask_against_input(inp: Self::Input, m: u8) -> u64;

        unsafe fn find_quote_mask(
            inp: Self::Input,
            prev_iter_inside_quote: &mut u64,
        ) -> (/*inside quotes*/ u64, /*quote locations*/ u64);
    }

    #[cfg(target_arch = "x86_64")]
    pub unsafe fn default_x86_find_quote_mask<V: Vector>(
        inp: V::Input,
        prev_iter_inside_quote: &mut u64,
    ) -> (/*inside quotes*/ u64, /*quote locations*/ u64) {
        use std::arch::x86_64::*;
        // This is about finding a mask that has 1s for all characters inside a quoted pair, plus
        // the starting quote, but not the ending one. For example:
        // [unquoted text "quoted text"]
        // has the mask
        // [000000000000001111111111110]
        // We will use this mask to avoid splitting on commas that are inside a quoted field. We
        // start by generating a mask for all the quote characters appearing in the string.
        let quote_bits = V::cmp_mask_against_input(inp, '"' as u8);
        // Then we pull this trick from the simdjson paper. Lets use the example from the comments
        // above:
        // [unquoted text "quoted text"]
        // quoted_bits is going to be
        // [000000000000001000000000001]
        // We want to fill anything between the ones with ones. One way to do this is to replace
        // bit i with the prefix sum mod 2 (ie xor) of all the bits up to that point. We could do
        // this by repeated squaring, but intel actually just gives us this instruction for free:
        // it's "carryless multiplication" by all 1s. Caryless multiplication takes the prefix xor
        // of a each bit up to bit i, and-ed by corresponding bits in the other operand; anding
        // with all 1s gives us the sum we want.
        let mut quote_mask =
            _mm_cvtsi128_si64(/* grab the lower 64 bits */ _mm_clmulepi64_si128(
                /*pad with zeros*/ _mm_set_epi64x(0i64, quote_bits as i64),
                /*all 1s*/ _mm_set1_epi8(!0),
                0,
            )) as u64;
        let prev = *prev_iter_inside_quote;
        // We need to invert the mask if we started off inside a quote.
        quote_mask ^= prev;
        // We want all 1s if we ended in a quote, all zeros if not
        *prev_iter_inside_quote = (quote_mask as i64).wrapping_shr(63) as u64;
        (quote_mask, quote_bits)
    }

    #[inline(always)]
    unsafe fn flatten_bits(base_p: *mut u64, start_offset: &mut u64, start_ix: u64, mut bits: u64) {
        // We want to write indexes of set bits into the array base_p.
        // start_offset is the initial offset into base_p where we start writing.
        // start_ix is the index into the corpus pointed to by the initial bits of `bits`.
        if bits == 0 {
            return;
        }
        let count = bits.count_ones();
        let next_start = (*start_offset) + count as u64;
        // Unroll the loop for the first 8 bits. As in simdjson, we are potentially "overwriting"
        // here. We do not know if we have 8 bits to write, but we are writing them anyway! This is
        // "safe" so long as we have enough space in the array pointed to by base_p because:
        // * Excessive writes will only write a zero
        // * we will set next start to point to the right location.
        // Why do the extra work? We want to avoid extra branches.
        macro_rules! write_offset_inner {
            ($ix:expr) => {
                *base_p.offset(*start_offset as isize + $ix) =
                    start_ix + bits.trailing_zeros() as u64;
                bits = bits & bits.wrapping_sub(1);
            };
        }
        macro_rules! write_offsets {
             ($($ix:expr),*) => { $(write_offset_inner!($ix);)* };
        }
        write_offsets!(0, 1, 2, 3, 4, 5, 6, 7);
        // Similarly for bits 8->16
        if count > 8 {
            write_offsets!(8, 9, 10, 11, 12, 13, 14, 15);
        }
        // Just do a loop for the last 48 bits.
        if count > 16 {
            *start_offset += 16;
            loop {
                write_offsets!(0);
                if bits == 0 {
                    break;
                }
                *start_offset += 1;
            }
        }
        *start_offset = next_start;
    }

    // The find_indexes functions are progressively simple takes on "do a vectorized comparison
    // against a byte sequence, write the indexes of matching indexes into Offsets." The first is
    // very close to the simd-csv variant; simpler formats do a bit less.

    pub unsafe fn find_indexes_csv<V: Vector>(
        buf: &[u8],
        offsets: &mut Offsets,
        mut prev_iter_inside_quote: u64, /*start at 0*/
        mut prev_iter_cr_end: u64,       /*start at 0*/
    ) -> (u64, u64) {
        offsets.fields.clear();
        offsets.start = 0;
        // This may cause us to overuse memory, but it's a safe upper bound and the plan is to
        // reuse this across different chunks.
        offsets.fields.reserve(buf.len());
        let buf_ptr = buf.as_ptr();
        let len = buf.len();
        let len_minus_64 = len.saturating_sub(V::INPUT_SIZE);
        let mut ix = 0;
        let base_ptr: *mut u64 = offsets.fields.get_unchecked_mut(0);
        let mut base = 0;

        // For ... reasons (better pipelining? better cache behavior?) we decode in blocks.
        const BUFFER_SIZE: usize = 4;
        macro_rules! iterate {
            ($buf:expr) => {{
                std::intrinsics::prefetch_read_data($buf.offset(128), 3);
                // find commas not inside quotes
                let inp = V::fill_input($buf);
                let (quote_mask, quote_locs) = V::find_quote_mask(inp, &mut prev_iter_inside_quote);
                let sep = V::cmp_mask_against_input(inp, ',' as u8);
                let esc = V::cmp_mask_against_input(inp, '\\' as u8);

                let cr = V::cmp_mask_against_input(inp, 0x0d);
                let cr_adjusted = cr.wrapping_shl(1) | prev_iter_cr_end;
                let lf = V::cmp_mask_against_input(inp, 0x0a);
                // Allow for either \r\n or \n.
                let end = (lf & cr_adjusted) | lf;
                prev_iter_cr_end = cr.wrapping_shr(63);
                (((end | sep | cr) & !quote_mask) | (esc & quote_mask) | quote_locs)
            }};
        }
        if len_minus_64 > V::INPUT_SIZE * BUFFER_SIZE {
            let mut fields = [0u64; BUFFER_SIZE];
            while ix < len_minus_64 - V::INPUT_SIZE * BUFFER_SIZE + 1 {
                for b in 0..BUFFER_SIZE {
                    fields[b] = iterate!(buf_ptr.offset((V::INPUT_SIZE * b + ix) as isize));
                }
                for b in 0..BUFFER_SIZE {
                    let internal_ix = V::INPUT_SIZE * b + ix;
                    flatten_bits(base_ptr, &mut base, internal_ix as u64, fields[b]);
                }
                ix += V::INPUT_SIZE * BUFFER_SIZE;
            }
        }
        // Do an unbuffered version for the remaining data
        while ix < len_minus_64 {
            let field_sep = iterate!(buf_ptr.offset(ix as isize));
            flatten_bits(base_ptr, &mut base, ix as u64, field_sep);
            ix += V::INPUT_SIZE;
        }
        // For any text that remains, just copy the results to the stack with some padding and do
        // one more iteration.
        let remaining = len - ix;
        if remaining > 0 {
            let mut rest = [0u8; MAX_INPUT_SIZE];
            std::ptr::copy_nonoverlapping(
                buf_ptr.offset(ix as isize),
                rest.as_mut_ptr(),
                remaining,
            );
            let field_sep = iterate!(rest.as_mut_ptr());
            flatten_bits(base_ptr, &mut base, ix as u64, field_sep);
        }
        offsets.fields.set_len(base as usize);
        (prev_iter_inside_quote, prev_iter_cr_end)
    }

    pub unsafe fn find_indexes_unquoted<V: Vector, F: Fn(*const u8) -> u64>(
        buf: &[u8],
        offsets: &mut Offsets,
        f: F,
    ) -> (u64, u64) {
        offsets.fields.clear();
        offsets.start = 0;
        // This may cause us to overuse memory, but it's a safe upper bound and the plan is to
        // reuse this across different chunks.
        offsets.fields.reserve(buf.len());
        let buf_ptr = buf.as_ptr();
        let len = buf.len();
        let len_minus_64 = len.saturating_sub(V::INPUT_SIZE);
        let mut ix = 0;
        let base_ptr: *mut u64 = offsets.fields.get_unchecked_mut(0);
        let mut base = 0;

        // For ... reasons (better pipelining? better cache behavior?) we decode in blocks.
        const BUFFER_SIZE: usize = 4;
        macro_rules! iterate {
            ($buf:expr) => {{
                // TODO/XXX Padding is only going to be 32 bytes ... what happens when you prefetch
                // an invalid address? It may slow things down, not to mention we may be giving the
                // compiler the wrong idea.
                //
                // Either increase padding, or decrease the prefetch interval.
                // find commas not inside quotes
                std::intrinsics::prefetch_read_data($buf.offset(128), 3);
                f($buf)
            }};
        }
        if len_minus_64 > V::INPUT_SIZE * BUFFER_SIZE {
            let mut fields = [0u64; BUFFER_SIZE];
            while ix < len_minus_64 - V::INPUT_SIZE * BUFFER_SIZE + 1 {
                for b in 0..BUFFER_SIZE {
                    fields[b] = iterate!(buf_ptr.offset((V::INPUT_SIZE * b + ix) as isize));
                }
                for b in 0..BUFFER_SIZE {
                    let internal_ix = V::INPUT_SIZE * b + ix;
                    flatten_bits(base_ptr, &mut base, internal_ix as u64, fields[b]);
                }
                ix += V::INPUT_SIZE * BUFFER_SIZE;
            }
        }
        // Do an unbuffered version for the remaining data
        while ix < len_minus_64 {
            let field_sep = iterate!(buf_ptr.offset(ix as isize));
            flatten_bits(base_ptr, &mut base, ix as u64, field_sep);
            ix += V::INPUT_SIZE;
        }
        // For any text that remains, just copy the results to the stack with some padding and do
        // one more iteration.
        let remaining = len - ix;
        if remaining > 0 {
            let mut rest = [0u8; MAX_INPUT_SIZE];
            std::ptr::copy_nonoverlapping(
                buf_ptr.offset(ix as isize),
                rest.as_mut_ptr(),
                remaining,
            );
            let field_sep = iterate!(rest.as_mut_ptr());
            flatten_bits(base_ptr, &mut base, ix as u64, field_sep);
        }
        offsets.fields.set_len(base as usize);
        (0, 0)
    }

    pub unsafe fn find_indexes_tsv<V: Vector>(
        buf: &[u8],
        offsets: &mut Offsets,
        // These two are ignored for TSV
        _prev_iter_inside_quote: u64,
        _prev_iter_cr_end: u64,
    ) -> (u64, u64) {
        find_indexes_unquoted::<V, _>(buf, offsets, |ptr| {
            let inp = V::fill_input(ptr);
            let sep = V::cmp_mask_against_input(inp, '\t' as u8);
            let esc = V::cmp_mask_against_input(inp, '\\' as u8);
            let lf = V::cmp_mask_against_input(inp, '\n' as u8);
            sep | esc | lf
        });
        (0, 0)
    }

    pub unsafe fn find_indexes_byte<V: Vector>(
        buf: &[u8],
        offsets: &mut Offsets,
        field_sep: u8,
        record_sep: u8,
    ) {
        find_indexes_unquoted::<V, _>(buf, offsets, |ptr| {
            let inp = V::fill_input(ptr);
            let fs = V::cmp_mask_against_input(inp, field_sep);
            let rs = V::cmp_mask_against_input(inp, record_sep);
            fs | rs
        });
    }
}

#[cfg(target_arch = "x86_64")]
mod sse2 {
    use super::generic::{default_x86_find_quote_mask, Vector};
    // This is in a large part based on geofflangdale/simdcsv, which is an adaptation of some of
    // the techniques from the simdjson paper (by that paper's authors, it appears) to CSV parsing.
    use std::arch::x86_64::*;
    pub struct Impl;
    #[derive(Copy, Clone)]
    pub struct Input {
        lo: __m128i,
        hi: __m128i,
    }

    impl Vector for Impl {
        const VEC_BYTES: usize = 16;
        type Input = Input;
        #[inline(always)]
        unsafe fn fill_input(bptr: *const u8) -> Input {
            Input {
                lo: _mm_loadu_si128(bptr as *const _),
                hi: _mm_loadu_si128(bptr.offset(Self::VEC_BYTES as isize) as *const _),
            }
        }

        #[inline(always)]
        unsafe fn cmp_mask_against_input(inp: Input, m: u8) -> u64 {
            // Load the mask into all lanes.
            let mask = _mm_set1_epi8(m as i8);
            // Compare against lo and hi. Store them in a single u64.
            let cmp_res_0 = _mm_cmpeq_epi8(inp.lo, mask);
            let res_0 = _mm_movemask_epi8(cmp_res_0) as u32 as u64;
            let cmp_res_1 = _mm_cmpeq_epi8(inp.hi, mask);
            let res_1 = _mm_movemask_epi8(cmp_res_1) as u64;
            res_0 | (res_1 << Self::VEC_BYTES)
        }

        unsafe fn find_quote_mask(
            inp: Self::Input,
            prev_iter_inside_quote: &mut u64,
        ) -> (/*inside quotes*/ u64, /*quote locations*/ u64) {
            default_x86_find_quote_mask::<Self>(inp, prev_iter_inside_quote)
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::generic::{default_x86_find_quote_mask, Vector};
    // This is in a large part based on geofflangdale/simdcsv, which is an adaptation of some of
    // the techniques from the simdjson paper (by that paper's authors, it appears) to CSV parsing.
    use std::arch::x86_64::*;
    pub struct Impl;
    #[derive(Copy, Clone)]
    pub struct Input {
        lo: __m256i,
        hi: __m256i,
    }

    impl Vector for Impl {
        const VEC_BYTES: usize = 32;
        type Input = Input;
        #[inline(always)]
        unsafe fn fill_input(bptr: *const u8) -> Input {
            Input {
                lo: _mm256_loadu_si256(bptr as *const _),
                hi: _mm256_loadu_si256(bptr.offset(Self::VEC_BYTES as isize) as *const _),
            }
        }

        #[inline(always)]
        unsafe fn cmp_mask_against_input(inp: Input, m: u8) -> u64 {
            // Load the mask into all lanes.
            let mask = _mm256_set1_epi8(m as i8);
            // Compare against lo and hi. Store them in a single u64.
            let cmp_res_0 = _mm256_cmpeq_epi8(inp.lo, mask);
            let res_0 = _mm256_movemask_epi8(cmp_res_0) as u32 as u64;
            let cmp_res_1 = _mm256_cmpeq_epi8(inp.hi, mask);
            let res_1 = _mm256_movemask_epi8(cmp_res_1) as u64;
            res_0 | (res_1 << Self::VEC_BYTES)
        }
        unsafe fn find_quote_mask(
            inp: Self::Input,
            prev_iter_inside_quote: &mut u64,
        ) -> (/*inside quotes*/ u64, /*quote locations*/ u64) {
            default_x86_find_quote_mask::<Self>(inp, prev_iter_inside_quote)
        }
    }
}

pub fn bytereader_supported() -> bool {
    get_find_indexes_bytes().is_some()
}

pub struct ByteReader<P> {
    prod: P,
    cur_chunk: OffsetChunk,
    cur_buf: Buf,
    buf_len: usize,
    used_fields: FieldSet,
    // Progress in the current buffer.
    progress: usize,
    record_sep: u8,

    last_len: usize,
}

impl ByteReader<Box<dyn ChunkProducer<Chunk = OffsetChunk>>> {
    pub fn new(
        r: impl Read + 'static,
        field_sep: u8,
        record_sep: u8,
        chunk_size: usize,
        name: impl Borrow<str>,
    ) -> Self {
        ByteReader {
            prod: Box::new(chunk::new_offset_chunk_producer_bytes(
                r,
                chunk_size,
                name.borrow(),
                field_sep,
                record_sep,
                /*version=*/ 1,
            )),
            cur_chunk: OffsetChunk::default(),
            cur_buf: UniqueBuf::new(0).into_buf(),
            buf_len: 0,
            progress: 0,
            record_sep,
            used_fields: FieldSet::all(),
            last_len: usize::max_value(),
        }
    }
}

impl LineReader for ByteReader<Box<dyn ChunkProducer<Chunk = OffsetChunk>>> {
    type Line = DefaultLine;
    fn filename(&self) -> Str<'static> {
        Str::from(self.cur_chunk.get_name()).unmoor()
    }
    fn request_handles(&self, size: usize) -> Vec<Box<dyn FnOnce() -> Self + Send>> {
        let producers = self.prod.try_dyn_resize(size);
        let mut res = Vec::with_capacity(producers.len());
        for p_factory in producers.into_iter() {
            let used_fields = self.used_fields.clone();
            let record_sep = self.record_sep;
            res.push(Box::new(move || ByteReader {
                prod: p_factory(),
                cur_chunk: OffsetChunk::default(),
                cur_buf: UniqueBuf::new(0).into_buf(),
                buf_len: 0,
                progress: 0,
                record_sep,
                last_len: usize::max_value(),
                used_fields,
            }) as _)
        }
        res
    }
    fn read_line(&mut self, _pat: &Str, _rc: &mut RegexCache) -> Result<(bool, DefaultLine)> {
        let mut line = DefaultLine::default();
        let changed = self.read_line_reuse(_pat, _rc, &mut line)?;
        Ok((changed, line))
    }
    fn read_line_reuse<'a, 'b: 'a>(
        &'b mut self,
        _pat: &Str,
        _rc: &mut RegexCache,
        old: &'a mut DefaultLine,
    ) -> Result<bool> {
        let start = self.cur_chunk.version == 0;
        old.diverged = false;
        // We use the same protocol as DefaultSplitter, RegexSplitter. See comments for more info.
        if start {
            old.used_fields = self.used_fields.clone();
        } else if old.used_fields != self.used_fields {
            self.used_fields = old.used_fields.clone()
        }
        let mut old_fields = old.fields.get_cleared_vec();
        let changed = self.read_line_inner(&mut old.line, &mut old_fields)?;
        old.fields = LazyVec::from_vec(old_fields);
        Ok(changed)
    }
    fn read_state(&self) -> i64 {
        if self.cur_chunk.version != 0 && self.last_len == 0 {
            ReaderState::EOF as i64
        } else {
            ReaderState::OK as i64
        }
    }

    fn next_file(&mut self) -> Result<bool> {
        self.prod.next_file()
    }

    fn set_used_fields(&mut self, field_set: &FieldSet) {
        self.used_fields = field_set.clone();
    }
}

impl<P: ChunkProducer<Chunk = OffsetChunk>> ByteReader<P> {
    // This will panic if the current architecture does not support SIMD, etc. bytereader_supported
    // does this check.
    fn refresh_buf(&mut self) -> Result<(/* eof */ bool, /*file changed*/ bool)> {
        let prev_version = self.cur_chunk.version;
        if self.prod.get_chunk(&mut self.cur_chunk)? {
            return Ok((true, false));
        }
        self.cur_buf = self.cur_chunk.buf.take().unwrap().into_buf();
        self.buf_len = self.cur_chunk.len;
        self.progress = 0;
        Ok((false, prev_version != self.cur_chunk.version))
    }

    fn read_line_inner<'a, 'b: 'a>(
        &'b mut self,
        line: &'a mut Str<'static>,
        fields: &'a mut Vec<Str<'static>>,
    ) -> Result</* file changed */ bool> {
        let mut changed = false;
        if self.cur_chunk.off.start == self.cur_chunk.off.fields.len() {
            // What's going on with this second test? self.refresh_buf() returns Ok(true) if we
            // were unable to fetch more data due to an EOF. The last execution consumed buffer up
            // to the last record separator in the input, but there may be remaining data filling
            // up the rest of the buffer with no more record or field separators. In that case, we
            // want to return the rest of the input as a single-field record, which one more
            // `consume_line` will indeed accomplish.
            let (is_eof, has_changed) = self.refresh_buf()?;
            changed = has_changed;
            if is_eof && self.progress == self.buf_len {
                *line = Str::default();
                self.last_len = 0;
                debug_assert!(!changed);
                return Ok(false);
            }
        }
        let (next_line, consumed) = unsafe { self.consume_line(fields) };
        *line = next_line;
        self.last_len = consumed;
        Ok(changed)
    }

    unsafe fn consume_line<'a, 'b: 'a>(
        &'b mut self,
        fields: &'a mut Vec<Str<'static>>,
    ) -> (Str<'static>, /*bytes consumed */ usize) {
        let buf = &self.cur_buf;
        macro_rules! get_field {
            ($fld:expr, $start:expr, $end:expr) => {
                if self.used_fields.get($fld) {
                    buf.slice_to_str($start, $end)
                } else {
                    Str::default()
                }
            };
            ($index:expr) => {
                get_field!(fields.len() + 1, self.progress, $index)
            };
        }

        let line_start = self.progress;
        let bytes = &buf.as_bytes()[0..self.buf_len];
        let offs = &mut self.cur_chunk.off;
        for index in &offs.fields[offs.start..] {
            let index = *index as usize;
            debug_assert!(index < self.buf_len);

            offs.start += 1;
            let is_record_sep = *bytes.get_unchecked(index) == self.record_sep;
            fields.push(get_field!(index));
            self.progress = index + 1;
            if is_record_sep {
                let line = get_field!(0, line_start, index);
                return (line, self.progress - line_start);
            }
        }
        fields.push(get_field!(self.buf_len));
        self.progress = self.buf_len;
        let line = get_field!(0, line_start, self.buf_len);
        (line, self.buf_len - line_start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::LazyVec;
    fn smoke_test<V: generic::Vector>() {
        let text: &'static str = r#"This,is,"a line with a quoted, comma",and
unquoted,commas,"as well, including some long ones", and there we have it."#;
        let mut mem: Vec<u8> = text.as_bytes().iter().cloned().collect();
        mem.reserve(32);
        let mut offsets: Offsets = Default::default();
        let (in_quote, in_cr) =
            unsafe { generic::find_indexes_csv::<V>(&mem[..], &mut offsets, 0, 0) };
        assert_eq!(in_quote, 0);
        assert_eq!(in_cr, 0);
        assert_eq!(
            &offsets.fields[..],
            &[4, 7, 8, 36, 37, 41, 50, 57, 58, 92, 93],
            "offset_fields={:?}",
            offsets
                .fields
                .iter()
                .cloned()
                .map(|x| (x, mem[x as usize] as char))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn avx2_smoke_test() {
        if is_x86_feature_detected!("avx2") {
            smoke_test::<avx2::Impl>();
        }
    }

    #[test]
    fn sse2_smoke_test() {
        smoke_test::<sse2::Impl>();
    }
    fn read_to_vec<T: Clone + Default>(lv: &LazyVec<T>) -> Vec<T> {
        let mut res = Vec::with_capacity(lv.len());
        for i in 0..lv.len() {
            res.push(lv.get(i).unwrap_or_else(Default::default))
        }
        res
    }
    fn disp_vec(v: &Vec<Str>) -> String {
        format!(
            "{:?}",
            v.iter().map(|s| format!("{}", s)).collect::<Vec<String>>()
        )
    }

    fn tsv_split(corpus: &str) {
        // Replace spaces with tabs to get some traction here.
        let fs = b'\t';
        let rs = b'\n';
        let corpus = String::from(corpus).replace(" ", "\t");
        let mut _cache = RegexCache::default();
        let _pat = Str::default();
        let expected: Vec<Vec<Str<'static>>> = corpus
            .split(rs as char)
            .map(|line| {
                line.split(fs as char)
                    .map(|x| Str::from(x).unmoor())
                    .collect()
            })
            .collect();
        let mut got = Vec::with_capacity(corpus.len());
        let reader = std::io::Cursor::new(corpus);
        let mut reader = CSVReader::new(
            reader,
            InputFormat::TSV,
            /*chunk_size=*/ 512,
            "fake-stdin",
            ExecutionStrategy::Serial,
        );
        loop {
            let (_, line) = reader
                .read_line(&_pat, &mut _cache)
                .expect("failed to read line");
            if reader.read_state() != 1 {
                break;
            }
            got.push(line.fields.clone());
        }
        if got != expected {
            eprintln!(
                "test failed! got vector of length {}, expected {} lines",
                got.len(),
                expected.len()
            );
            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                eprintln!("===============");
                if g == e {
                    eprintln!("line {} matches", i);
                } else {
                    eprintln!("line {} has a mismatch", i);
                    eprintln!("got:  {}", disp_vec(g));
                    eprintln!("want: {}", disp_vec(e));
                }
            }
            panic!("test failed. See debug output");
        }
    }

    #[test]
    fn tsv_splitter_basic() {
        if !bytereader_supported() {
            return;
        }
        tsv_split(crate::test_string_constants::VIRGIL);
        tsv_split(crate::test_string_constants::PRIDE_PREJUDICE_CH2);
    }

    fn bytes_split(fs: u8, rs: u8, corpus: &'static str) {
        let mut _cache = RegexCache::default();
        let _pat = Str::default();
        let expected: Vec<Vec<Str<'static>>> = corpus
            .split(rs as char)
            .map(|line| {
                line.split(fs as char)
                    .map(|x| Str::from(x).unmoor())
                    .collect()
            })
            .collect();
        let reader = std::io::Cursor::new(corpus);
        let mut reader = ByteReader::new(reader, fs, rs, 1024, "fake-stdin");
        let mut got = Vec::with_capacity(corpus.len());
        loop {
            let (_, line) = reader
                .read_line(&_pat, &mut _cache)
                .expect("failed to read line");
            if reader.read_state() != 1 {
                break;
            }
            got.push(read_to_vec(&line.fields));
        }
        if got != expected {
            eprintln!(
                "test failed! got vector of length {}, expected {} lines",
                got.len(),
                expected.len()
            );
            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                eprintln!("===============");
                if g == e {
                    eprintln!("line {} matches", i);
                } else {
                    eprintln!("line {} has a mismatch", i);
                    eprintln!("got:  {}", disp_vec(g));
                    eprintln!("want: {}", disp_vec(e));
                }
            }
            panic!("test failed. See debug output");
        }
    }

    #[test]
    fn bytes_splitter_basic() {
        if !bytereader_supported() {
            return;
        }
        bytes_split(b' ', b'\n', crate::test_string_constants::VIRGIL);
        bytes_split(
            b' ',
            b'\n',
            crate::test_string_constants::PRIDE_PREJUDICE_CH2,
        );
    }

    #[test]
    fn bytes_splitter_no_linesep() {
        // Lots of fields, but line separator not present
        bytes_split(b' ', 0u8, crate::test_string_constants::PRIDE_PREJUDICE_CH2);
    }

    #[test]
    fn bytes_splitter_no_fieldsep() {
        // Many lines, lots of single fields
        bytes_split(
            0u8,
            b'\n',
            crate::test_string_constants::PRIDE_PREJUDICE_CH2,
        );
    }

    #[test]
    fn bytes_splitter_no_sep() {
        // One line, One field
        bytes_split(0u8, 0u8, crate::test_string_constants::PRIDE_PREJUDICE_CH2);
    }
}
