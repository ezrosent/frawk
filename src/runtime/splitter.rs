//! This module implements line reading in the style of AWK's getline. In particular, it has the
//! cumbersome API of letting you know if there was an error, or EOF, after the read has completed.
//!
//! In addition to this API, it also handles reading in chunks, with appropriate handling of UTF8
//! characters that cross chunk boundaries, or multi-chunk "lines".
use super::csv;
use super::str_impl::{Buf, Str, UniqueBuf};
use super::utf8::{is_utf8, validate_utf8_clipped};
use super::{Int, LazyVec, Line, LineReader, RegexCache};
use crate::common::Result;
use crate::pushdown::FieldSet;

use regex::Regex;
use smallvec::SmallVec;

use std::io::{ErrorKind, Read};
use std::str;

#[repr(i64)]
#[derive(PartialEq, Eq, Copy, Clone)]
pub(crate) enum ReaderState {
    ERROR = -1,
    EOF = 0,
    OK = 1,
}

pub struct RegexLine {
    line: Str<'static>,
    used_fields: FieldSet,
    fields: LazyVec<Str<'static>>,
    // Has someone assigned into `fields` without us regenerating `line`?
    // AWK lets you do
    //  $1 = "turnip"
    //  $2 = "rutabaga"
    //  print $0; # "turnip rutabaga ..."
    //
    // After that first line, we set diverged to true, so we know to regenerate $0 when $0 is asked
    // for. This speeds up cases where multiple fields are assigned in a row.
    diverged: bool,
}

impl Default for RegexLine {
    fn default() -> RegexLine {
        RegexLine {
            line: Str::default(),
            used_fields: FieldSet::all(),
            fields: LazyVec::new(),
            diverged: false,
        }
    }
}

impl RegexLine {
    fn split_if_needed(&mut self, pat: &Str, rc: &mut RegexCache) -> Result<()> {
        if self.fields.len() == 0 {
            rc.split_regex(pat, &self.line, &self.used_fields, &mut self.fields)?;
        }
        Ok(())
    }
}

impl<'a> Line<'a> for RegexLine {
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
        // Should have split before calling this function.
        debug_assert!(self.fields.len() > 0);
        let (start, end) = super::normalize_join_indexes(start, end, nf)?;
        Ok(self
            .fields
            .join_by(&sep.clone().unmoor(), start, end, trans)
            .upcast())
    }
    fn nf(&mut self, pat: &Str, rc: &mut RegexCache) -> Result<usize> {
        self.split_if_needed(pat, rc)?;
        Ok(self.fields.len())
    }
    fn get_col(&mut self, col: Int, pat: &Str, ofs: &Str, rc: &mut RegexCache) -> Result<Str<'a>> {
        if col < 0 {
            return err!("attempt to access field {}; field must be nonnegative", col);
        }
        let res = if col == 0 && !self.diverged {
            self.line.clone()
        } else if col == 0 && self.diverged {
            if self.used_fields != FieldSet::all() {
                // We projected out fields, but now we have set one of the interior fields and need
                // to print out $0. That means we have to split $0 in its entirety and then copy
                // over the fields that were already set.
                //
                // This is strictly more work than just reading all of the fields in the first
                // place; so once we hit this condition we overwrite the used fields with all() so
                // this doesn't happen again for a while.
                let old_set = std::mem::replace(&mut self.used_fields, FieldSet::all());
                let mut new_vec = LazyVec::new();
                rc.split_regex(pat, &self.line, &self.used_fields, &mut new_vec)?;
                for i in 0..new_vec.len() {
                    if old_set.get(i + 1) {
                        new_vec.insert(i, self.fields.get(i).unwrap_or_else(Str::default));
                    }
                }
                self.fields = new_vec;
            }
            let res = self.fields.join_all(&ofs.clone().unmoor());
            self.line = res.clone();
            self.diverged = false;
            res
        } else {
            self.split_if_needed(pat, rc)?;
            self.fields
                .get((col - 1) as usize)
                .unwrap_or_else(Str::default)
        };
        Ok(res.upcast())
    }
    fn set_col(&mut self, col: Int, s: &Str<'a>, pat: &Str, rc: &mut RegexCache) -> Result<()> {
        if col == 0 {
            self.line = s.clone().unmoor();
            self.fields.clear();
            return Ok(());
        }
        if col < 0 {
            return err!("attempt to access field {}; field must be nonnegative", col);
        }
        self.split_if_needed(pat, rc)?;
        self.fields.insert(col as usize - 1, s.clone().unmoor());
        self.diverged = true;
        Ok(())
    }
}

pub struct RegexSplitter<R> {
    reader: Reader<R>,
    name: Str<'static>,
    used_fields: FieldSet,
    // Used to trigger updating FILENAME on the first read.
    start: bool,
}

impl<R: Read> LineReader for RegexSplitter<R> {
    type Line = RegexLine;
    fn filename(&self) -> Str<'static> {
        self.name.clone()
    }

    // The _reuse variant not only allows us to reuse the memory in the `fields` vec, it also
    // allows us to reuse the old FieldSet, which may have been overwritten with all() if the more
    // expensive join path was taken.
    fn read_line_reuse<'a, 'b: 'a>(
        &'b mut self,
        pat: &Str,
        rc: &mut super::RegexCache,
        old: &'a mut Self::Line,
    ) -> Result<bool> {
        let start = self.start;
        if start {
            old.used_fields = self.used_fields.clone();
        }
        self.start = false;
        old.diverged = false;
        old.fields.clear();
        rc.with_regex(pat, |re| {
            old.line = self.read_line_regex(re);
        })?;
        Ok(/* file changed */ start)
    }

    fn read_line(&mut self, pat: &Str, rc: &mut super::RegexCache) -> Result<(bool, Self::Line)> {
        let start = self.start;
        self.start = false;
        let line = rc.with_regex(pat, |re| RegexLine {
            line: self.read_line_regex(re),
            fields: LazyVec::new(),
            used_fields: self.used_fields.clone(),
            diverged: false,
        })?;
        Ok((/* file changed */ start, line))
    }
    fn read_state(&self) -> i64 {
        self.reader.read_state()
    }
    fn next_file(&mut self) -> bool {
        // There is just one file. Set EOF.
        self.reader.force_eof();
        false
    }
    fn set_used_fields(&mut self, used_fields: &FieldSet) {
        self.used_fields = used_fields.clone();
    }
}

impl<R: Read> RegexSplitter<R> {
    pub fn new(r: R, chunk_size: usize, name: impl Into<Str<'static>>) -> Self {
        RegexSplitter {
            reader: Reader::new(r, chunk_size),
            name: name.into(),
            used_fields: FieldSet::all(),
            start: true,
        }
    }

    pub fn read_line_regex(&mut self, pat: &Regex) -> Str<'static> {
        // We keep this as a separate method because it helps in writing tests.
        let res = self.read_line_inner(pat);
        self.reader.last_len = res.len();
        res
    }

    fn read_line_inner(&mut self, pat: &Regex) -> Str<'static> {
        macro_rules! handle_err {
            ($e:expr) => {
                if let Ok(e) = $e {
                    e
                } else {
                    self.reader.state = ReaderState::ERROR;
                    return Str::default();
                }
            };
        }
        let mut prefix: Str<'static> = Default::default();
        if self.reader.is_eof() {
            return prefix;
        }
        self.reader.state = ReaderState::OK;
        loop {
            // Why this map invocation? Match objects hold a reference to the substring, which
            // makes it harder for us to call mutable methods like advance in the body, so just get
            // the start and end pointers.
            let s = unsafe {
                str::from_utf8_unchecked(
                    &self.reader.buf.as_bytes()[self.reader.start..self.reader.end],
                )
            };
            let start_offset = self.reader.start;
            match pat.find(s).map(|m| (m.start(), m.end())) {
                Some((start, end)) => {
                    // Valid offsets guaranteed by correctness of regex `find`.
                    let res = unsafe {
                        self.reader
                            .buf
                            .slice_to_str(start_offset, start_offset + start)
                    };
                    // NOTE if we get a read error here, then we will stop one line early.
                    // That seems okay, but we could find out that it actually isn't, in which case
                    // we would want some more complicated error handling here.
                    handle_err!(self.reader.advance(end));
                    return Str::concat(prefix, res);
                }
                None => {
                    // Valid offsets guaranteed by read_buf
                    let cur: Str =
                        unsafe { self.reader.buf.slice_to_str(start_offset, self.reader.end) };
                    let remaining = self.reader.remaining();
                    handle_err!(self.reader.advance(remaining));
                    prefix = Str::concat(prefix, cur);
                    if self.reader.is_eof() {
                        // All done! Just return the rest of the buffer.
                        return prefix;
                    }
                }
            }
        }
    }
}

pub struct CSVReader<R> {
    inner: Reader<R>,
    name: Str<'static>,
    cur_offsets: csv::Offsets,
    prev_ix: usize,
    prev_iter_inside_quote: u64,
    prev_iter_cr_end: u64,
    // Used to trigger updating FILENAME on the first read.
    start: bool,
    ifmt: csv::InputFormat,
    field_set: FieldSet,

    // This is a function pointer because we query the preferred instruction set at construction
    // time.
    find_indexes: unsafe fn(&[u8], &mut csv::Offsets, u64, u64) -> (u64, u64),
}

impl<R: Read> LineReader for CSVReader<R> {
    type Line = csv::Line;
    fn filename(&self) -> Str<'static> {
        self.name.clone()
    }
    fn read_line(&mut self, _pat: &Str, _rc: &mut super::RegexCache) -> Result<(bool, csv::Line)> {
        let mut line = csv::Line::default();
        let changed = self.read_line_reuse(_pat, _rc, &mut line)?;
        Ok((changed, line))
    }
    fn read_line_reuse<'a, 'b: 'a>(
        &'b mut self,
        _pat: &Str,
        _rc: &mut super::RegexCache,
        old: &'a mut csv::Line,
    ) -> Result<bool> {
        let start = self.start;
        self.start = false;
        self.read_line_inner(old)?;
        Ok(start)
    }
    fn read_state(&self) -> i64 {
        let res = self.inner.read_state();
        res
    }
    fn next_file(&mut self) -> bool {
        self.inner.force_eof();
        false
    }
    fn set_used_fields(&mut self, field_set: &FieldSet) {
        self.field_set = field_set.clone();
    }
}

// TODO rename as it handles CSV and TSV
impl<R: Read> CSVReader<R> {
    // TODO give this the same signature as RegexSplitter (passing in chunk size, or just making
    // two constructors)
    pub fn new(r: R, ifmt: csv::InputFormat, name: impl Into<Str<'static>>) -> Self {
        CSVReader {
            start: true,
            inner: Reader::new(r, super::CHUNK_SIZE),
            name: name.into(),
            cur_offsets: Default::default(),
            prev_ix: 0,
            prev_iter_inside_quote: 0,
            prev_iter_cr_end: 0,
            find_indexes: csv::get_find_indexes(ifmt),
            field_set: FieldSet::all(),
            ifmt,
        }
    }
    fn refresh_buf(&mut self) -> Result<bool> {
        // exhausted. Fetch a new `cur`.
        self.inner.advance(self.inner.remaining())?;
        if self.inner.is_eof() {
            return Ok(true);
        }
        let (next_iq, next_cre) = unsafe {
            (self.find_indexes)(
                &self.inner.buf.as_bytes()[self.inner.start..self.inner.end],
                &mut self.cur_offsets,
                self.prev_iter_inside_quote,
                self.prev_iter_cr_end,
            )
        };
        self.prev_iter_inside_quote = next_iq;
        self.prev_iter_cr_end = next_cre;
        Ok(false)
    }
    fn stepper<'a, 'b: 'a>(
        &'b mut self,
        st: csv::State,
        line: &'a mut csv::Line,
    ) -> csv::Stepper<'a> {
        csv::Stepper {
            buf: &self.inner.buf,
            buf_len: self.inner.end,
            off: &mut self.cur_offsets,
            prev_ix: self.prev_ix,
            ifmt: self.ifmt,
            field_set: self.field_set.clone(),
            line,
            st,
        }
    }
    pub fn read_line_inner<'a, 'b: 'a>(&'b mut self, mut line: &'a mut csv::Line) -> Result<()> {
        line.clear();
        let mut st = csv::State::Init;
        let mut prev_ix = self.prev_ix;
        loop {
            self.prev_ix = prev_ix;
            // TODO: should this be ==? We get failures in that case, but is that a bug?
            if self.prev_ix >= self.inner.remaining() {
                if self.refresh_buf()? {
                    // Out of space.
                    line.promote();
                    self.inner.last_len = line.len();
                    return Ok(());
                }
                self.prev_ix = 0;
            }
            let mut stepper = self.stepper(st, &mut line);
            prev_ix = unsafe { stepper.step() };
            if let csv::State::Done = stepper.st {
                self.prev_ix = prev_ix;
                self.inner.last_len = line.len();
                return Ok(());
            }
            st = stepper.st;
        }
    }
}

struct Reader<R> {
    inner: R,
    // The "stray bytes" that will be prepended to the next buffer.
    prefix: SmallVec<[u8; 8]>,
    // TODO we probably want this to be a Buf, not a Str, but Buf's API gives out bytes not
    // strings.
    buf: Buf,
    start: usize,
    end: usize,
    chunk_size: usize,
    state: ReaderState,
    last_len: usize,
}

fn read_to_slice(r: &mut impl Read, mut buf: &mut [u8]) -> Result<usize> {
    let mut read = 0;
    while buf.len() > 0 {
        match r.read(buf) {
            Ok(n) => {
                if n == 0 {
                    break;
                }
                buf = &mut buf[n..];
                read += n;
            }
            Err(e) => match e.kind() {
                ErrorKind::Interrupted => continue,
                ErrorKind::UnexpectedEof => {
                    break;
                }
                _ => return err!("read error {}", e),
            },
        }
    }
    Ok(read)
}
impl<R: Read> Reader<R> {
    pub(crate) fn new(r: R, chunk_size: usize) -> Self {
        let res = Reader {
            inner: r,
            prefix: Default::default(),
            buf: UniqueBuf::new(0).into_buf(),
            start: 0,
            end: 0,
            chunk_size,
            state: ReaderState::OK,
            last_len: 0,
        };
        res
    }

    fn remaining(&self) -> usize {
        self.end - self.start
    }

    pub(crate) fn is_eof(&self) -> bool {
        self.end == self.start && self.state == ReaderState::EOF
    }

    fn force_eof(&mut self) {
        self.start = self.end;
        self.state = ReaderState::EOF;
    }

    fn read_state(&self) -> i64 {
        match self.state {
            ReaderState::OK => self.state as i64,
            ReaderState::ERROR | ReaderState::EOF => {
                if self.last_len == 0 {
                    self.state as i64
                } else {
                    ReaderState::OK as i64
                }
            }
        }
    }

    fn advance(&mut self, n: usize) -> Result<()> {
        let len = self.end - self.start;
        if len > n {
            self.start += n;
            return Ok(());
        }
        if self.is_eof() {
            return Ok(());
        }

        let residue = n - len;
        let (next_buf, next_len) = self.get_next_buf()?;
        self.buf = next_buf;
        self.end = next_len;
        self.start = 0;
        self.advance(residue)
    }

    fn get_next_buf(&mut self) -> Result<(Buf, usize)> {
        // For CSV
        // TODO disable for regex-based splitting.
        const PADDING: usize = 32;
        let mut done = false;
        // NB: UniqueBuf fills the allocation with zeros.
        let mut data = UniqueBuf::new(self.chunk_size + PADDING);
        let mut bytes = &mut data.as_mut_bytes()[..self.chunk_size];
        for (i, b) in self.prefix.iter().cloned().enumerate() {
            bytes[i] = b;
        }
        let plen = self.prefix.len();
        self.prefix.clear();
        // Try to fill up the rest of `data` with new bytes.
        let bytes_read = plen + read_to_slice(&mut self.inner, &mut bytes[plen..])?;
        self.prefix.clear();
        if bytes_read != self.chunk_size {
            done = true;
            bytes = &mut bytes[..bytes_read];
        }

        // For the odd benchmark to measure the impact of utf8 validation
        const SKIP_UTF8: bool = false;

        let ulen = if SKIP_UTF8 {
            bytes.len()
        } else {
            let opt = if done {
                if is_utf8(bytes) {
                    Some(bytes.len())
                } else {
                    None
                }
            } else {
                validate_utf8_clipped(bytes)
            };
            if let Some(u) = opt {
                u
            } else {
                // Invalid utf8. Get the error.
                return match std::str::from_utf8(bytes) {
                    Ok(_) => err!("bug in UTF8 validation!"),
                    Err(e) => err!("invalid utf8: {}", e),
                };
            }
        };
        if !done && ulen != bytes_read {
            // We clipped a utf8 character at the end of the buffer. Add it to prefix.
            self.prefix.extend_from_slice(&bytes[ulen..]);
        }
        if done {
            self.state = ReaderState::EOF;
        }
        Ok((data.into_buf(), ulen))
    }
}

#[cfg(test)]
mod tests {
    // need to benchmark batched splitting vs. regular splitting to get a feel for things.
    extern crate test;
    use super::LineReader;
    use super::Str;
    use lazy_static::lazy_static;
    use regex::Regex;
    use test::{black_box, Bencher};
    lazy_static! {
        static ref STR: String = String::from_utf8(bytes(1 << 20, 0.001, 0.05)).unwrap();
        static ref LINE: Regex = Regex::new("\n").unwrap();
        static ref SPACE: Regex = Regex::new(" ").unwrap();
    }

    // Helps type inference along.
    fn ref_str<'a>(s: &'a str) -> Str<'a> {
        s.into()
    }

    #[test]
    fn test_line_split() {
        use std::io::Cursor;
        let chunk_size = 1 << 9;
        let bs: String = crate::test_string_constants::PRIDE_PREJUDICE_CH2.into();
        let c = Cursor::new(bs.clone());
        let mut rdr = super::RegexSplitter::new(c, chunk_size, "");
        let mut lines = Vec::new();
        while !rdr.reader.is_eof() {
            let line = rdr.read_line_regex(&*LINE).upcast();
            assert!(rdr.read_state() != -1);
            lines.push(line);
        }

        let expected: Vec<_> = LINE.split(bs.as_str()).map(ref_str).collect();
        if lines != expected {
            eprintln!("lines.len={}, expected.len={}", lines.len(), expected.len());
            for (i, (l, e)) in lines.iter().zip(expected.iter()).enumerate() {
                if l != e {
                    eprintln!("mismatch at index {}:\ngot={:?}\nwant={:?}", i, l, e);
                }
            }
            assert!(false, "lines do not match");
        }
    }

    #[test]
    fn test_clipped_chunk_split() {
        use std::io::Cursor;

        let corpus_size = 1 << 18;
        let chunk_size = 1 << 9;

        let multi_byte = "學";
        assert!(multi_byte.len() > 1);
        let mut bs = bytes(corpus_size, 0.001, 0.05);
        let start = chunk_size - 1;
        for (i, b) in multi_byte.as_bytes().iter().enumerate() {
            bs[start + i] = *b;
        }

        let s = String::from_utf8(bs).unwrap();
        let c = Cursor::new(s.clone());
        let mut rdr = super::RegexSplitter::new(c, chunk_size, "");
        let mut lines = Vec::new();
        while !rdr.reader.is_eof() {
            let line = rdr.read_line_regex(&*LINE).upcast();
            assert!(rdr.read_state() != -1);
            lines.push(line);
        }
        let expected: Vec<_> = LINE.split(s.as_str()).map(ref_str).collect();
        if lines != expected {
            eprintln!("lines.len={}, expected.len={}", lines.len(), expected.len());
            for (i, (l, e)) in lines.iter().zip(expected.iter()).enumerate() {
                if l != e {
                    eprintln!("mismatch at index {}:\ngot={:?}\nwant={:?}", i, l, e);
                }
            }
            assert!(false, "lines do not match");
        }
    }

    #[bench]
    fn bench_find_iter(b: &mut Bencher) {
        let bs = STR.as_str();
        let line = &*LINE;
        let space = &*SPACE;
        b.iter(|| {
            let mut lstart = 0;
            let mut cstart = 0;
            while lstart < bs.len() {
                let (start, end) = match line.find_at(bs, lstart) {
                    Some(m) => (m.start(), m.end()),
                    None => (bs.len(), bs.len()),
                };
                while cstart < start {
                    if let Some(m) = space.find_at(bs, cstart) {
                        black_box(m.as_str());
                        cstart = m.end();
                    } else {
                        break;
                    }
                }
                lstart = end;
            }
        })
    }

    #[bench]
    fn bench_find_iter_u8(b: &mut Bencher) {
        use regex::bytes;
        let bs = STR.as_bytes();
        let line = bytes::Regex::new("\n").unwrap();
        let space = bytes::Regex::new(" ").unwrap();
        b.iter(|| {
            let mut lstart = 0;
            let mut cstart = 0;
            while lstart < bs.len() {
                let (start, end) = match line.find_at(bs, lstart) {
                    Some(m) => (m.start(), m.end()),
                    None => (bs.len(), bs.len()),
                };
                while cstart < start {
                    if let Some(m) = space.find_at(bs, cstart) {
                        black_box(m.as_bytes());
                        cstart = m.end();
                    } else {
                        break;
                    }
                }
                lstart = end;
            }
        })
    }

    #[bench]
    fn bench_split_batched(b: &mut Bencher) {
        let bs = STR.as_str();
        let line = &*LINE;
        let space = &*SPACE;
        b.iter(|| {
            for s in line.split(bs) {
                black_box(s);
            }
            for s in space.split(bs) {
                black_box(s);
            }
        })
    }

    #[bench]
    fn bench_split_batched_u8(b: &mut Bencher) {
        use regex::bytes;
        let bs = STR.as_bytes();
        let line = bytes::Regex::new("\n").unwrap();
        let space = bytes::Regex::new(" ").unwrap();
        b.iter(|| {
            for s in line.split(bs) {
                black_box(s);
            }
            for s in space.split(bs) {
                black_box(s);
            }
        })
    }

    #[bench]
    fn bench_split_by_line(b: &mut Bencher) {
        let bs = STR.as_str();
        let line = &*LINE;
        let space = &*SPACE;
        b.iter(|| {
            for s in line.split(bs) {
                for ss in space.split(s) {
                    black_box(ss);
                }
            }
        })
    }

    fn bytes(n: usize, line_pct: f64, space_pct: f64) -> Vec<u8> {
        let mut res = Vec::with_capacity(n);
        use rand::distributions::{Distribution, Uniform};
        let between = Uniform::new_inclusive(0.0, 1.0);
        let ascii = Uniform::new_inclusive(33u8, 126u8);
        let mut rng = rand::thread_rng();
        for _ in 0..n {
            let s = between.sample(&mut rng);
            if s < line_pct {
                res.push(b'\n')
            } else if s < space_pct {
                res.push(b' ')
            } else {
                res.push(ascii.sample(&mut rng))
            }
        }
        res
    }
}
