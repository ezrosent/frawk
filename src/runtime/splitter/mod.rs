//! This module implements line reading in the style of AWK's getline. In particular, it has the
//! cumbersome API of letting you know if there was an error, or EOF, after the read has completed.
//!
//! In addition to this API, it also handles reading in chunks, with appropriate handling of UTF8
//! characters that cross chunk boundaries, or multi-chunk "lines".

// TODO: add padding to the linereader trait
pub mod batch;
pub mod regex;

use super::str_impl::{Buf, Str, UniqueBuf};
use super::utf8::{is_utf8, validate_utf8_clipped};
use super::{Int, LazyVec, RegexCache};
use crate::common::Result;
use crate::pushdown::FieldSet;

use std::io::{ErrorKind, Read};

// We have several implementations of "read and split a line"; they are governed by the LineReader
// and Line traits.

pub trait Line<'a>: Default {
    fn join_cols<F>(
        &mut self,
        start: Int,
        end: Int,
        sep: &Str<'a>,
        nf: usize,
        trans: F,
    ) -> Result<Str<'a>>
    where
        F: FnMut(Str<'static>) -> Str<'static>;
    fn nf(&mut self, pat: &Str, rc: &mut RegexCache) -> Result<usize>;
    fn get_col(&mut self, col: Int, pat: &Str, ofs: &Str, rc: &mut RegexCache) -> Result<Str<'a>>;
    fn set_col(&mut self, col: Int, s: &Str<'a>, pat: &Str, rc: &mut RegexCache) -> Result<()>;
}

pub trait LineReader {
    type Line: for<'a> Line<'a>;
    fn filename(&self) -> Str<'static>;
    // TODO we should probably have the default impl the other way around.
    fn read_line(
        &mut self,
        pat: &Str,
        rc: &mut RegexCache,
    ) -> Result<(/*file changed*/ bool, Self::Line)>;
    fn read_line_reuse<'a, 'b: 'a>(
        &'b mut self,
        pat: &Str,
        rc: &mut RegexCache,
        old: &'a mut Self::Line,
    ) -> Result</* file changed */ bool> {
        let (changed, mut new) = self.read_line(pat, rc)?;
        std::mem::swap(old, &mut new);
        Ok(changed)
    }
    fn read_state(&self) -> i64;
    fn next_file(&mut self) -> bool;
    fn set_used_fields(&mut self, _used_fields: &crate::pushdown::FieldSet);
}

fn normalize_join_indexes(start: Int, end: Int, nf: usize) -> Result<(usize, usize)> {
    if start <= 0 || end <= 0 {
        return err!("smallest joinable column is 1, got {}", start);
    }
    let mut start = start as usize - 1;
    let mut end = end as usize;
    if end > nf {
        end = nf;
    }
    if end < start {
        start = end;
    }
    Ok((start, end))
}

// Default implementation of Line; it supports assignment into fields as well as lazy splitting.
pub struct DefaultLine {
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
impl Default for DefaultLine {
    fn default() -> DefaultLine {
        DefaultLine {
            line: Str::default(),
            used_fields: FieldSet::all(),
            fields: LazyVec::new(),
            diverged: false,
        }
    }
}

impl DefaultLine {
    fn split_if_needed(&mut self, pat: &Str, rc: &mut RegexCache) -> Result<()> {
        if self.fields.len() == 0 {
            rc.split_regex(pat, &self.line, &self.used_fields, &mut self.fields)?;
        }
        Ok(())
    }
}

impl<'a> Line<'a> for DefaultLine {
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
        let (start, end) = normalize_join_indexes(start, end, nf)?;
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

pub struct ChainedReader<R>(Vec<R>);

impl<R> ChainedReader<R> {
    pub fn new(rs: impl Iterator<Item = R>) -> ChainedReader<R> {
        let mut v: Vec<_> = rs.collect();
        v.reverse();
        ChainedReader(v)
    }
}

impl<R: LineReader> LineReader for ChainedReader<R>
where
    R::Line: Default,
{
    type Line = R::Line;
    fn filename(&self) -> Str<'static> {
        self.0
            .last()
            .map(LineReader::filename)
            .unwrap_or_else(Str::default)
    }
    fn read_line(&mut self, pat: &Str, rc: &mut RegexCache) -> Result<(bool, R::Line)> {
        let mut line = R::Line::default();
        let changed = self.read_line_reuse(pat, rc, &mut line)?;
        Ok((changed, line))
    }
    fn read_line_reuse<'a, 'b: 'a>(
        &'b mut self,
        pat: &Str,
        rc: &mut RegexCache,
        old: &'a mut Self::Line,
    ) -> Result<bool> {
        let cur = match self.0.last_mut() {
            Some(cur) => cur,
            None => {
                *old = Default::default();
                return Ok(false);
            }
        };
        let changed = cur.read_line_reuse(pat, rc, old)?;
        if cur.read_state() == 0 /* EOF */ && self.next_file() {
            self.read_line_reuse(pat, rc, old)?;
            Ok(true)
        } else {
            Ok(changed)
        }
    }
    fn read_state(&self) -> i64 {
        match self.0.last() {
            Some(cur) => cur.read_state(),
            None => 0, /* EOF */
        }
    }
    fn next_file(&mut self) -> bool {
        match self.0.last_mut() {
            Some(e) => {
                if !e.next_file() {
                    self.0.pop();
                }
                true
            }
            None => false,
        }
    }
    fn set_used_fields(&mut self, used_fields: &FieldSet) {
        for i in self.0.iter_mut() {
            i.set_used_fields(used_fields);
        }
    }
}

fn is_ascii_whitespace(b: u8) -> bool {
    matches!(b, b' ' | b'\x09'..=b'\x0d')
}

fn is_ascii_whitespace_not_nl(b: u8) -> bool {
    matches!(b, b' ' | b'\x09' | b'\x0b'..=b'\x0d')
}

/// Split input by (ASCII) whitespace, using Awk semantics.
pub struct DefaultSplitter<R> {
    reader: Reader<R>,
    name: Str<'static>,
    used_fields: FieldSet,
    // As in RegexSplitter, used to trigger updating FILENAME on the first read.
    start: bool,
}

impl<R: Read> LineReader for DefaultSplitter<R> {
    type Line = DefaultLine;
    fn filename(&self) -> Str<'static> {
        self.name.clone()
    }
    fn read_state(&self) -> i64 {
        self.reader.read_state()
    }
    fn next_file(&mut self) -> bool {
        self.reader.force_eof();
        false
    }
    fn set_used_fields(&mut self, used_fields: &FieldSet) {
        self.used_fields = used_fields.clone();
    }
    fn read_line_reuse<'a, 'b: 'a>(
        &'b mut self,
        _pat: &Str,
        _rc: &mut RegexCache,
        old: &'a mut DefaultLine,
    ) -> Result</* file changed */ bool> {
        let mut old_fields = old.fields.get_cleared_vec();
        let (line, consumed) = self.read_line_inner(&mut old_fields);
        old.fields = LazyVec::from_vec(old_fields);
        old.line = line;
        old.diverged = false;
        let start = self.start;
        if start {
            old.used_fields = self.used_fields.clone();
        } else if old.used_fields != self.used_fields {
            // Doing this every time relies on the used field set being small, and easy to compare.
            // If we wanted to make it arbitrary-sized, we could switch this to being a length
            // comparison, as we'll never remove used fields dynamically.
            self.used_fields = old.used_fields.clone();
        }
        self.reader.last_len = consumed;
        self.start = false;
        Ok(start)
    }
    fn read_line(&mut self, _pat: &Str, _rc: &mut RegexCache) -> Result<(bool, DefaultLine)> {
        let mut line = DefaultLine::default();
        let changed = self.read_line_reuse(_pat, _rc, &mut line)?;
        Ok((changed, line))
    }
}

impl<R: Read> DefaultSplitter<R> {
    pub fn new(r: R, chunk_size: usize, name: impl Into<Str<'static>>) -> Self {
        DefaultSplitter {
            reader: Reader::new(r, chunk_size, /*padding=*/ 0),
            name: name.into(),
            used_fields: FieldSet::all(),
            start: true,
        }
    }
    fn read_line_inner(&mut self, fields: &mut Vec<Str<'static>>) -> (Str<'static>, usize) {
        if self.reader.is_eof() {
            return (Str::default(), 0);
        }
        self.reader.state = ReaderState::OK;
        let mut cur = self.reader.start;
        let mut cur_field_start = cur;
        macro_rules! get_field {
            ($n:expr, $from:expr) => {
                if self.used_fields.get($n) {
                    unsafe { self.reader.buf.slice_to_str($from, cur) }
                } else {
                    Str::default()
                }
            };
        }
        loop {
            let bytes = &self.reader.buf.as_bytes()[..self.reader.end];
            'inner: while cur < bytes.len() {
                let mut cur_b = unsafe { *bytes.get_unchecked(cur) };
                if cur_b == b'\n' {
                    let line = get_field!(0, self.reader.start);
                    if cur_field_start != cur {
                        let last_field = get_field!(fields.len() + 1, cur_field_start);
                        fields.push(last_field);
                    }
                    let new_start = cur + 1;
                    let consumed = new_start - self.reader.start;
                    self.reader.start = new_start;
                    return (line, consumed);
                }
                if is_ascii_whitespace(cur_b) {
                    if cur != cur_field_start {
                        let last_field = get_field!(fields.len() + 1, cur_field_start);
                        fields.push(last_field);
                    }
                    cur_field_start = cur;
                    while is_ascii_whitespace_not_nl(cur_b) {
                        cur += 1;
                        cur_field_start += 1;
                        if cur == bytes.len() {
                            break 'inner;
                        }
                        cur_b = unsafe { *bytes.get_unchecked(cur) };
                    }
                } else {
                    cur += 1;
                }
            }
            // Out of space. Let's preserve the start of the current line and fetch a new buffer,
            // resetting current_field_start as appropriate.
            let next_cur_field_start = cur_field_start - self.reader.start;
            let next_cur = cur - self.reader.start;
            match self.reader.reset() {
                Ok(true) => {
                    // EOF. Grab what we have of the last field and the line.
                    cur = self.reader.end;
                    let line = get_field!(0, self.reader.start);
                    if cur_field_start != cur {
                        let last_field = get_field!(fields.len() + 1, cur_field_start);
                        fields.push(last_field);
                    }
                    let consumed = self.reader.end - self.reader.start;
                    self.reader.start = self.reader.end;
                    return (line, consumed);
                }
                Ok(false) => {
                    // There's a new buffer! Reset the offsets.
                    cur_field_start = next_cur_field_start;
                    cur = next_cur;
                }
                Err(_) => {
                    fields.clear();
                    self.reader.state = ReaderState::ERROR;
                    return (Str::default(), 0);
                }
            }
        }
    }
}

// Buffer management and io

#[repr(i64)]
#[derive(PartialEq, Eq, Copy, Clone)]
pub(crate) enum ReaderState {
    ERROR = -1,
    EOF = 0,
    OK = 1,
}

struct Reader<R> {
    inner: R,
    buf: Buf,
    // The current "read head" into buf.
    start: usize,
    // Upper bound on readable bytes into buf (not including padding and clipped UTF8 bytes).
    end: usize,
    // Upper bound on all bytes read from input, not including padding.
    input_end: usize,
    chunk_size: usize,
    // Padding is used for the splitters in the [batch] module, which may read some bytes past the
    // end of the buffer.
    padding: usize,
    state: ReaderState,
    // Reads of the "error state" lag behind reads from the buffer. last_len helps us determine
    // when an EOF has been reached from an external perspective.
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
    pub(crate) fn new(r: R, chunk_size: usize, padding: usize) -> Self {
        let res = Reader {
            inner: r,
            buf: UniqueBuf::new(0).into_buf(),
            start: 0,
            end: 0,
            input_end: 0,
            chunk_size,
            padding,
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
                // NB: last_len should really be "bytes consumed"; i.e. it should be the length
                // of the line including any trimmed characters, and the record separator. I.e.
                // "empty lines" that are actually in the input should result in a nonzero value
                // here.
                if self.last_len == 0 {
                    self.state as i64
                } else {
                    ReaderState::OK as i64
                }
            }
        }
    }

    fn reset(&mut self) -> Result</*done*/ bool> {
        if self.state == ReaderState::EOF {
            return Ok(true);
        }
        let (next_buf, next_len, input_len) = self.get_next_buf(self.start)?;
        self.buf = next_buf;
        self.end = next_len;
        self.input_end = input_len;
        self.start = 0;
        Ok(false)
    }

    // TODO: get rid of advance()
    fn advance(&mut self, n: usize) -> Result<()> {
        let len = self.end - self.start;
        if len > n {
            self.start += n;
            return Ok(());
        }
        if self.is_eof() {
            return Ok(());
        }

        self.start = self.end;
        let residue = n - len;
        self.reset()?;
        self.advance(residue)
    }

    fn get_next_buf(
        &mut self,
        consume: usize,
    ) -> Result<(Buf, /*end*/ usize, /*input_end*/ usize)> {
        let mut done = false;
        let plen = self.input_end.saturating_sub(consume);
        // Double the chunk size if it is too small to read a sufficient batch given the prefix
        // size.
        if plen > self.chunk_size / 2 {
            self.chunk_size = std::cmp::max(self.chunk_size * 2, 1024);
        }
        // NB: UniqueBuf fills the allocation with zeros.
        let mut data = UniqueBuf::new(self.chunk_size + self.padding);

        // First, append the remaining bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.buf.as_ptr().offset(consume as isize),
                data.as_mut_ptr(),
                plen,
            );
        }
        let mut bytes = &mut data.as_mut_bytes()[..self.chunk_size];
        let bytes_read = plen + read_to_slice(&mut self.inner, &mut bytes[plen..])?;
        if bytes_read != self.chunk_size {
            done = true;
            bytes = &mut bytes[..bytes_read];
        }

        let ulen = {
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
        if done {
            self.state = ReaderState::EOF;
        }
        Ok((data.into_buf(), ulen, bytes_read))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn whitespace_split(corpus: &str) {
        let mut _cache = RegexCache::default();
        let _pat = Str::default();
        let expected: Vec<Vec<Str<'static>>> = corpus
            .split('\n')
            .map(|line| {
                line.split(|c: char| c.is_ascii_whitespace())
                    // trim of leading and trailing whitespace.
                    .filter(|x| *x != "")
                    .map(|x| Str::from(x).unmoor())
                    .collect()
            })
            .collect();
        let reader = std::io::Cursor::new(corpus);
        let mut reader = DefaultSplitter::new(reader, 1024, "fake-stdin");
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
                if g != e {
                    eprintln!("===============");
                    eprintln!("line {} has a mismatch", i);
                    eprintln!("got:  {}", disp_vec(g));
                    eprintln!("want: {}", disp_vec(e));
                }
            }
            panic!("test failed. See debug output");
        }
    }

    #[test]
    fn whitespace_splitter() {
        whitespace_split(crate::test_string_constants::PRIDE_PREJUDICE_CH2);
        whitespace_split(crate::test_string_constants::VIRGIL);
        whitespace_split("   leading whitespace   \n and some    more");
    }
}
