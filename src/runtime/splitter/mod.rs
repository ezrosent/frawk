//! This module implements line reading in the style of AWK's getline. In particular, it has the
//! cumbersome API of letting you know if there was an error, or EOF, after the read has completed.
//!
//! In addition to this API, it also handles reading in chunks, with appropriate handling of UTF8
//! characters that cross chunk boundaries, or multi-chunk "lines".

// TODO: add padding to the linereader trait
pub mod batch;
pub mod regex;

use std::io::Write;
use std::mem;

use super::str_impl::{Buf, DynamicBufHeap, Str, UniqueBuf};
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
        // How many bytes has this call to read consumed?
        let mut consumed = 0;
        // We must handle lines and fields that cross buffer refresh boundaries. These fields
        // contain prefixes of those values that we can prepend when we complete the field or line.
        let mut line_prefix: Str<'static> = Default::default();
        let mut last_field_prefix: Str<'static> = Default::default();
        // Should we prepend last_field_prefix to the next field we find?
        let mut has_field_prefix = false;
        if self.reader.is_eof() {
            return (line_prefix, consumed);
        }
        macro_rules! handle_err {
            ($e:expr) => {
                if let Ok(e) = $e {
                    e
                } else {
                    fields.clear();
                    self.reader.state = ReaderState::ERROR;
                    return (Str::default(), 0);
                }
            };
        }
        self.reader.state = ReaderState::OK;
        loop {
            let bytes = &self.reader.buf.as_bytes()[self.reader.start..self.reader.end];
            let mut cur = 0;
            let mut cur_index = self.reader.start;
            let mut current_field_start = self.reader.start;

            // Grab the current field out of the buffer, if it is used.
            macro_rules! get_field {
                () => {
                    if self.used_fields.get(fields.len() + 1) {
                        let mut res =
                            unsafe { self.reader.buf.slice_to_str(current_field_start, cur_index) };
                        if has_field_prefix {
                            res = Str::concat(
                                mem::replace(&mut last_field_prefix, Str::default()),
                                res,
                            );
                        }
                        res
                    } else {
                        Str::default()
                    };
                };
            }

            // Loop over the remaining bytes in the current buffer.
            'inner: while cur < bytes.len() {
                let mut cur_b = unsafe { *bytes.get_unchecked(cur) };
                if cur_b == b'\n' {
                    // Complete the line: get the line and the last field and add them to the
                    // output, keeping in mind that empty fields are omitted using the Awk
                    // whitespace splitting discipline.
                    let line = if self.used_fields.get(0) {
                        Str::concat(mem::replace(&mut line_prefix, Str::default()), unsafe {
                            self.reader.buf.slice_to_str(self.reader.start, cur_index)
                        })
                    } else {
                        Str::default()
                    };
                    if has_field_prefix || current_field_start != cur_index {
                        let last_field = get_field!();
                        fields.push(last_field);
                    }
                    let progress = cur + 1;
                    consumed += progress;
                    handle_err!(self.reader.advance(progress));
                    return (line, consumed);
                }
                // We've found some whitespace. This might be the end of a field.
                if is_ascii_whitespace(cur_b) {
                    if has_field_prefix || cur_index != current_field_start {
                        // not an empty field
                        let last_field = get_field!();
                        fields.push(last_field);
                    }
                    // Skip any adjacent whitespace.
                    current_field_start = cur_index;
                    while is_ascii_whitespace_not_nl(cur_b) {
                        cur += 1;
                        cur_index += 1;
                        current_field_start += 1;
                        if cur == bytes.len() {
                            break 'inner;
                        }
                        cur_b = unsafe { *bytes.get_unchecked(cur) };
                    }
                } else {
                    // Continue the current field.
                    cur += 1;
                    cur_index += 1;
                }
            }

            // Out of space in the current buffer. Populate last_field_prefix and line_prefix if we
            // need them.
            if current_field_start != cur_index {
                // We still need to make sure we are skipping empty fields.
                if self.used_fields.get(fields.len() + 1) {
                    last_field_prefix = Str::concat(last_field_prefix, unsafe {
                        self.reader.buf.slice_to_str(current_field_start, cur_index)
                    });
                }
                has_field_prefix = true;
            }
            if self.used_fields.get(0) {
                line_prefix = Str::concat(line_prefix, unsafe {
                    self.reader.buf.slice_to_str(self.reader.start, cur_index)
                });
            }

            // Grab a new buffer.
            let remaining = self.reader.remaining();
            consumed += remaining;
            handle_err!(self.reader.advance(remaining));
            if self.reader.is_eof() {
                // If there aren't any new buffers, return the last field and lines.
                if has_field_prefix {
                    fields.push(last_field_prefix);
                }
                return (line_prefix, consumed);
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
    // The "stray bytes" that will be prepended to the next buffer.
    prefix: DynamicBufHeap,
    buf: Buf,
    start: usize,
    end: usize,
    chunk_size: usize,
    // Padding is used for the splitters in the [batch] module, which may read some bytes past the
    // end of the buffer.
    padding: usize,
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
    pub(crate) fn new(r: R, chunk_size: usize, padding: usize) -> Self {
        let res = Reader {
            inner: r,
            prefix: DynamicBufHeap::new(chunk_size + padding),
            buf: UniqueBuf::new(0).into_buf(),
            start: 0,
            end: 0,
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
        let mut done = false;
        // NB: DynamicBufHeap fills the allocation with zeros.
        let mut data = mem::replace(
            &mut self.prefix,
            DynamicBufHeap::new(self.chunk_size + self.padding),
        );
        let plen = data.write_head();
        let mut bytes = &mut data.as_mut_bytes()[..self.chunk_size];
        // Try to fill up the rest of `data` with new bytes.
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
        if !done && ulen != bytes_read {
            // We clipped a utf8 character at the end of the buffer. Add it to prefix.
            self.prefix.write(&bytes[ulen..]).unwrap();
        }
        if done {
            self.state = ReaderState::EOF;
        }
        Ok((data.into_buf(), ulen))
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
