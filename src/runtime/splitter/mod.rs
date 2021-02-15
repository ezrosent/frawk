//! This module implements line reading in the style of AWK's getline. In particular, it has the
//! cumbersome API of letting you know if there was an error, or EOF, after the read has completed.
//!
//! In addition to this API, it also handles reading in chunks, with appropriate handling of UTF8
//! characters that cross chunk boundaries, or multi-chunk "lines".

// TODO: add padding to the linereader trait
pub mod batch;
pub mod chunk;
pub mod regex;

use super::str_impl::{Buf, Str, UniqueBuf};
use super::utf8::{is_utf8, validate_utf8_clipped};
use super::{Int, RegexCache};
use crate::common::Result;
use crate::pushdown::FieldUsage;

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

pub trait LineReader: Sized {
    type Line: for<'a> Line<'a>;
    fn filename(&self) -> Str<'static>;
    fn request_handles(&self, _size: usize) -> Vec<Box<dyn FnOnce() -> Self + Send>> {
        vec![]
    }
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
    fn next_file(&mut self) -> Result<bool>;
    fn set_used_fields(&mut self, used_fields: &FieldUsage);
    // Whether or not this LineReader is configured to check for valid UTF-8. This is used to
    // propagate consistent options across multiple LineReader instances.
    fn check_utf8(&self) -> bool;
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
    used_fields: FieldUsage,
    fields: Vec<Str<'static>>,
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
            used_fields: Default::default(),
            fields: Vec::new(),
            diverged: false,
        }
    }
}

impl DefaultLine {
    fn split_if_needed(&mut self, pat: &Str, rc: &mut RegexCache) -> Result<()> {
        if self.fields.len() == 0 {
            rc.split_regex(pat, &self.line, &self.used_fields.strs, &mut self.fields)?;
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
        Ok(sep
            .clone()
            .unmoor()
            // TODO: update join_slice to work for this case
            .join(self.fields[start..end].iter().cloned().map(trans))
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
            if !self.used_fields.parse_all() {
                // We projected out fields, but now we have set one of the interior fields and need
                // to print out $0. That means we have to split $0 in its entirety and then copy
                // over the fields that were already set.
                //
                // This is strictly more work than just reading all of the fields in the first
                // place; so once we hit this condition we overwrite the used fields with all() so
                // this doesn't happen again for a while.
                let old_set = std::mem::replace(&mut self.used_fields, Default::default());
                let mut new_vec = Vec::with_capacity(self.fields.len());
                rc.split_regex(pat, &self.line, &self.used_fields.strs, &mut new_vec)?;

                for (i, field) in self.fields.iter().enumerate().rev() {
                    if i >= new_vec.len() {
                        new_vec.resize_with(i + 1, Str::default);
                    }
                    if old_set.strs.get(i + 1) {
                        new_vec[i] = field.clone()
                    }
                }
                self.fields = new_vec;
            }
            let res = ofs.join_slice(&self.fields[..]);
            self.line = res.clone();
            self.diverged = false;
            res
        } else {
            self.split_if_needed(pat, rc)?;
            self.fields
                .get((col - 1) as usize)
                .cloned()
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
        let col = col as usize - 1;
        if col >= self.fields.len() {
            self.fields.resize_with(col + 1, Str::default);
        }
        self.fields[col] = s.clone().unmoor();
        self.diverged = true;
        Ok(())
    }
}

pub struct ChainedReader<R>(Vec<R>, /*check_utf8=*/ bool);

impl<R: LineReader> ChainedReader<R> {
    pub fn new(rs: impl Iterator<Item = R>) -> ChainedReader<R> {
        let mut v: Vec<_> = rs.collect();
        v.reverse();
        let check_utf8 = if let Some(r) = v.last() {
            r.check_utf8()
        } else {
            false
        };
        ChainedReader(v, check_utf8)
    }
}

impl<R: LineReader + 'static> LineReader for ChainedReader<R>
where
    R::Line: Default,
{
    type Line = R::Line;
    fn check_utf8(&self) -> bool {
        self.1
    }
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
        if cur.read_state() == 0 /* EOF */ && self.next_file()? {
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
    fn next_file(&mut self) -> Result<bool> {
        Ok(match self.0.last_mut() {
            Some(e) => {
                if !e.next_file()? {
                    self.0.pop();
                }
                true
            }
            None => false,
        })
    }
    fn set_used_fields(&mut self, used_fields: &FieldUsage) {
        for i in self.0.iter_mut() {
            i.set_used_fields(used_fields);
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

/// frawk inputs read chunks of data into large contiguous buffers, and then advance progress
/// within those buffers. The logic for reading, and conserving unused portions of previous buffers
/// when reading a new one, is handled by the Reader type.
///
/// Reader is currently not a great abstraction boundary, all of its state tends to "leak" into the
/// surrounding implementations of the LineReader trait that use it.
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

    // Validate input as UTF-8
    check_utf8: bool,
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
    pub(crate) fn new(r: R, chunk_size: usize, padding: usize, check_utf8: bool) -> Self {
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
            check_utf8,
        };
        res
    }

    pub(crate) fn check_utf8(&self) -> bool {
        self.check_utf8
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

    fn clear_buf(&mut self) {
        self.start = 0;
        self.end = 0;
        self.input_end = 0;
        self.buf = UniqueBuf::new(0).into_buf();
    }

    fn reset(&mut self) -> Result</*done*/ bool> {
        if self.state == ReaderState::EOF {
            return Ok(true);
        }
        let (next_buf, next_len, input_len) = self.get_next_buf(self.start)?;
        self.buf = next_buf.into_buf();
        self.end = next_len;
        self.input_end = input_len;
        self.start = 0;
        Ok(false)
    }

    fn get_next_buf(
        &mut self,
        consume: usize,
    ) -> Result<(UniqueBuf, /*end*/ usize, /*input_end*/ usize)> {
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
        let mut ulen = bytes.len();
        if self.check_utf8 {
            ulen = {
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
        }

        if done {
            self.state = ReaderState::EOF;
        }
        Ok((data, ulen, bytes_read))
    }
}
