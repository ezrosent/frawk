//! This module implements line reading in the style of AWK's getline. In particular, it has the
//! cumbersome API of letting you know if there was an error, or EOF, after the read has completed.
//!
//! In addition to this API, it also handles reading in chunks, with appropriate handling of UTF8
//! characters that cross chunk boundaries, or multi-chunk "lines".
use super::bytes_rc::{Str, UniqueBuf};
use super::utf8::{is_utf8, validate_utf8_clipped};
use crate::common::Result;

use regex::Regex;
use smallvec::SmallVec;

use std::io::{ErrorKind, Read};

#[repr(i64)]
#[derive(PartialEq, Eq, Copy, Clone)]
pub(crate) enum ReaderState {
    ERROR = -1,
    EOF = 0,
    OK = 1,
}

pub(crate) struct Reader<R> {
    inner: R,
    // The "stray bytes" that will be prepended to the next buffer.
    prefix: SmallVec<[u8; 8]>,
    // TODO we probably want this to be a Buf, not a Str, but Buf's API gives out bytes not
    // strings.
    cur: Str<'static>,
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
    pub(crate) fn new(r: R, chunk_size: usize) -> Result<Self> {
        let mut res = Reader {
            inner: r,
            prefix: Default::default(),
            cur: Default::default(),
            chunk_size,
            state: ReaderState::OK,
            last_len: 0,
        };
        res.advance(0)?;
        Ok(res)
    }
    pub(crate) fn read_state(&self) -> i64 {
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
    pub(crate) fn is_eof(&self) -> bool {
        self.cur.len() == 0 && self.state == ReaderState::EOF
    }
    pub(crate) fn read_line(&mut self, pat: &Regex) -> Str<'static> {
        let res = self.read_line_inner(pat);
        self.last_len = res.len();
        res
    }
    fn read_line_inner(&mut self, pat: &Regex) -> Str<'static> {
        macro_rules! handle_err {
            ($e:expr) => {
                if let Ok(e) = $e {
                    e
                } else {
                    self.state = ReaderState::ERROR;
                    return "".into();
                }
            };
        }
        self.state = ReaderState::OK;
        let mut prefix: Str<'static> = Default::default();
        loop {
            // Why this map invocation? Match objects hold a reference to the substring, which
            // makes it harder for us to call mutable methods like advance in the body, so just get
            // the start and end pointers.
            match self
                .cur
                .with_str(|s| pat.find(s).map(|m| (m.start(), m.end())))
            {
                Some((start, end)) => {
                    let res = self.cur.slice(0, start);
                    // NOTE if we get a read error here, then we will stop one line early.
                    // That seems okay, but we could find out that it actually isn't, in which case
                    // we would want some more complicated error handling here.
                    handle_err!(self.advance(end));
                    return if prefix.with_str(|s| s.len() > 0) {
                        Str::concat(prefix, res.into())
                    } else {
                        res.into()
                    };
                }
                None => {
                    let cur: Str = self.cur.clone();
                    handle_err!(self.advance(self.cur.len()));
                    prefix = Str::concat(prefix, cur);
                    if self.is_eof() {
                        // All done! Just return the rest of the buffer.
                        return prefix;
                    }
                }
            }
        }
    }

    fn advance(&mut self, n: usize) -> Result<()> {
        let len = self.cur.len();
        if len > n {
            self.cur = self.cur.slice(n, len);
            return Ok(());
        }
        if self.is_eof() {
            self.cur = Default::default();
            return Ok(());
        }

        let residue = n - len;
        self.cur = self.get_next_buf()?;
        self.advance(residue)
    }
    fn get_next_buf(&mut self) -> Result<Str<'static>> {
        let mut done = false;
        // NB: UniqueBuf fills the allocation with zeros.
        let mut data = UniqueBuf::new(self.chunk_size);
        let mut bytes = data.as_mut_bytes();
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
                return err!("invalid UTF8");
            }
        };
        if !done && ulen != bytes_read {
            // We clipped a utf8 character at the end of the buffer. Add it to prefix.
            self.prefix.extend_from_slice(&bytes[ulen..]);
        }
        if done {
            self.state = ReaderState::EOF;
        }
        Ok(unsafe { data.into_buf().into_str() }.slice(0, ulen))
    }
}

#[cfg(test)]
mod test {
    // need to benchmark batched splitting vs. regular splitting to get a feel for things.
    extern crate test;
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
        let mut rdr = super::Reader::new(c, chunk_size).unwrap();
        let mut lines = Vec::new();
        while !rdr.is_eof() {
            let line = rdr.read_line(&*LINE).upcast();
            assert!(rdr.read_state() != -1);
            lines.push(line);
        }

        let expected: Vec<_> = LINE.split(bs.as_str()).map(ref_str).collect();
        assert_eq!(lines, expected);
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
        let mut rdr = super::Reader::new(c, chunk_size).unwrap();
        let mut lines = Vec::new();
        while !rdr.is_eof() {
            let line = rdr.read_line(&*LINE).upcast();
            assert!(rdr.read_state() != -1);
            lines.push(line);
        }
        let expected: Vec<_> = LINE.split(s.as_str()).map(ref_str).collect();
        assert_eq!(lines, expected);
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
                res.push('\n' as u8)
            } else if s < space_pct {
                res.push(' ' as u8)
            } else {
                res.push(ascii.sample(&mut rng))
            }
        }
        res
    }
}
