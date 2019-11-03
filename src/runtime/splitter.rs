use super::shared::Shared;
use super::utf8::{parse_utf8, parse_utf8_clipped};
use super::Str;
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
    prefix: SmallVec<[u8; 8]>,
    cur: Shared<str>,
    chunk_size: usize,
    state: ReaderState,
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

thread_local! {
    static EMPTY: Shared<str> = Shared::from(Box::from(""));
}

impl<R: Read> Reader<R> {
    pub(crate) fn new(r: R, chunk_size: usize) -> Result<Self> {
        let mut res = Reader {
            inner: r,
            prefix: Default::default(),
            cur: EMPTY.with(Clone::clone),
            chunk_size,
            state: ReaderState::OK,
        };
        res.advance(0)?;
        Ok(res)
    }
    pub(crate) fn read_state(&self) -> i64 {
        self.state as i64
    }
    pub(crate) fn is_eof(&self) -> bool {
        self.get().len() == 0 && self.state == ReaderState::EOF
    }
    pub(crate) fn read_line<'a>(&mut self, pat: &Regex) -> Str<'a> {
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
        let mut prefix: Str = "".into();
        if self.is_eof() {
            return "".into();
        }
        loop {
            // Why this map invocation? Match objects hold a reference to the substring, which
            // makes it harder for us to call mutable methods like advance in the body, so just get
            // the start and end pointers.
            match pat.find(self.get()).map(|m| (m.start(), m.end())) {
                Some((start, end)) => {
                    let res = self.cur.extend(|s| &s[0..start]);
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
                    let cur: Str = self.cur.clone().into();
                    handle_err!(self.advance(self.get().len()));
                    if self.is_eof() {
                        // All done! Just return the rest of the buffer.
                        return cur.into();
                    }
                    prefix = Str::concat(prefix, cur.into());
                }
            }
        }
    }
    fn get(&self) -> &str {
        self.cur.get()
    }
    fn advance(&mut self, n: usize) -> Result<()> {
        let len = self.get().len();
        if len > n {
            self.cur = self.cur.extend(|s| &s[n..]);
            return Ok(());
        }
        if self.is_eof() {
            self.cur = EMPTY.with(Clone::clone);
            return Ok(());
        }

        let residue = n - len;
        self.cur = self.get_next_buf()?;
        self.advance(residue)
    }
    fn get_next_buf(&mut self) -> Result<Shared<str>> {
        let mut done = false;
        let mut data = vec![0u8; self.chunk_size];
        for (i, b) in self.prefix.iter().enumerate() {
            data[i] = *b;
        }
        let bytes_read = read_to_slice(&mut self.inner, &mut data[self.prefix.len()..])?;
        self.prefix.clear();
        if bytes_read != self.chunk_size {
            done = true;
            data.truncate(bytes_read);
        }
        let bytes = Shared::<[u8]>::from(Box::from(data));
        let utf8 = {
            let opt = if done {
                bytes.extend_opt(|bs| parse_utf8(bs))
            } else {
                bytes.extend_opt(|bs| parse_utf8_clipped(bs))
            };
            if let Some(u) = opt {
                u
            } else {
                return err!("invalid UTF8");
            }
        };
        let ulen = utf8.get().len();
        if !done && ulen != bytes_read {
            self.prefix.extend_from_slice(&bytes.get()[ulen..]);
        }
        if done {
            self.state = ReaderState::EOF;
        }
        Ok(utf8)
    }
}

#[cfg(test)]
mod test {
    // need to benchmark batched splitting vs. regular splitting to get a feel for things.
    extern crate test;
    use lazy_static::lazy_static;
    use regex::Regex;
    use test::{black_box, Bencher};
    lazy_static! {
        static ref STR: String = String::from_utf8(bytes(1 << 20, 0.001, 0.05)).unwrap();
        static ref LINE: Regex = Regex::new("\n").unwrap();
        static ref SPACE: Regex = Regex::new(" ").unwrap();
    }

    #[test]
    fn test_line_split() {
        use super::Str;
        use std::io::Cursor;
        let bs = String::from_utf8(bytes(1 << 18, 0.001, 0.05)).unwrap();
        let c = Cursor::new(bs.clone());
        let mut rdr = super::Reader::new(c, 1 << 9).unwrap();
        let mut lines = Vec::new();
        while !rdr.is_eof() {
            let line = rdr.read_line(&*LINE);
            assert!(rdr.read_state() != -1);
            lines.push(line);
        }
        let mut expected: Vec<_> = LINE.split(bs.as_str()).map(|x| Str::from(x)).collect();
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
        let ascii = Uniform::new_inclusive(0u8, 127u8);
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
