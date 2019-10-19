use super::shared::{Shared, SharedSlice};
use super::utf8::{parse_utf8, parse_utf8_clipped};
use super::{Inner, Str};
use crate::common::Result;

use regex::Regex;
use smallvec::SmallVec;

use std::cell::RefCell;
use std::io::{ErrorKind, Read};
use std::rc::Rc;
// TODO: get "Splitter" to be a trait to allow for specialized splitters for the default non-regex
// ones. Regex splitting is fast, but not as fast as it would be to split on whitespace and
// line-breaks.
// TODO: implement splitter for regexes, splitting columns in a line-by-line fashion.

trait Line<'a> {
    fn text(&self) -> Str<'a>;
    fn columns(&self, v: &mut Vec<Str<'a>>);
}
trait Splitter<'a> {
    type Line: Line<'a>;
    fn get_line<R: Read>(&mut self, r: &mut Reader<R>) -> Self::Line;
}

const CHUNK_SIZE: usize = 4 << 10;

// 1) Read up to CHUNK_SIZE from inner along with prefix.
// 2) Split its contents (into a Shared), append clipped bytes to prefix.
// 3) have a get_next() that returns next line as a Str, blocking until it gets another one.

pub(crate) struct Reader<R> {
    inner: R,
    prefix: SmallVec<[u8; 8]>,
    done: bool,
}

fn read_to_slice(r: &mut impl Read, mut buf: &mut [u8]) -> Result<usize> {
    let mut read = 0;
    loop {
        match r.read(buf) {
            Ok(n) => {
                if n == buf.len() {
                    break;
                }
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
    fn get_next_buf(&mut self) -> Result<Option<Shared<str>>> {
        if self.done {
            return Ok(None);
        }
        let mut data = vec![0u8; CHUNK_SIZE];
        for (i, b) in self.prefix.iter().enumerate() {
            data[i] = *b;
        }
        let bytes_read = read_to_slice(&mut self.inner, &mut data[self.prefix.len()..])?;
        self.prefix.clear();
        if bytes_read != CHUNK_SIZE {
            self.done = true;
            data.truncate(bytes_read);
        }
        let bytes = Shared::<[u8]>::from(Box::from(data));
        let utf8 = {
            let opt = if self.done {
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
        if !self.done && ulen != bytes_read {
            self.prefix.extend_from_slice(&bytes.get()[ulen..]);
        }
        Ok(Some(utf8))
    }
}

// pass an &mut to RegexCursor, read until we get more than one, or else a split
// advance gets called on FileBuffer and RegexCursor
struct FileBuffer<R> {
    pub contents: Vec<Shared<str>>,
    reader: Reader<R>,
}

impl<R: Read> FileBuffer<R> {
    fn consume(&mut self) {
        self.contents.pop();
    }

    fn refill(&mut self) -> Result<bool /*done*/> {
        Ok(match self.reader.get_next_buf()? {
            Some(x) => {
                self.contents.push(x);
                false
            }
            None => true,
        })
    }
}

// Cursor (holding onto current offset) with (eager) regex registry
// (kept as LRU cache).
//
//
// * Regex splitting logic can happen at the level of a vector of Shared<[u8],str>
//   - Split each element.
//   - Ask for a slice (Str? for concat) of all elements up to a given offset
// * Splitter points to a shared (mutable) vector of strings with an
//   advance method that keeps an offset in the current string and pops
//   it off if need be. Whenever a read happens that requires a farther
//   offset, we read the next chunk.

struct RegexCursor {
    re: Regex,
    // offset into file where current strings start
    start: usize,
    // splits
    splits: Vec<SharedSlice<str>>,
}

struct RegexLine<'a> {
    pat: Regex,
    text: Str<'a>,
}

impl<'a> Line<'a> for RegexLine<'a> {
    fn text(&self) -> Str<'a> {
        self.text.clone()
    }
    fn columns(&self, v: &mut Vec<Str<'a>>) {
        match &*self.text.0.borrow() {
            Inner::Line(s) => unimplemented!(),
            x => unimplemented!(), // s.with_str(|s| for s in self.pat.split(x)
        }
    }
}

// impl<'a> Splitter<'a> for RegexCursor {
//     type Line = Str<'a>;
//     fn get_line<R: Read>(&mut self, r: &mut Reader<R>) -> Self::Line {
//         unimplemented!()
//     }
// }

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

    #[bench]
    fn bench_split_by_line(b: &mut Bencher) {
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
    fn bench_split_batched(b: &mut Bencher) {
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
