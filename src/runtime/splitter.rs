use super::shared::{IterRefFn, Shared, SharedSlice};
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
    fn get_line<R: Read>(&mut self, r: &mut Reader<R>) -> Result<Option<Self::Line>>;
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
// TODO: Rework this logic to work line by line
//      * Reader just holds onto (shrinking) current buffer. Hands out `Shared`s
//      * "Cursor" (rename "Splitter") just hands out Lines as normal, but instead of batching
//        splits internally it just reads off of the top and stores a Str that holds a (delimited)
//        reference into the buffer.
//      * No need for caching or anything: pass the regexes into split (XXX this raises issues for
//      split).


struct RegexCursor<'a> {
    line_re: Regex,
    col_re: Rc<Regex>,
    // prefix to append to the next line
    partial: Str<'a>,
    // splits
    splits: Vec<Str<'a>>,
}

impl<'a> RegexCursor<'a> {
    pub(crate) fn new(line: Regex, column: Regex) -> RegexCursor<'a> {
        RegexCursor {
            line_re: line,
            col_re: Rc::new(column),
            partial: "".into(),
            splits: Default::default(),
        }
    }
}

struct RegexLine<'a> {
    pat: Rc<Regex>,
    text: Str<'a>,
}

impl<'a> Line<'a> for RegexLine<'a> {
    fn text(&self) -> Str<'a> {
        self.text.clone()
    }
    fn columns(&self, v: &mut Vec<Str<'a>>) {
        self.text.split(&*self.pat, |s| v.push(s))
    }
}

impl<'a> Splitter<'a> for RegexCursor<'a> {
    type Line = RegexLine<'a>;
    fn get_line<R: Read>(&mut self, r: &mut Reader<R>) -> Result<Option<Self::Line>> {
        // This method is pretty complicated, largely because it has to
        // handle lines that cross chunk boundaries.
        //
        // In the (hopefully) common case, we have already pre-split
        // some lines and we can just hand one of them out.
        if let Some(s) = self.splits.pop() {
            debug_assert!(self.partial.with_str(|s| s == ""));
            return Ok(Some(RegexLine {
                pat: self.col_re.clone(),
                text: s,
            }));
        }

        // Some helpers for the slow path. This is an IterRefFn that
        // wraps Regex::split for use with splitting a Shared<str> into
        // lines. See the comments on IterRefFn if you're curious about
        // why the extra struct is necessary.
        #[derive(Copy, Clone)]
        struct RSplit<'a>(&'a Regex);
        impl<'a, 'b> IterRefFn<'b, str, str> for RSplit<'a> {
            type I = regex::Split<'a, 'b>;
            #[inline]
            fn invoke(self, s: &'b str) -> Self::I {
                self.0.split(s)
            }
        }
        // Helper function for extracting self.partial and replacing it
        // with the empty string. This would be nicer as a method, but
        // that defeats the NLL borrow checker in some uses, so we'll
        // settle for this.
        fn replace<'a>(s: &mut Str<'a>) -> Str<'a> {
            std::mem::replace(s, "".into())
        }

        let splitter = RSplit(&self.line_re);
        let s = if let Some(s) = r.get_next_buf()? {
            s
        } else {
            // EOF
            return Ok(None);
        };

        // Hopefully there was at least one line in the buffer we just
        // read, but if not we will have to keep reading buffers and
        // appending them to self.partial until we reach a line break,
        // or an EOF.
        let mut slc = s.extend_slice_iter(splitter);
        while slc.len() == 1 {
            let prefix = replace(&mut self.partial);
            self.partial = Str::concat(prefix, slc.get_shared(0).unwrap().into());
            match r.get_next_buf()? {
                Some(s) => {
                    slc = s.extend_slice_iter(splitter);
                }
                None => {
                    return Ok(Some(RegexLine {
                        pat: self.col_re.clone(),
                        text: replace(&mut self.partial),
                    }))
                }
            }
        }

        // Process the split output and push it onto
        // self.splits. "Process" here means prepend self.partial to the
        // first string, and assign the last element to partial.
        let len = slc.len();
        assert!(len > 1);
        for (i, s) in slc.iter_shared().enumerate() {
            if i == 0 {
                // first
                let prefix = replace(&mut self.partial);
                self.splits.push(Str::concat(prefix.into(), s.into()));
            } else if i == len - 1 && s.get() != "" {
                // last
                self.partial = s.into();
            } else {
                self.splits.push(s.into());
            }
        }
        self.get_line(r)
    }
}

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
