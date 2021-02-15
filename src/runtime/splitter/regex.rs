//! Regex-based splitting routines
use std::io::Read;

use crate::common::Result;
use crate::pushdown::FieldUsage;
use crate::runtime::Str;
use regex::bytes::Regex;

use super::{DefaultLine, LineReader, Reader, ReaderState};

// TODO: this can probably just be "Splitter"
pub struct RegexSplitter<R> {
    reader: Reader<R>,
    name: Str<'static>,
    used_fields: FieldUsage,
    // Used to trigger updating FILENAME on the first read.
    start: bool,
}

impl<R: Read> LineReader for RegexSplitter<R> {
    type Line = DefaultLine;
    fn filename(&self) -> Str<'static> {
        self.name.clone()
    }
    fn check_utf8(&self) -> bool {
        self.reader.check_utf8()
    }

    // The _reuse variant not only allows us to reuse the memory in the `fields` vec, it also
    // allows us to reuse the old FieldUsage, which may have been overwritten with all() if the more
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
        let line = rc.with_regex(pat, |re| DefaultLine {
            line: self.read_line_regex(re),
            fields: Default::default(),
            used_fields: self.used_fields.clone(),
            diverged: false,
        })?;
        Ok((/* file changed */ start, line))
    }
    fn read_state(&self) -> i64 {
        self.reader.read_state()
    }
    fn next_file(&mut self) -> Result<bool> {
        // There is just one file. Set EOF.
        self.reader.force_eof();
        Ok(false)
    }
    fn set_used_fields(&mut self, used_fields: &FieldUsage) {
        self.used_fields = used_fields.clone();
    }
}

impl<R: Read> RegexSplitter<R> {
    pub fn new(r: R, chunk_size: usize, name: impl Into<Str<'static>>, check_utf8: bool) -> Self {
        RegexSplitter {
            reader: Reader::new(r, chunk_size, /*padding=*/ 0, check_utf8),
            name: name.into(),
            used_fields: Default::default(),
            start: true,
        }
    }

    pub fn read_line_regex(&mut self, pat: &Regex) -> Str<'static> {
        // We keep this as a separate method because it helps in writing tests.
        let (res, consumed) = self.read_line_inner(pat);
        self.reader.last_len = consumed;
        res
    }

    fn read_line_inner(&mut self, pat: &Regex) -> (Str<'static>, usize) {
        if self.reader.is_eof() {
            return (Str::default(), 0);
        }
        loop {
            let s = &self.reader.buf.as_bytes()[self.reader.start..self.reader.end];
            // Why this map invocation? Match objects hold a reference to the substring, which
            // makes it harder for us to call mutable methods like advance in the body, so just get
            // the start and end pointers.
            match pat.find(s).map(|m| (m.start(), m.end())) {
                // We need this check in case the regex matches across a chunk boundary.
                Some((start, end)) if end + self.reader.start < self.reader.end => {
                    // Valid offsets guaranteed by correctness of regex `find`.
                    let res = self
                        .reader
                        .buf
                        .slice_to_str(self.reader.start, self.reader.start + start);
                    self.reader.start += end;
                    return (res, end);
                }
                None => {
                    let consumed = self.reader.end - self.reader.start;
                    return match self.reader.reset() {
                        Ok(true) => {
                            // EOF: yield the rest of the buffer
                            let line = self
                                .reader
                                .buf
                                .slice_to_str(self.reader.start, self.reader.end);
                            self.reader.start = self.reader.end;
                            (line, consumed)
                        }
                        Ok(false) => {
                            // search the new (potentially larger) buffer.
                            // NB: isn't this wasteful? The new buffer could be as much as half
                            // already-read bytes, and we'll search those again in our next loop.
                            //
                            // That's true, and while that may be a safe optimization to make for
                            // many common regexes (e.g. single-character ones), it does not work
                            // if the the particular regex match is split across buffers.
                            // Furthermore, we have no way to detect the offset at which such a
                            // "partial match" might start. So, we take the performance hit here,
                            // noting that:
                            //
                            // (1) Most record separators are going to be small and relatively
                            // frequent, so that we will amortize the cost of rescanning bytes over
                            // several successful matches that were found over the course of a
                            // single traversal.
                            // (2) Many common splitting strategies have special-case strategies
                            // that avoid this kind of rescanning.
                            continue;
                        }
                        Err(_) => {
                            self.reader.state = ReaderState::ERROR;
                            (Str::default(), 0)
                        }
                    };
                }
                Some((start, end)) => {
                    return match self.reader.reset() {
                        Ok(true) => {
                            // Valid offsets guaranteed by correctness of regex `find`.
                            let res = self
                                .reader
                                .buf
                                .slice_to_str(self.reader.start, self.reader.start + start);
                            self.reader.start += end;
                            (res, end)
                        }
                        Ok(false) => {
                            // See comment in the previous branch.
                            continue;
                        }
                        Err(_) => {
                            self.reader.state = ReaderState::ERROR;
                            (Str::default(), 0)
                        }
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // need to benchmark batched splitting vs. regular splitting to get a feel for things.
    use super::*;
    use lazy_static::lazy_static;
    use regex::bytes::Regex;
    use std::io::Cursor;

    lazy_static! {
        static ref STR: String = String::from_utf8(bytes(1 << 20, 0.001, 0.05)).unwrap();
        static ref LINE: Regex = Regex::new("\n").unwrap();
        static ref SPACE: Regex = Regex::new(" ").unwrap();
        static ref BS: Regex = Regex::new("b+").unwrap();
    }

    // Helps type inference along.
    fn ref_str<'a>(s: &'a [u8]) -> Str<'a> {
        std::str::from_utf8(s).unwrap().into()
    }

    #[test]
    fn test_line_split_big_sep() {
        let mut buf = Vec::new();
        for _ in 0..10 {
            buf.extend("hi".as_bytes().iter().cloned());
            buf.extend((0..1024).map(|_| b'a'));
            buf.extend("there".as_bytes().iter().cloned());
            buf.extend((0..2048).map(|_| b'b'));
        }
        let bs = String::from_utf8(buf).unwrap();
        let c = Cursor::new(bs.clone());
        let mut rdr =
            RegexSplitter::new(c, /*chunk_size=*/ 512, "", /*check_utf8=*/ false);
        let mut lines = Vec::new();
        while !rdr.reader.is_eof() {
            let line = rdr.read_line_regex(&*BS).upcast();
            assert!(rdr.read_state() != -1);
            lines.push(line);
        }
        let mut expected: Vec<_> = BS.split(bs.as_bytes()).map(ref_str).collect();
        // remove trailing empty line
        assert_eq!(expected.pop(), Some(Str::default()));
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
    fn test_line_split() {
        let chunk_size = 1 << 9;
        let bs: String = crate::test_string_constants::PRIDE_PREJUDICE_CH2.into();
        let c = Cursor::new(bs.clone());
        let mut rdr = RegexSplitter::new(c, chunk_size, "", /*check_utf8=*/ true);
        let mut lines = Vec::new();
        while !rdr.reader.is_eof() {
            let line = rdr.read_line_regex(&*LINE).upcast();
            assert!(rdr.read_state() != -1);
            lines.push(line);
        }

        let expected: Vec<_> = LINE.split(bs.as_bytes()).map(ref_str).collect();
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
    fn test_clipped_chunk_split_pp() {
        // _random is more thorough, but this works as a sort of smoke test.

        let chunk_size = 1 << 9;

        let multi_byte = "學";
        assert!(multi_byte.len() > 1);
        let mut bs = Vec::new();
        bs.extend(crate::test_string_constants::PRIDE_PREJUDICE_CH2.bytes());
        let start = chunk_size - 1;
        for (i, b) in multi_byte.as_bytes().iter().enumerate() {
            bs[start + i] = *b;
        }

        let s = String::from_utf8(bs).unwrap();
        let c = Cursor::new(s.clone());
        let mut rdr = RegexSplitter::new(c, chunk_size, "", /*check_utf8=*/ true);
        let mut lines = Vec::new();
        while !rdr.reader.is_eof() {
            let line = rdr.read_line_regex(&*LINE).upcast();
            assert!(rdr.read_state() != -1);
            lines.push(line);
        }
        let expected: Vec<_> = LINE.split(s.as_bytes()).map(ref_str).collect();
        if lines != expected {
            eprintln!("lines.len={}, expected.len={}", lines.len(), expected.len());
            for (i, (l, e)) in lines.iter().zip(expected.iter()).enumerate() {
                if l != e {
                    eprintln!("mismatch at index {}:\ngot=<{}>\nwant=<{}>", i, l, e);
                }
            }
            assert!(false, "number of lines does not match");
        }
    }

    #[test]
    fn test_clipped_chunk_split_random() {
        const N_RUNS: usize = 5;
        for iter in 0..N_RUNS {
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
            let mut rdr = RegexSplitter::new(c, chunk_size, "", /*check_utf8=*/ true);
            let mut lines = Vec::new();
            while !rdr.reader.is_eof() {
                let line = rdr.read_line_regex(&*LINE).upcast();
                assert!(rdr.read_state() != -1);
                lines.push(line);
            }
            let expected: Vec<_> = LINE.split(s.as_bytes()).map(ref_str).collect();
            if lines != expected {
                eprintln!(
                    "Failed after {} runs. lines.len={}, expected.len={}",
                    iter,
                    lines.len(),
                    expected.len()
                );
                for (i, (l, e)) in lines.iter().zip(expected.iter()).enumerate() {
                    if l != e {
                        eprintln!("mismatch at index {}:\ngot=<{}>\nwant=<{}>", i, l, e);
                    }
                }
                assert!(false, "number of lines does not match");
            }
        }
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

#[cfg(all(feature = "unstable", test))]
mod bench {
    // need to benchmark batched splitting vs. regular splitting to get a feel for things.
    extern crate test;
    use lazy_static::lazy_static;
    use regex::bytes::Regex;
    use test::{black_box, Bencher};

    lazy_static! {
        static ref STR: String = String::from_utf8(bytes(1 << 20, 0.001, 0.05)).unwrap();
        static ref LINE: Regex = Regex::new("\n").unwrap();
        static ref SPACE: Regex = Regex::new(" ").unwrap();
        static ref BS: Regex = Regex::new("b+").unwrap();
    }

    #[bench]
    fn bench_find_iter(b: &mut Bencher) {
        let bs = STR.as_bytes();
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
        let bs = STR.as_bytes();
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
        let bs = STR.as_bytes();
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
