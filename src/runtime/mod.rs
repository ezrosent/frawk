use crate::common::{Either, Result};
use hashbrown::HashMap;
use regex::Regex;
use std::cell::{Cell, RefCell};
use std::fs::File;
use std::hash::Hash;
use std::io::{self, BufWriter, Write};
use std::iter::FromIterator;
use std::rc::Rc;

pub mod shared;
pub mod splitter;
pub mod str_impl;
pub mod strton;
pub mod utf8;

pub(crate) use str_impl::Str;

// TODO(ezr): this IntMap can probably be unboxed, but wait until we decide whether or not to
// specialize the IntMap implementation.
pub(crate) type LazyVec<T> = Either<Vec<T>, IntMap<T>>;

impl<T> LazyVec<T> {
    pub(crate) fn clear(&mut self) {
        *self = match self {
            Either::Left(v) => {
                v.clear();
                return;
            }
            Either::Right(_) => Either::Left(Default::default()),
        }
    }
    pub(crate) fn len(&self) -> usize {
        for_either!(self, |x| x.len())
    }
}

impl<T: Clone> LazyVec<T> {
    pub(crate) fn get(&self, ix: usize) -> Option<T> {
        match self {
            Either::Left(v) => v.get(ix).cloned(),
            Either::Right(m) => m.get(&(ix as i64)),
        }
    }
}
impl<T> LazyVec<T> {
    pub(crate) fn new() -> LazyVec<T> {
        Either::Left(Default::default())
    }
}

impl<T: Default> LazyVec<T> {
    pub(crate) fn push(&mut self, t: T) {
        self.insert(self.len(), t)
    }
    pub(crate) fn insert(&mut self, ix: usize, t: T) {
        *self = loop {
            match self {
                Either::Left(v) => {
                    if ix < v.len() {
                        v[ix] = t;
                    } else if ix == v.len() {
                        v.push(t);
                    // XXX: this is a heuristic to keep a dense representation, perhaps we should
                    // remove and just upgrade?
                    } else if ix < v.len() + 16 {
                        while ix > v.len() {
                            v.push(Default::default())
                        }
                        v.push(t);
                    } else {
                        break Either::Right(
                            v.drain(..)
                                .enumerate()
                                .map(|(ix, t)| (ix as i64, t))
                                .collect::<IntMap<T>>(),
                        );
                    }
                }
                Either::Right(m) => {
                    m.insert(ix as i64, t);
                }
            }
            return;
        };
    }
}

pub(crate) struct Variables<'a> {
    pub argc: Int,
    pub argv: IntMap<Str<'a>>,
    pub fs: Str<'a>,
    pub ofs: Str<'a>,
    pub rs: Str<'a>,
    pub nf: Int,
    pub nr: Int,
    pub filename: Str<'a>,
}

impl<'a> Default for Variables<'a> {
    fn default() -> Variables<'a> {
        Variables {
            argc: 0,
            argv: Default::default(),
            fs: "[ \t]+".into(),
            ofs: " ".into(),
            rs: "\n".into(),
            nr: 0,
            nf: 0,
            filename: Default::default(),
        }
    }
}

#[derive(Default)]
pub(crate) struct RegexCache(Registry<Regex>);

impl RegexCache {
    pub(crate) fn with_regex<T>(&mut self, pat: &Str, mut f: impl FnMut(&Regex) -> T) -> Result<T> {
        self.0.get(
            pat,
            |s| match Regex::new(s) {
                Ok(r) => Ok(r),
                Err(e) => err!("{}", e),
            },
            |x| f(x),
        )
    }
    pub(crate) fn get_line<'a>(
        &mut self,
        file: &Str<'a>,
        pat: &Str<'a>,
        reg: &mut FileRead,
    ) -> Result<Str<'a>> {
        self.0.get_fallible(
            pat,
            |s| match Regex::new(s) {
                Ok(r) => Ok(r),
                Err(e) => err!("{}", e),
            },
            |re| reg.get_line(file, re),
        )
    }
    pub(crate) fn get_line_stdin<'a>(
        &mut self,
        pat: &Str<'a>,
        reg: &mut FileRead,
    ) -> Result<Str<'a>> {
        self.0.get(
            pat,
            |s| match Regex::new(s) {
                Ok(r) => Ok(r),
                Err(e) => err!("{}", e),
            },
            |re| reg.get_line_stdin(re),
        )
    }
    pub(crate) fn split_regex<'a>(
        &mut self,
        pat: &Str<'a>,
        s: &Str<'a>,
        v: &mut LazyVec<Str<'a>>,
    ) -> Result<()> {
        self.with_regex(pat, |re| s.split(re, |s| v.push(s)))?;
        Ok(())
    }

    pub(crate) fn split_regex_intmap<'a>(
        &mut self,
        pat: &Str<'a>,
        s: &Str<'a>,
        m: &IntMap<Str<'a>>,
    ) -> Result<()> {
        let mut i = 0i64;
        self.with_regex(pat, |re| {
            s.split(re, |s| {
                i += 1;
                m.insert(i, s);
            })
        })?;
        Ok(())
    }

    pub(crate) fn split_regex_strmap<'a>(
        &mut self,
        pat: &Str<'a>,
        s: &Str<'a>,
        m: &StrMap<'a, Str<'a>>,
    ) -> Result<()> {
        let mut i = 0i64;
        self.with_regex(pat, |re| {
            s.split(re, |s| {
                i += 1;
                m.insert(convert::<i64, Str<'_>>(i), s);
            })
        })?;
        Ok(())
    }

    pub(crate) fn match_regex(&mut self, pat: &Str, s: &Str) -> Result<bool> {
        self.with_regex(pat, |re| s.with_str(|s| re.is_match(s)))
    }
}

pub(crate) struct FileWrite {
    files: Registry<io::BufWriter<File>>,
    stdout: Box<dyn io::Write>,
}

impl FileWrite {
    pub(crate) fn new(w: impl io::Write + 'static) -> FileWrite {
        let stdout: Box<dyn io::Write> = Box::new(w);
        FileWrite {
            files: Default::default(),
            stdout,
        }
    }

    fn with_handle(
        &mut self,
        append: bool,
        path: &Str,
        f: impl FnOnce(&mut io::BufWriter<File>) -> Result<()>,
    ) -> Result<()> {
        self.files.get_fallible(
            path,
            |s| match std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .append(append)
                .open(s)
            {
                Ok(f) => Ok(BufWriter::new(f)),
                Err(e) => err!(
                    "failed to open file {} for {}: {}",
                    s,
                    if append { "append" } else { "write" },
                    e
                ),
            },
            f,
        )
    }

    pub(crate) fn write_str_stdout(&mut self, s: &Str) -> Result<()> {
        if let Err(e) = s.with_str(|s| self.stdout.write_all(s.as_bytes())) {
            err!("failed to write to stdout (stdout closed?): {}", e)
        } else {
            Ok(())
        }
    }

    pub(crate) fn write_str(&mut self, path: &Str, s: &Str, append: bool) -> Result<()> {
        self.with_handle(append, path, |writer| {
            if let Err(e) = s.with_str(|s| writer.write_all(s.as_bytes())) {
                err!("failed to write to file: {}", e)
            } else {
                Ok(())
            }
        })
    }

    pub(crate) fn write_line(&mut self, path: &Str, s: &Str, append: bool) -> Result<()> {
        self.with_handle(append, path, |writer| {
            s.with_str(|s| {
                if let Err(e) = writer.write_all(s.as_bytes()) {
                    err!("failed to write to file: {}", e)
                } else if let Err(e) = writer.write_all("\n".as_bytes()) {
                    err!("failed to write newline to file: {}", e)
                } else {
                    Ok(())
                }
            })
        })
    }
}

const CHUNK_SIZE: usize = 2 << 10;

pub(crate) struct FileRead {
    files: Registry<splitter::Reader<File>>,
    stdin: Option<splitter::Reader<Box<dyn io::Read>>>,
}

impl FileRead {
    pub(crate) fn new(r: impl io::Read + 'static) -> FileRead {
        let d: Box<dyn io::Read> = Box::new(r);
        FileRead {
            files: Default::default(),
            stdin: splitter::Reader::new(d, CHUNK_SIZE).ok(),
        }
    }
    pub(crate) fn get_line_stdin<'a>(&mut self, pat: &Regex) -> Str<'a> {
        match &mut self.stdin {
            Some(s) => s.read_line(pat),
            None => "".into(),
        }
    }
    pub(crate) fn read_err_stdin<'a>(&mut self) -> Int {
        self.stdin
            .as_ref()
            .map(|s| s.read_state())
            .unwrap_or(splitter::ReaderState::EOF as Int)
    }
    pub(crate) fn get_line<'a>(&mut self, path: &Str<'a>, pat: &Regex) -> Result<Str<'a>> {
        self.with_file(path, |reader| Ok(reader.read_line(pat)))
    }
    pub(crate) fn read_err<'a>(&mut self, path: &Str<'a>) -> Result<Int> {
        self.with_file(path, |reader| Ok(reader.read_state()))
    }
    fn with_file<'a, R>(
        &mut self,
        path: &Str<'a>,
        f: impl FnMut(&mut splitter::Reader<File>) -> Result<R>,
    ) -> Result<R> {
        self.files.get_fallible(
            path,
            |s| match File::open(s) {
                Ok(f) => Ok(splitter::Reader::new(f, CHUNK_SIZE)?),
                Err(e) => err!("failed to open file '{}': {}", s, e),
            },
            f,
        )
    }
}

pub(crate) struct Registry<T> {
    // TODO(ezr): we could potentially increase speed here if we did pointer equality (and
    // length) for lookups.
    // We could be fine having duplicates for Regex. We could also also intern strings
    // as we go by swapping out one Rc for another as we encounter them. That would keep the
    // fast path fast, but we would have to make sure we weren't keeping any Refs alive.
    cached: HashMap<Rc<str>, T>,
}
impl<T> Default for Registry<T> {
    fn default() -> Self {
        Registry {
            cached: Default::default(),
        }
    }
}

impl<T> Registry<T> {
    fn get<R>(
        &mut self,
        s: &Str,
        new: impl FnMut(&str) -> Result<T>,
        getter: impl FnOnce(&mut T) -> R,
    ) -> Result<R> {
        self.get_fallible(s, new, |t| Ok(getter(t)))
    }
    fn get_fallible<R>(
        &mut self,
        s: &Str,
        mut new: impl FnMut(&str) -> Result<T>,
        getter: impl FnOnce(&mut T) -> Result<R>,
    ) -> Result<R> {
        use hashbrown::hash_map::Entry;
        let k_str = s.clone_str();
        match self.cached.entry(k_str) {
            Entry::Occupied(mut o) => getter(o.get_mut()),
            Entry::Vacant(v) => {
                let raw_str = &*v.key();
                let mut val = new(raw_str)?;
                let res = getter(&mut val);
                v.insert(val);
                res
            }
        }
    }
}

pub(crate) trait Convert<S, T> {
    fn convert(s: S) -> T;
}

pub(crate) struct _Carrier;

impl Convert<Float, Int> for _Carrier {
    fn convert(f: Float) -> Int {
        f as Int
    }
}
impl Convert<Int, Float> for _Carrier {
    fn convert(i: Int) -> Float {
        i as Float
    }
}
impl<'a> Convert<Int, Str<'a>> for _Carrier {
    fn convert(i: Int) -> Str<'a> {
        format!("{}", i).into()
    }
}
impl<'a> Convert<Float, Str<'a>> for _Carrier {
    fn convert(f: Float) -> Str<'a> {
        let mut buffer = ryu::Buffer::new();
        let printed = buffer.format(f);
        let p_str: String = printed.into();
        p_str.into()
    }
}
impl<'a> Convert<Str<'a>, Float> for _Carrier {
    fn convert(s: Str<'a>) -> Float {
        s.with_str(strton::strtod)
    }
}
impl<'a> Convert<Str<'a>, Int> for _Carrier {
    fn convert(s: Str<'a>) -> Int {
        s.with_str(strton::strtoi)
    }
}
impl<'b, 'a> Convert<&'b Str<'a>, Float> for _Carrier {
    fn convert(s: &'b Str<'a>) -> Float {
        s.with_str(strton::strtod)
    }
}
impl<'b, 'a> Convert<&'b Str<'a>, Int> for _Carrier {
    fn convert(s: &'b Str<'a>) -> Int {
        s.with_str(strton::strtoi)
    }
}

pub(crate) fn convert<S, T>(s: S) -> T
where
    _Carrier: Convert<S, T>,
{
    _Carrier::convert(s)
}

// AWK arrays are inherently shared and mutable, so we have to do this, even if it is a code smell.
pub(crate) struct SharedMap<K, V>(Rc<RefCell<HashMap<K, V>>>);

impl<K, V> Default for SharedMap<K, V> {
    fn default() -> SharedMap<K, V> {
        SharedMap(Rc::new(RefCell::new(Default::default())))
    }
}

impl<K, V> Clone for SharedMap<K, V> {
    fn clone(&self) -> Self {
        SharedMap(self.0.clone())
    }
}

impl<K: Hash + Eq, V> SharedMap<K, V> {
    pub(crate) fn len(&self) -> usize {
        self.0.borrow().len()
    }
    pub(crate) fn insert(&self, k: K, v: V) {
        self.0.borrow_mut().insert(k, v);
    }
    pub(crate) fn delete(&self, k: &K) {
        self.0.borrow_mut().remove(k);
    }
}
impl<K: Hash + Eq, V: Clone> SharedMap<K, V> {
    pub(crate) fn get(&self, k: &K) -> Option<V> {
        self.0.borrow().get(k).cloned()
    }
}
impl<K: Hash + Eq + Clone, V> SharedMap<K, V> {
    pub(crate) fn to_iter(&self) -> Iter<K> {
        self.0.borrow().keys().cloned().collect()
    }
}

impl<K: Hash + Eq, V> FromIterator<(K, V)> for SharedMap<K, V> {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (K, V)>,
    {
        SharedMap(Rc::new(RefCell::new(
            iter.into_iter().collect::<HashMap<K, V>>(),
        )))
    }
}

pub(crate) type Int = i64;
pub(crate) type Float = f64;
pub(crate) type IntMap<V> = SharedMap<Int, V>;
pub(crate) type StrMap<'a, V> = SharedMap<Str<'a>, V>;

pub(crate) struct Iter<S> {
    cur: Cell<usize>,
    items: Vec<S>,
}

impl<S> Default for Iter<S> {
    fn default() -> Iter<S> {
        None.into_iter().collect()
    }
}

impl<S> FromIterator<S> for Iter<S> {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = S>,
    {
        Iter {
            cur: Cell::new(0),
            items: Vec::from_iter(iter),
        }
    }
}

impl<S> Iter<S> {
    #[inline]
    pub(crate) fn has_next(&self) -> bool {
        self.cur.get() < self.items.len()
    }
    #[inline]
    pub(crate) unsafe fn get_next(&self) -> &S {
        debug_assert!(self.has_next());
        let cur = self.cur.get();
        let res = self.items.get_unchecked(cur);
        self.cur.set(cur + 1);
        res
    }
}
