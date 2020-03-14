use crate::common::{Either, Result};
use hashbrown::HashMap;
use regex::Regex;
use std::cell::{Cell, RefCell};
use std::fs::File;
use std::hash::Hash;
use std::io::{self, BufWriter, Write};
use std::iter::FromIterator;
use std::rc::Rc;

pub mod csv;
pub mod float_parse;
pub mod printf;
pub mod splitter;
pub mod str_impl;
pub mod utf8;

pub(crate) use crate::builtins::Variables;
pub(crate) use float_parse::{strtod, strtoi};
pub(crate) use printf::FormatArg;
pub use str_impl::Str;

pub(crate) trait Line<'a> {
    fn as_str(&self) -> &Str<'a>;
    fn split(&self, pat: &Str, rc: &mut RegexCache, push: impl FnMut(Str<'a>)) -> Result<()>;
    fn assign_from_str(&mut self, s: &Str<'a>);
}

pub(crate) trait LineReader {
    type Line: for<'a> Line<'a>;
    fn read_line(&mut self, pat: &Str, rc: &mut RegexCache) -> Result<Self::Line>;
    fn read_state(&self) -> i64;
}

impl<'a> Line<'a> for Str<'static> {
    fn as_str(&self) -> &Str<'a> {
        self.upcast_ref()
    }
    fn split(&self, pat: &Str, rc: &mut RegexCache, mut push: impl FnMut(Str<'a>)) -> Result<()> {
        rc.with_regex(pat, |re| self.split(re, |s| push(s.upcast())))
    }
    fn assign_from_str(&mut self, s: &Str<'a>) {
        *self = s.clone().unmoor();
    }
}

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

impl<'a> LazyVec<Str<'a>> {
    pub(crate) fn join(&self, sep: &Str<'a>) -> Str<'a> {
        match self {
            Either::Left(v) => sep.join(v.iter()),
            Either::Right(m) => sep.join(m.0.borrow().values()),
        }
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

#[derive(Default)]
pub struct RegexCache(Registry<Regex>);

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
    pub(crate) fn get_line<'a, LR: LineReader>(
        &mut self,
        file: &Str<'a>,
        pat: &Str<'a>,
        reg: &mut FileRead<LR>,
    ) -> Result<Str<'a>> {
        Ok(reg
            .with_file(file, |reader| reader.read_line(pat, self))?
            .as_str()
            .clone())
    }
    pub(crate) fn get_line_stdin<'a, LR: LineReader>(
        &mut self,
        pat: &Str<'a>,
        reg: &mut FileRead<LR>,
    ) -> Result<Str<'a>> {
        Ok(reg.stdin.read_line(pat, self)?.as_str().clone())
    }
    pub(crate) fn split_regex<'a>(
        &mut self,
        pat: &Str<'a>,
        s: &Str<'a>,
        v: &mut LazyVec<Str<'a>>,
    ) -> Result<()> {
        self.with_regex(pat, |re| s.split(re, |s| v.push(s)))
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
        })
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

    pub(crate) fn regex_match_loc(
        &mut self,
        vars: &mut Variables,
        pat: &Str,
        s: &Str,
    ) -> Result<Int> {
        use crate::builtins::Variable;
        // We use the awk semantics for `match`. If we match
        let (start, len) = self.with_regex(pat, |re| {
            s.with_str(|s| match re.find(s) {
                Some(m) => {
                    let start = m.start() as Int;
                    let end = m.end() as Int;
                    (start + 1, end - start)
                }
                None => (0, -1),
            })
        })?;
        vars.store_int(Variable::RSTART, start)?;
        vars.store_int(Variable::RLENGTH, len)?;
        Ok(start)
    }

    pub(crate) fn is_regex_match(&mut self, pat: &Str, s: &Str) -> Result<bool> {
        self.with_regex(pat, |re| s.with_str(|s| re.is_match(s)))
    }
}

pub(crate) struct FileWrite {
    files: Registry<io::BufWriter<File>>,
    stdout: Box<dyn io::Write>,
}

impl FileWrite {
    pub(crate) fn close(&mut self, path: &Str) {
        self.files.remove(path);
    }
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

    pub(crate) fn printf(
        &mut self,
        path: Option<(&Str, bool)>,
        spec: &Str,
        pa: &[printf::FormatArg],
    ) -> Result<()> {
        if let Some((out_file, append)) = path {
            self.with_handle(append, out_file, |writer| {
                spec.with_str(|spec| printf::printf(writer, spec, pa))
            })
        } else {
            spec.with_str(|spec| printf::printf(&mut self.stdout, spec, pa))
        }
    }

    pub(crate) fn write_str_stdout(&mut self, s: &Str) -> Result<()> {
        if let Err(e) = s.with_str(|s| self.stdout.write_all(s.as_bytes())) {
            err!("failed to write to stdout (stdout closed?): {}", e)
        } else {
            Ok(())
        }
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

const CHUNK_SIZE: usize = 16 << 10;

pub(crate) struct FileRead<LR = splitter::Reader<Box<dyn io::Read>>> {
    files: Registry<splitter::Reader<File>>,
    stdin: LR,
}

impl FileRead {
    pub(crate) fn new_transitional(r: impl io::Read + 'static) -> FileRead {
        let d: Box<dyn io::Read> = Box::new(r);
        FileRead {
            files: Default::default(),
            stdin: splitter::Reader::new(d, CHUNK_SIZE),
        }
    }
}

impl<LR: LineReader> FileRead<LR> {
    pub(crate) fn close(&mut self, path: &Str) {
        self.files.remove(path);
    }
    pub(crate) fn new(stdin: LR) -> FileRead<LR> {
        FileRead {
            files: Default::default(),
            stdin,
        }
    }
    pub(crate) fn read_err_stdin<'a>(&mut self) -> Int {
        self.stdin.read_state()
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
                Ok(f) => Ok(splitter::Reader::new(f, CHUNK_SIZE)),
                Err(e) => err!("failed to open file '{}': {}", s, e),
            },
            f,
        )
    }
}

pub(crate) struct Registry<T> {
    // TODO(ezr): use the raw bucket interface so we can avoid calls to `unmoor` here.
    // TODO(ezr): we could potentially increase speed here if we did pointer equality (and
    // length) for lookups.
    // We could be fine having duplicates for Regex. We could also also intern strings
    // as we go by swapping out one Rc for another as we encounter them. That would keep the
    // fast path fast, but we would have to make sure we weren't keeping any Refs alive.
    cached: HashMap<Str<'static>, T>,
}
impl<T> Default for Registry<T> {
    fn default() -> Self {
        Registry {
            cached: Default::default(),
        }
    }
}

impl<T> Registry<T> {
    fn remove(&mut self, s: &Str) {
        self.cached.remove(&s.clone().unmoor());
    }
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
        let k_str = s.clone().unmoor();
        match self.cached.entry(k_str) {
            Entry::Occupied(mut o) => getter(o.get_mut()),
            Entry::Vacant(v) => {
                let (val, res) = v.key().with_str(|raw_str| {
                    let mut val = new(raw_str)?;
                    let res = getter(&mut val);
                    Ok((val, res))
                })?;
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

// See str_impl.rs for how these first two are implemented.
impl<'a> Convert<Int, Str<'a>> for _Carrier {
    fn convert(i: Int) -> Str<'a> {
        i.into()
    }
}
impl<'a> Convert<Float, Str<'a>> for _Carrier {
    fn convert(f: Float) -> Str<'a> {
        f.into()
    }
}
impl<'a> Convert<Str<'a>, Float> for _Carrier {
    fn convert(s: Str<'a>) -> Float {
        s.with_str(strtod)
    }
}
impl<'a> Convert<Str<'a>, Int> for _Carrier {
    fn convert(s: Str<'a>) -> Int {
        s.with_str(strtoi)
    }
}
impl<'b, 'a> Convert<&'b Str<'a>, Float> for _Carrier {
    fn convert(s: &'b Str<'a>) -> Float {
        s.with_str(strtod)
    }
}
impl<'b, 'a> Convert<&'b Str<'a>, Int> for _Carrier {
    fn convert(s: &'b Str<'a>) -> Int {
        s.with_str(strtoi)
    }
}

pub(crate) fn convert<S, T>(s: S) -> T
where
    _Carrier: Convert<S, T>,
{
    _Carrier::convert(s)
}

// AWK arrays are inherently shared and mutable, so we have to do this, even if it is a code smell.
// NB These are repr(transparent) because we pass them around as void* when compiling with LLVM.
#[repr(transparent)]
#[derive(Debug)]
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
    pub(crate) fn to_vec(&self) -> Vec<K> {
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
