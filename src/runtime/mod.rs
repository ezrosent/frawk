use crate::common::{Either, FileSpec, Result};
use hashbrown::HashMap;
use regex::bytes::Regex;
use std::cell::{Cell, RefCell};
use std::fs::File;
use std::hash::Hash;
use std::io;
use std::iter::FromIterator;
use std::process::ChildStdout;
use std::rc::Rc;
use std::str;

mod command;
pub mod float_parse;
pub mod printf;
pub mod splitter;
pub mod str_impl;
pub mod string_search;
pub mod utf8;
pub mod writers;

use crate::pushdown::FieldSet;
use splitter::regex::RegexSplitter;

// TODO: remove the pub use for Variables here.
pub(crate) use crate::builtins::Variables;
pub(crate) use float_parse::{hextoi, strtod, strtoi};
pub(crate) use printf::FormatArg;
pub use splitter::{
    batch::{escape_csv, escape_tsv},
    ChainedReader, Line, LineReader,
};
pub use str_impl::{Str, UniqueStr};
pub use command::run_command;

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
    pub fn from_vec(v: Vec<T>) -> Self {
        Either::Left(v)
    }
    pub fn get_cleared_vec(&mut self) -> Vec<T> {
        match self {
            Either::Left(v) => {
                let mut res = std::mem::replace(v, Vec::new());
                res.clear();
                res
            }
            Either::Right(_) => {
                *self = Either::Left(Vec::new());
                Vec::new()
            }
        }
    }
    pub(crate) fn len(&self) -> usize {
        for_either!(self, |x| x.len())
    }
}

impl LazyVec<Str<'static>> {
    pub(crate) fn join_by<F>(
        &self,
        sep: &Str<'static>,
        start: usize,
        end: usize,
        mut by: F,
    ) -> Str<'static>
    where
        F: FnMut(Str<'static>) -> Str<'static>,
    {
        // assumes zero-indexing, doesn't do len-checks.
        match self {
            Either::Left(v) => sep.join(v[start..end].iter().cloned().map(by)),
            Either::Right(m) => {
                let r = m.0.borrow();
                let mut v: Vec<_> = r.keys().cloned().collect();
                v.sort();
                sep.join(
                    v.into_iter()
                        .filter(|ix| *ix >= start as Int && *ix < end as Int)
                        .map(|i| by(r[&i].clone())),
                )
            }
        }
    }
}
impl<'a> LazyVec<Str<'a>> {
    pub(crate) fn join_all(&self, sep: &Str<'a>) -> Str<'a> {
        match self {
            Either::Left(v) => sep.join(v.iter().cloned()),
            Either::Right(m) => {
                let r = m.0.borrow();
                let mut v: Vec<_> = r.keys().cloned().collect();
                v.sort();
                sep.join(v.into_iter().map(|i| r[&i].clone()))
            }
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
                    //
                    // Note that this only works with $, and will stop working if we implement
                    // something like `join` in the language (it inserts things that the user never
                    // inserted).
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
            // eta-expansion required to get this compiling..
            |x| f(x),
        )
    }

    pub(crate) fn get_line<'a, LR: LineReader>(
        &mut self,
        file: &Str<'a>,
        pat: &Str<'a>,
        reg: &mut FileRead<LR>,
        is_file: bool,
    ) -> Result<Str<'a>> {
        Ok(if is_file {
            reg.with_file(file, |reader| {
                self.with_regex(pat, |re| reader.read_line_regex(re))
            })?
        } else {
            reg.with_cmd(file, |reader| {
                self.with_regex(pat, |re| reader.read_line_regex(re))
            })?
        }
        .clone()
        .upcast())
    }

    // This only gets used if getline is invoked explicitly without an input file argument.
    pub(crate) fn get_line_stdin<'a, LR: LineReader>(
        &mut self,
        pat: &Str<'a>,
        reg: &mut FileRead<LR>,
    ) -> Result<(/* file changed */ bool, Str<'a>)> {
        let (changed, mut line) = reg.stdin.read_line(pat, self)?;
        // NB both of these `pat`s are "wrong" but we are fine because they are only used
        // when the column is nonzero, or someone has overwritten a nonzero column.
        Ok((changed, line.get_col(0, pat, pat, self)?.clone().upcast()))
    }
    pub(crate) fn get_line_stdin_reuse<'a, LR: LineReader>(
        &mut self,
        pat: &Str<'a>,
        reg: &mut FileRead<LR>,
        old_line: &mut LR::Line,
    ) -> Result</*file changed */ bool> {
        reg.stdin.read_line_reuse(pat, self, old_line)
    }
    fn split_internal<'a>(
        &mut self,
        pat: &Str,
        s: &Str<'a>,
        used_fields: &FieldSet,
        mut push: impl FnMut(Str<'a>),
    ) -> Result<()> {
        if pat == &Str::from(" ") {
            self.with_regex(&Str::from(r#"[ \t]+"#), |re| {
                s.split(
                    re,
                    |s, is_empty| {
                        if !is_empty {
                            push(s);
                            1
                        } else {
                            0
                        }
                    },
                    used_fields,
                )
            })
        } else {
            self.with_regex(pat, |re| {
                s.split(
                    re,
                    |s, _| {
                        push(s);
                        1
                    },
                    used_fields,
                )
            })
        }
    }
    pub(crate) fn split_regex<'a>(
        &mut self,
        pat: &Str,
        s: &Str<'a>,
        used_fields: &FieldSet,
        v: &mut LazyVec<Str<'a>>,
    ) -> Result<()> {
        self.split_internal(pat, s, used_fields, |s| v.push(s))
    }

    pub(crate) fn split_regex_intmap<'a>(
        &mut self,
        pat: &Str<'a>,
        s: &Str<'a>,
        m: &IntMap<Str<'a>>,
    ) -> Result<()> {
        let mut i = 0i64;
        self.split_internal(pat, s, &FieldSet::all(), |s| {
            i += 1;
            m.insert(i, s);
        })
    }

    pub(crate) fn split_regex_strmap<'a>(
        &mut self,
        pat: &Str<'a>,
        s: &Str<'a>,
        m: &StrMap<'a, Str<'a>>,
    ) -> Result<()> {
        let mut i = 0i64;
        self.split_internal(pat, s, &FieldSet::all(), |s| {
            i += 1;
            m.insert(convert::<i64, Str<'_>>(i), s);
        })
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
            s.with_bytes(|bs| match re.find(bs) {
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
        self.with_regex(pat, |re| s.with_bytes(|bs| re.is_match(bs)))
    }
}

#[derive(Clone)]
pub(crate) struct FileWrite(writers::Registry);

impl Default for FileWrite {
    fn default() -> FileWrite {
        FileWrite::new(writers::default_factory())
    }
}

impl FileWrite {
    pub(crate) fn flush_stdout(&mut self) -> Result<()> {
        self.0.get_file(None)?.flush()
    }
    pub(crate) fn close(&mut self, path: &Str) -> Result<()> {
        self.0.close(path)
    }
    pub(crate) fn new(ff: impl writers::FileFactory) -> FileWrite {
        FileWrite(writers::Registry::from_factory(ff))
    }

    pub(crate) fn shutdown(&mut self) -> Result<()> {
        self.0.destroy_and_flush_all_files()
    }

    pub(crate) fn printf(
        &mut self,
        path: Option<(&Str, FileSpec)>,
        spec: &Str,
        pa: &[printf::FormatArg],
    ) -> Result<()> {
        let (handle, fspec) = if let Some((out_file, fspec)) = path {
            (self.0.get_handle(Some(out_file), fspec)?, fspec)
        } else {
            (
                self.0.get_handle(None, FileSpec::default())?,
                FileSpec::default(),
            )
        };
        let mut text = str_impl::DynamicBuf::default();
        spec.with_bytes(|spec| printf::printf(&mut text, spec, pa))?;
        let s = unsafe { text.into_str() };
        handle.write(&s, fspec)
    }

    pub(crate) fn write_str_stdout(&mut self, s: &Str) -> Result<()> {
        self.0
            .get_handle(None, FileSpec::default())?
            .write(s, /*append=*/ FileSpec::Append)
    }

    pub(crate) fn write_str(&mut self, path: &Str, s: &Str, spec: FileSpec) -> Result<()> {
        self.0.get_handle(Some(path), spec)?.write(s, spec)
    }
}

pub const CHUNK_SIZE: usize = 8 << 10;

#[derive(Default)]
pub(crate) struct Inputs {
    files: Registry<RegexSplitter<File>>,
    commands: Registry<RegexSplitter<ChildStdout>>,
}

pub(crate) struct FileRead<LR = RegexSplitter<Box<dyn io::Read + Send>>> {
    pub(crate) inputs: Inputs,
    stdin: LR,
}

impl<LR: LineReader> FileRead<LR> {
    pub(crate) fn try_resize(&self, size: usize) -> Vec<impl FnOnce() -> Self + Send> {
        self.stdin
            .request_handles(size)
            .into_iter()
            .map(|x| {
                move || FileRead {
                    inputs: Default::default(),
                    stdin: x(),
                }
            })
            .collect()
    }

    pub(crate) fn close(&mut self, path: &Str) {
        self.inputs.files.remove(path);
    }

    pub(crate) fn new(stdin: LR, used_fields: &FieldSet) -> FileRead<LR> {
        let mut res = FileRead {
            inputs: Default::default(),
            stdin,
        };
        res.stdin.set_used_fields(used_fields);
        res
    }

    pub(crate) fn stdin_filename(&self) -> Str<'static> {
        self.stdin.filename()
    }

    pub(crate) fn read_err_stdin<'a>(&mut self) -> Int {
        self.stdin.read_state()
    }

    pub(crate) fn read_err<'a>(&mut self, path: &Str<'a>) -> Result<Int> {
        self.with_file(path, |reader| Ok(reader.read_state()))
    }
    pub(crate) fn read_err_cmd<'a>(&mut self, cmd: &Str<'a>) -> Result<Int> {
        self.with_cmd(cmd, |reader| Ok(reader.read_state()))
    }

    pub(crate) fn next_file(&mut self) -> Result<()> {
        let _ = self.stdin.next_file()?;
        Ok(())
    }

    fn with_cmd<'a, R>(
        &mut self,
        cmd: &Str<'a>,
        f: impl FnMut(&mut RegexSplitter<ChildStdout>) -> Result<R>,
    ) -> Result<R> {
        let check_utf8 = self.stdin.check_utf8();
        self.inputs.commands.get_fallible(
            cmd,
            |s| match command::command_for_read(s.as_bytes()) {
                Ok(r) => Ok(RegexSplitter::new(
                    r,
                    CHUNK_SIZE,
                    cmd.clone().unmoor(),
                    check_utf8,
                )),
                Err(e) => err!("failed to crate command for reading: {}", e),
            },
            f,
        )
    }

    fn with_file<'a, R>(
        &mut self,
        path: &Str<'a>,
        f: impl FnMut(&mut RegexSplitter<File>) -> Result<R>,
    ) -> Result<R> {
        let check_utf8 = self.stdin.check_utf8();
        self.inputs.files.get_fallible(
            path,
            |s| match File::open(s) {
                Ok(f) => Ok(RegexSplitter::new(
                    f,
                    CHUNK_SIZE,
                    path.clone().unmoor(),
                    check_utf8,
                )),
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
                let (val, res) = v.key().with_bytes(|raw_str| {
                    let s = match str::from_utf8(raw_str) {
                        Ok(s) => s,
                        Err(e) => return err!("invalid UTF-8 for file or regex: {}", e),
                    };
                    let mut val = new(s)?;
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
        s.with_bytes(strtod)
    }
}
impl<'a> Convert<Str<'a>, Int> for _Carrier {
    fn convert(s: Str<'a>) -> Int {
        s.with_bytes(strtoi)
    }
}
impl<'b, 'a> Convert<&'b Str<'a>, Float> for _Carrier {
    fn convert(s: &'b Str<'a>) -> Float {
        s.with_bytes(strtod)
    }
}
impl<'b, 'a> Convert<&'b Str<'a>, Int> for _Carrier {
    fn convert(s: &'b Str<'a>) -> Int {
        s.with_bytes(strtoi)
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
    pub(crate) fn iter<'a, F, R>(&'a self, f: F) -> R
    where
        F: FnOnce(hashbrown::hash_map::Iter<K, V>) -> R,
    {
        f(self.0.borrow().iter())
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
    #[cfg(feature = "llvm_backend")]
    pub(crate) fn to_vec(&self) -> Vec<K> {
        self.0.borrow().keys().cloned().collect()
    }
}

impl<K: Hash + Eq, V> From<HashMap<K, V>> for SharedMap<K, V> {
    fn from(m: HashMap<K, V>) -> SharedMap<K, V> {
        SharedMap(Rc::new(RefCell::new(m)))
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
    pub(crate) fn has_next(&self) -> bool {
        self.cur.get() < self.items.len()
    }
    pub(crate) unsafe fn get_next(&self) -> &S {
        debug_assert!(self.has_next());
        let cur = self.cur.get();
        let res = self.items.get_unchecked(cur);
        self.cur.set(cur + 1);
        res
    }
}
