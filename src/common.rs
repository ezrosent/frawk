//! This file contains common type definitions and utilities used in other parts of the project.
use hashbrown::HashSet;
use std::collections::VecDeque;
use std::fmt;
use std::hash::Hash;
use std::sync::{
    atomic::{AtomicBool, AtomicI64, Ordering},
    Arc, Condvar, Mutex,
};

pub(crate) type NumTy = u32;
pub(crate) type NodeIx = petgraph::graph::NodeIndex<NumTy>;
pub(crate) type Graph<V, E> = petgraph::Graph<V, E, petgraph::Directed, NumTy>;
pub(crate) type Result<T> = std::result::Result<T, CompileError>;

#[derive(Copy, Clone)]
pub enum ExecutionStrategy {
    /// Execute the script in a single thread. This is the default.
    Serial,
    /// Attempt to parallelize the script, breaking the input into chunks of records with different
    /// worker threads processing different chunks.
    // TODO: support multiple readers producing chunks from multiple files, if possible.
    ShardPerRecord,
    /// Attempt to parallelize the script, where multiple worker threads each process a file at a
    /// time.
    ShardPerFile,
}

impl ExecutionStrategy {
    pub fn num_workers(&self) -> usize {
        use ExecutionStrategy::*;
        match self {
            ShardPerFile | ShardPerRecord => num_cpus::get(),
            Serial => 1,
        }
    }
    pub fn stage(&self) -> Stage<()> {
        use ExecutionStrategy::*;
        match self {
            ShardPerRecord | ShardPerFile => Stage::Par {
                begin: None,
                main_loop: None,
                end: None,
            },
            Serial => Stage::Main(()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Stage<T> {
    Main(T),
    Par {
        begin: Option<T>,
        main_loop: Option<T>,
        end: Option<T>,
    },
}

impl<T: Default> Default for Stage<T> {
    fn default() -> Stage<T> {
        Stage::Main(Default::default())
    }
}

impl<T> Stage<T> {
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        use smallvec::{smallvec, SmallVec};
        let res: SmallVec<[&T; 3]> = match self {
            Stage::Main(t) => smallvec![t],
            Stage::Par {
                begin,
                main_loop,
                end,
            } => {
                let mut x = SmallVec::new();
                x.extend(begin.iter().chain(main_loop).chain(end));
                x
            }
        };
        res.into_iter()
    }

    pub fn map_ref<F, R>(&self, mut f: F) -> Stage<R>
    where
        F: FnMut(&T) -> R,
    {
        match self {
            Stage::Main(t) => Stage::Main(f(t)),
            Stage::Par {
                begin,
                main_loop,
                end,
            } => Stage::Par {
                begin: begin.as_ref().map(&mut f),
                main_loop: main_loop.as_ref().map(&mut f),
                end: end.as_ref().map(&mut f),
            },
        }
    }

    pub fn map<F, R>(self, mut f: F) -> Stage<R>
    where
        F: FnMut(T) -> R,
    {
        match self {
            Stage::Main(t) => Stage::Main(f(t)),
            Stage::Par {
                begin,
                main_loop,
                end,
            } => Stage::Par {
                begin: begin.map(&mut f),
                main_loop: main_loop.map(&mut f),
                end: end.map(&mut f),
            },
        }
    }
}

#[derive(Clone, Debug)]
pub enum Either<L, R> {
    Left(L),
    Right(R),
}

// borrowed from weld project.
macro_rules! c_str {
    ($s:expr) => {
        concat!($s, "\0").as_ptr() as *const libc::c_char
    };
}

macro_rules! for_either {
    ($e:expr, |$id:pat| $body:expr) => {{
        match $e {
            crate::common::Either::Left($id) => $body,
            crate::common::Either::Right($id) => $body,
        }
    }};
}

impl<L, R, T> Iterator for Either<L, R>
where
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
{
    type Item = T;
    fn next(&mut self) -> Option<T> {
        for_either!(self, |x| x.next())
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        for_either!(self, |x| x.size_hint())
    }
    fn count(self) -> usize {
        for_either!(self, |x| x.count())
    }
}

pub(crate) struct IntoIter<L, R>(pub Either<L, R>);

impl<L, R, T> IntoIterator for IntoIter<L, R>
where
    L: IntoIterator<Item = T>,
    R: IntoIterator<Item = T>,
{
    type Item = T;
    type IntoIter = Either<L::IntoIter, R::IntoIter>;
    fn into_iter(self) -> Self::IntoIter {
        match self.0 {
            Either::Left(l) => Either::Left(l.into_iter()),
            Either::Right(r) => Either::Right(r.into_iter()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompileError(pub String);

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

macro_rules! err_raw {
    ($head:expr) => {
        $crate::common::CompileError(
                format!(concat!("[", file!(), ":", line!(), ":", column!(), "] ", $head))
        )
    };
    ($head:expr, $($t:expr),+) => {
        $crate::common::CompileError(
                format!(concat!("[", file!(), ":", line!(), ":", column!(), "] ", $head), $($t),*)
        )
    };
}

macro_rules! err {
    ($($x:tt)*) => {
        Err(err_raw!($($x)*))
    };
}

// We use this for when we want to print an error message, but don't want to panic if we cannot
// write to standard error.
macro_rules! eprintln_ignore {
    ($($t:tt)*) => {{
        use std::io::Write;
        let mut err = std::io::stderr();
        let _ = writeln!(&mut err, $($t)*);
        let _ = err.flush();
    }};
}

macro_rules! static_map {
    ($name:ident<$kty:ty, $vty:ty>, $([$k:expr, $v:expr]),*) => {
        lazy_static::lazy_static! {
            pub(crate) static ref $name: hashbrown::HashMap<$kty,$vty> = {
                let mut m = hashbrown::HashMap::new();
                $(
                    m.insert($k, $v);
                )*
                m
            };
        }
    }
}

pub(crate) struct WorkList<T> {
    set: HashSet<T>,
    // TODO: switch back to Vec? That would probably help string_constants converge faster.
    mem: VecDeque<T>,
}

impl<T: Hash + Eq> Default for WorkList<T> {
    fn default() -> WorkList<T> {
        WorkList {
            set: Default::default(),
            mem: Default::default(),
        }
    }
}

impl<T: Clone + Hash + Eq> WorkList<T> {
    pub(crate) fn insert(&mut self, t: T) {
        if self.set.insert(t.clone()) {
            self.mem.push_back(t)
        }
    }
    pub(crate) fn extend(&mut self, ts: impl Iterator<Item = T>) {
        for t in ts {
            self.insert(t);
        }
    }
    pub(crate) fn pop(&mut self) -> Option<T> {
        let next = self.mem.pop_front()?;
        let _was_there = self.set.remove(&next);
        debug_assert!(_was_there);
        Some(next)
    }
}

/// Notification is a simple object used to synchronize multiple threads around a single event
/// occuring.
///
/// Notifications are "one-shot": they only transition from "not notified" to "notified" once.
/// Based on the absl object of the same name.
pub struct Notification {
    notified: AtomicBool,
    mu: Mutex<()>,
    cv: Condvar,
}

impl Default for Notification {
    fn default() -> Notification {
        Notification {
            notified: AtomicBool::new(false),
            mu: Mutex::new(()),
            cv: Condvar::new(),
        }
    }
}

impl Notification {
    pub fn has_been_notified(&self) -> bool {
        self.notified.load(Ordering::Acquire)
    }
    pub fn notify(&self) {
        if self.has_been_notified() {
            return;
        }
        let _guard = self.mu.lock().unwrap();
        self.notified.store(true, Ordering::Release);
        self.cv.notify_all();
    }
    pub fn wait(&self) {
        loop {
            // Fast path: check if the notification has already happened.
            if self.has_been_notified() {
                return;
            }
            // Slow path: grab the lock, and check notification state again before waiting on the
            // condition variable.
            let mut _guard = self.mu.lock().unwrap();
            if self.has_been_notified() {
                return;
            }
            _guard = self.cv.wait(_guard).unwrap();
        }
    }
}

/// A CancelSignal is a thread-safe handle for propagating error information. It is used for
/// cancelling processes running in parallel. The intent is that the cancellation status can be
/// queried cheaply by multiple threads. CancelSignal has "first writer wins" semantics.
#[derive(Clone)]
pub struct CancelSignal {
    state: Arc<AtomicI64>,
}

const SENTINEL: i64 = i64::MAX;

impl Default for CancelSignal {
    fn default() -> Self {
        Self {
            state: Arc::new(AtomicI64::new(SENTINEL)),
        }
    }
}

impl CancelSignal {
    pub fn get_code(&self) -> Option<i32> {
        let raw = self.state.load(Ordering::Acquire);
        if raw == SENTINEL {
            None
        } else {
            Some(raw as i32)
        }
    }
    pub fn cancelled(&self) -> bool {
        self.get_code().is_some()
    }
    pub fn cancel(&self, code: i32) {
        // We use relaxed here because we are not using the output of this variable at all.
        //
        // We could probably get away with weaker semantics than AcqRel: no one relies on the
        // output of the `get_code` value except when setting the error code. However, the
        // semantics of relaxed RMW operations are sufficiently murky that we should only go that
        // route if this code ends up being used in more performance-sensitive settings than it is
        // today.
        let _ =
            self.state
                .compare_exchange(SENTINEL, code as i64, Ordering::AcqRel, Ordering::Relaxed);
    }
}

pub struct Cleanup<Arg> {
    #[allow(clippy::type_complexity)]
    data: Option<Box<dyn for<'a> FnOnce(&'a mut Arg)>>,
}

impl<Arg> Cleanup<Arg> {
    pub fn null() -> Self {
        Cleanup { data: None }
    }
    pub fn cancel(&mut self) {
        self.data = None;
    }
    pub fn invoke(&mut self, arg: &mut Arg) {
        if let Some(f) = self.data.take() {
            f(arg)
        }
    }
    pub fn new(f: impl for<'a> FnOnce(&'a mut Arg) + 'static) -> Self {
        Cleanup {
            data: Some(Box::new(f) as _),
        }
    }
}

impl<Arg> Drop for Cleanup<Arg> {
    fn drop(&mut self) {
        assert!(
            self.data.is_none(),
            "destroying un-executed cleanup handler"
        );
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(i64)]
pub enum FileSpec {
    Trunc = 0,
    Append = 1,
    Cmd = 2,
}

#[derive(Debug)]
pub struct InvalidFileSpec;
impl fmt::Display for InvalidFileSpec {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt("invalid file spec", fmt)
    }
}

impl std::convert::TryFrom<i64> for FileSpec {
    type Error = InvalidFileSpec;
    fn try_from(i: i64) -> std::result::Result<FileSpec, InvalidFileSpec> {
        // uglier than a match, but stays in sync with the enum more easily.
        if i == FileSpec::Trunc as i64 {
            Ok(FileSpec::Trunc)
        } else if i == FileSpec::Append as i64 {
            Ok(FileSpec::Append)
        } else if i == FileSpec::Cmd as i64 {
            Ok(FileSpec::Cmd)
        } else {
            Err(InvalidFileSpec)
        }
    }
}

impl Default for FileSpec {
    fn default() -> FileSpec {
        FileSpec::Append
    }
}

pub(crate) fn traverse<T>(o: Option<Result<T>>) -> Result<Option<T>> {
    match o {
        Some(e) => Ok(Some(e?)),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_elems<T: Clone + Hash + Eq>(wl: &mut WorkList<T>) -> HashSet<T> {
        let mut res = HashSet::default();
        while let Some(e) = wl.pop() {
            assert!(res.insert(e));
        }
        res
    }

    #[test]
    fn worklist_elems() {
        let mut wl = WorkList::<i32>::default();
        for i in 0..10 {
            wl.insert(i);
        }
        assert_eq!(get_elems(&mut wl), (0i32..10).collect());
    }

    #[test]
    fn worklist_idempotent() {
        let mut wl = WorkList::<i32>::default();
        wl.extend(0..10);
        for i in 0..10 {
            wl.insert(i);
        }
        assert_eq!(get_elems(&mut wl), (0i32..10).collect());
    }
}
