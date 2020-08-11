//! This file contains common type definitions and utilities used in other parts of the project.
use hashbrown::HashSet;
use std::collections::VecDeque;
use std::hash::Hash;
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
    ShardPerRecord,
    /// Attempt to parallelize the script, where multiple worker threads each process a file at a
    /// time.
    ShardPerFile,
}

impl ExecutionStrategy {
    pub fn num_workers(&self) -> usize {
        use ExecutionStrategy::*;
        match self {
            ShardPerRecord | ShardPerFile => num_cpus::get(),
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
                x.extend(begin.into_iter().chain(main_loop).chain(end));
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
        concat!($s, "\0").as_ptr() as *const crate::libc::c_char
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

macro_rules! err {
    ($head:expr) => {
        Err($crate::common::CompileError(
                format!(concat!("[", file!(), ":", line!(), ":", column!(), "] ", $head))
        ))
    };
    ($head:expr, $($t:expr),+) => {
        Err($crate::common::CompileError(
                format!(concat!("[", file!(), ":", line!(), ":", column!(), "] ", $head), $($t),*)
        ))
    };
}

// We use this for when we want to print an error message, but don't want to panic if we cannot
// write to standard error.
macro_rules! eprintln_ignore {
    ($($t:tt)*) => {{
        use std::io::Write;
        let _ = writeln!(&mut std::io::stderr(), $($t)*);
        ()
    }};
}

macro_rules! static_map {
    ($name:ident<$kty:ty, $vty:ty>, $([$k:expr, $v:expr]),*) => {
        $crate::lazy_static::lazy_static! {
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
    // TODO: switch back to Vec?
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
