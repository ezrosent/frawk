//! This file contains common type definitions and utilities used in other parts of the project.
use hashbrown::HashSet;
use std::hash::Hash;
pub(crate) type NumTy = u32;
pub(crate) type NodeIx = petgraph::graph::NodeIndex<NumTy>;
pub(crate) type Graph<V, E> = petgraph::Graph<V, E, petgraph::Directed, NumTy>;
pub(crate) type Result<T> = std::result::Result<T, CompileError>;
pub(crate) enum Either<L, R> {
    Left(L),
    Right(R),
}

#[derive(Debug)]
pub(crate) struct CompileError(pub String);

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

macro_rules! err {
    ($($t:tt),*) => { Err($crate::common::CompileError(format!($($t),*))) }
}

pub(crate) struct WorkList<T> {
    set: HashSet<T>,
    mem: Vec<T>,
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
            self.mem.push(t)
        }
    }
    pub(crate) fn extend(&mut self, ts: impl Iterator<Item = T>) {
        for t in ts {
            self.insert(t);
        }
    }
    pub(crate) fn pop(&mut self) -> Option<T> {
        let next = self.mem.pop()?;
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
