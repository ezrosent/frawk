/// Support for basic "projection pushdown" for LineReader types.
///
/// N.B this is not "pushdown" in the sense of  "pushdown control flow analysis", just in the sense
/// of pushing down projections of relevant fields from the input storage.
use std::fmt;

use crate::bytecode::Reg;
use crate::common::{Graph, NodeIx, WorkList};
use crate::runtime::Int;

use hashbrown::HashMap;
use petgraph::Direction;
use smallvec::SmallVec;

#[derive(Clone, PartialEq, Eq)]
pub struct FieldSet(u64);

impl Default for FieldSet {
    fn default() -> FieldSet {
        FieldSet::all()
    }
}

const MAX_INDEX: u32 = 63;

impl FieldSet {
    pub fn singleton(index: u32) -> FieldSet {
        if index > MAX_INDEX {
            Self::all()
        } else {
            FieldSet(1 << index)
        }
    }
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }
    pub fn all() -> FieldSet {
        FieldSet(!0)
    }
    pub fn empty() -> FieldSet {
        FieldSet(0)
    }
    pub fn union(&mut self, other: &FieldSet) {
        self.0 = self.0 | other.0;
    }
    pub fn get(&self, index: u32) -> bool {
        (1u64.wrapping_shl(index) & self.0) != 0
    }
    pub fn set(&mut self, index: u32) {
        if index < MAX_INDEX {
            self.0 |= 1u64 << index;
        } else {
            *self = Self::all();
        }
    }
}

impl fmt::Debug for FieldSet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self == &FieldSet::all() {
            return write!(f, "<ALL>");
        }
        let v: Vec<_> = (0..=MAX_INDEX).filter(|i| self.get(*i)).collect();
        write!(f, "{:?}", v)
    }
}

// Good to think about this in two "stages":
// - in the initialization phase we can't guarantee the order in which we will add nodes to the
//   graph, so we'll start them off as empty sets and then fill nodes in as we encounter them.
// - Before transitioning to the solving phase, we find all the empty sets and replace them with
//   full sets.
// - Then we enter the solving phase, where we just iteratively union nodes with their
//   dependencies (and themselves)
// This trick only works because no node will actually get an empty set to start with. The only way
// to get an empty set is to have no GetCol instructions at all.

#[derive(Default)]
pub struct UsedFieldAnalysis {
    assign_graph: Graph<FieldSet, ()>,
    regs: HashMap<Reg<Int>, NodeIx>,
    relevant: SmallVec<[NodeIx; 2]>,
}

impl UsedFieldAnalysis {
    fn get_node(&mut self, reg: Reg<Int>) -> NodeIx {
        use hashbrown::hash_map::Entry;
        match self.regs.entry(reg) {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(v) => {
                let n = self.assign_graph.add_node(FieldSet::empty());
                v.insert(n);
                n
            }
        }
    }
    pub fn add_field(&mut self, reg: Reg<Int>, index: u32) {
        use hashbrown::hash_map::Entry;
        match self.regs.entry(reg) {
            Entry::Occupied(o) => self
                .assign_graph
                .node_weight_mut(*o.get())
                .unwrap()
                .set(index),
            Entry::Vacant(v) => {
                let n = self.assign_graph.add_node(FieldSet::singleton(index));
                v.insert(n);
            }
        }
    }
    pub fn add_dep(&mut self, from_reg: Reg<Int>, to_reg: Reg<Int>) {
        let from_node = self.get_node(from_reg);
        let to_node = self.get_node(to_reg);
        self.assign_graph.add_edge(from_node, to_node, ());
    }
    pub fn add_col(&mut self, col_reg: Reg<Int>) {
        let col_node = self.get_node(col_reg);
        self.relevant.push(col_node);
    }
    pub fn solve(mut self) -> FieldSet {
        self.solve_internal();
        let mut res = FieldSet::empty();
        for i in self.relevant.iter().cloned() {
            res.union(self.assign_graph.node_weight(i).unwrap());
        }
        res
    }
    fn solve_internal(&mut self) {
        let mut wl = WorkList::default();
        wl.extend((0..self.assign_graph.node_count()).map(|x| NodeIx::new(x)));

        let mut last = false;
        loop {
            while let Some(n) = wl.pop() {
                let start = self.assign_graph.node_weight(n).unwrap().clone();
                let mut new = start.clone();
                for n in self.assign_graph.neighbors_directed(n, Direction::Incoming) {
                    new.union(self.assign_graph.node_weight(n).unwrap());
                }
                if start == new {
                    continue;
                }
                *self.assign_graph.node_weight_mut(n).unwrap() = new;
                for n in self.assign_graph.neighbors_directed(n, Direction::Outgoing) {
                    wl.insert(n);
                }
            }
            if last {
                break;
            }
            last = true;
            for n in self.assign_graph.node_indices() {
                let redo = {
                    let w = self.assign_graph.node_weight_mut(n).unwrap();
                    let redo = w.is_empty();
                    if redo {
                        *w = FieldSet::all();
                    }
                    redo
                };
                if redo {
                    for n in self.assign_graph.neighbors_directed(n, Direction::Outgoing) {
                        wl.insert(n);
                    }
                }
            }
        }
    }
}
