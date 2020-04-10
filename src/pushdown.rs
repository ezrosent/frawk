//! Support for basic "projection pushdown" for LineReader types.
//!
//! **NB** this is not "pushdown" in the sense of  "pushdown control flow analysis", just in the
//! sense of pushing down projections of relevant fields from the input storage.
//!
//! Short scripts can spend a surprising amount of time just slicing and escaping strings for each
//! input record; this module provides the core components of a static analysis that constructs a
//! conservative representation of all fields referenced in a given program. By "conservative" we
//! mean taht it will sometimes return more fields than a program actually uses, but it will never
//! produce false negatives.
//!
//! # Overview of the analysis
//!
//! The basic idea is to keep track of every GetCol instruction and accumulate all of the numbers
//! passed to that operation. In the IR, GetCol takes a register, so we need to be able to guess
//! the numbers to which a given register corresponds. To do that, we track all StoreConstInt,
//! MovInt and Phi instructions of the appropriate type. We place all of these registers in a graph
//! where an edge from node X to node Y indicates that Y can take on at least any of the values
//! that X can. Schematically, with constants in `[square brackets]`
//!
//!     StoreConstInt(dst, [c]) : [c] =>  dst
//!     MovInt(dst, src) : src => dst
//!     Phi(dst, Int, ..., pred_i, ...]) : ... pred_i-1 => dst, pred_i => dst ...
//!
//! This graph corresponds to a set of (recursive) equation of the form
//!
//!     Fields([c]) = {c}
//!     Fields(node) = union_{n s.t. (n, node) is an edge} Fields(n)
//!
//! By convention, empty unions produce the empty set {}. Starting off all non-constant nodes at
//! the empty set and then iterating this rule will converge to the least fixed point of these
//! equations as the set of possible constants is finite, sets ordered by inclusion form a lattice,
//! and union is monotone on that lattice. (Apologies if I misstated something there).
//!
//! Then, all we need to do is to union all of the field sets corresponding to registers passed to
//! GetCol! Well, not quite. The rules for generating equations don't cover more complex operations
//! like math, or functions. Suppose we had the following sequence:
//!
//!     GetCol(1, [2])
//!     StrToInt(0, 1)
//!     GetCol(dst, 0)
//!
//! Which corresponds roughly to the AWK snippet `$$2`, or "the field corresponding to the value of
//! the second column." We cannot predict this value ahead of time, but our rules do not generate
//! any constraints for the StrToInt instruction, or for any number of other instructions that can
//! assign to integer registers. If we apply the algorithm as written, register 0 will have no
//! incoming edges, and so will contribute no fields to the dst register, thereby producing false
//! negatives.
//!
//! The most direct solution here would be to contribute "full sets" (sets that contain all
//! possible fields --- [FieldSet::all] below) to any register stored to by an instruction other
//! than StoreConstInt, MovInt, or Phi. This would work, but it would require a lot more code, and
//! we would have to continually update the analysis code as we added or removed instructions from
//! the bytecode.
//!
//! Instead, we do this implicitly by running the algorithm twice: the first time as is, the second
//! time by replacing any empty nodes with no incoming edges with full sets. The reasoning here is
//! that we will only produce nodes without incoming edges if we have a constant, or if they were
//! the result of some "black box" instruction like StrToInt that we do not want to analyze. In the
//! former case, there will always be at least one field present in the given node; that means any
//! remaining nodes should be treated as potentially representing any arbitrary field number. Once
//! we "flip" these empty nodes to full sets, we re-run the algorithm and read the result out of
//! the GetCol registers.
use std::fmt;

use crate::bytecode::Reg;
use crate::common::{Graph, NodeIx, WorkList};
use crate::runtime::Int;

use hashbrown::HashMap;
use petgraph::Direction;
use smallvec::SmallVec;

/// Most AWK scripts do not use more than 63 fields, so we represent our sets of used fields
/// "lossy bitsets" that can precisely represent subsets of [0, 63] but otherwise just say "yes" to
/// all queries. This is a lowsy choice for a general bitset type, but it's a sound and efficient
/// choice for this analysis, where we're free to overapproximate the fields that are used by a
/// particular program.
#[derive(Clone, PartialEq, Eq)]
pub struct FieldSet(u64);

impl Default for FieldSet {
    fn default() -> FieldSet {
        FieldSet::all()
    }
}

// A note on the type we use for indexes: Why usize when shl, etc. take u32? The main clients of
// this library will be field-splitting routines, which will often be passing in counters or vector
// lengths. We may as well handle the (however unlikely to be exercised) overflow logic here rather
// than up the stack, in more complicated code, in multiple locations.
//
// TODO: this could probably be 64, not that it matters a great deal.
const MAX_INDEX: usize = 63;

impl FieldSet {
    pub fn singleton(index: usize) -> FieldSet {
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
    pub fn get(&self, index: usize) -> bool {
        (index > MAX_INDEX) || (1u64 << (index as u32)) & self.0 != 0
    }
    pub fn set(&mut self, index: usize) {
        if index <= MAX_INDEX {
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

#[derive(Default)]
pub struct UsedFieldAnalysis {
    assign_graph: Graph<FieldSet, ()>,
    regs: HashMap<Reg<Int>, NodeIx>,
    relevant: SmallVec<[NodeIx; 2]>,
}

impl UsedFieldAnalysis {
    /// Get the node corresponding to a given regiter, or allocate a fresh one with an empty set.
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

    /// Add a node with a constant: these correspond to the StoreConstInt nodes mentioned above.
    pub fn add_field(&mut self, reg: Reg<Int>, index: usize) {
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

    /// Mark a given register as a "column" node: one that will be part of the used field set
    /// returned from [UsedFieldAnalysis::solve].
    pub fn add_col(&mut self, col_reg: Reg<Int>) {
        let col_node = self.get_node(col_reg);
        self.relevant.push(col_node);
    }

    /// Return the set of all fields mentioned by column nodes.
    pub fn solve(mut self) -> FieldSet {
        self.solve_internal();
        let mut res = FieldSet::empty();
        for i in self.relevant.iter().cloned() {
            res.union(self.assign_graph.node_weight(i).unwrap());
        }
        res
    }
    fn solve_internal(&mut self) {
        // The core solving portion of the analysis. Start with a full worklist.
        let mut wl = WorkList::default();
        wl.extend((0..self.assign_graph.node_count()).map(|x| NodeIx::new(x)));

        // We will do the core iteration twice: once with empties once with empty sets flipped to
        // full.
        //
        // TODO We could probably just do a single pass and detect empty nodes initially.
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

            // Flip "black box" nodes to full. The module comment talks about these nodes as though
            // they are just the "root nodes" with no incoming edges. Here we take any empty nodes.
            // The two approaches would appear to be equivalent.
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
