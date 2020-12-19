//! A simple data-flow analysis for finding string constants.
//!
//! This analysis is currently used to perform constant folding on regular expressions, but it may
//! be used for more things in the future. It largely follows the structure of the analysis in
//! `pushdown.rs`.
use crate::builtins::Variable;
use crate::bytecode::{Accum, Instr, Reg};
use crate::common::{Graph, NodeIx, NumTy, WorkList};
use crate::compile::{HighLevel, Ty};
use crate::dataflow::{self, JoinSemiLattice, Key};
use hashbrown::{HashMap, HashSet};
use petgraph::{visit::Dfs, Direction};

use std::mem;

// TODO: replace with a bitset? The keys will be very dense, though we won't know them ahead of
// time.
struct ApproximateSet(Option<HashSet<usize>>);

impl ApproximateSet {
    fn iter(&self) -> impl Iterator<Item = &usize> + '_ {
        self.0.as_ref().into_iter().flat_map(|x| x.iter())
    }
    fn singleton(u: usize) -> ApproximateSet {
        let mut res = HashSet::with_capacity(1);
        res.insert(u);
        ApproximateSet(Some(res))
    }
    fn unknown() -> Self {
        ApproximateSet(None)
    }
    // This turns a set that we don't know anything about _at the moment_ into a set that we will
    // _never_ know anything about. Sets that contain some information are not changed
    fn sour(&mut self) {
        if matches!(self.0.as_ref().map(|s| s.len() == 0), Some(true)) {
            self.0 = None;
        }
    }
    fn insert(&mut self, item: usize) {
        if let Some(dst) = &mut self.0 {
            dst.insert(item);
        }
    }
    fn merge(&mut self, other: &Self) -> bool /*changed*/ {
        match (&mut self.0, &other.0) {
            (Some(dst), Some(src)) => {
                let mut changed = false;
                for s in src.iter().cloned() {
                    changed |= dst.insert(s);
                }
                changed
            }
            (Some(_), None) => {
                *self = Self::unknown();
                true
            }
            (None, _) => false,
        }
    }
}

impl JoinSemiLattice for ApproximateSet {
    type Func = ();
    fn bottom() -> ApproximateSet {
        ApproximateSet(Some(Default::default()))
    }
    fn invoke(&mut self, other: &ApproximateSet, (): &()) -> bool {
        self.merge(other)
    }
}

impl Default for ApproximateSet {
    fn default() -> Self {
        ApproximateSet(Some(Default::default()))
    }
}

pub(crate) struct StringConstantAnalysis<'a> {
    intern_l: HashMap<&'a [u8], usize>,
    intern_r: HashMap<usize, &'a [u8]>,
    dfa: dataflow::Analysis<ApproximateSet>,
    sentinel: NodeIx,
    flows: Graph<ApproximateSet, ()>,
    reg_map: HashMap<Key, NodeIx>,
    wl: WorkList<NodeIx>,
}

pub struct Config {
    // Collect possible regexes, with the purpose of constant-folding them
    pub query_regex: bool,
    // Collect the strings used to query FI, for the purpose of doing pushdown on named columns
    pub fi_refs: bool,
}

impl<'a> Default for StringConstantAnalysis<'a> {
    fn default() -> Self {
        let mut res = StringConstantAnalysis {
            sentinel: Default::default(),
            flows: Default::default(),
            reg_map: Default::default(),
            wl: Default::default(),
            intern_l: Default::default(),
            intern_r: Default::default(),
            dfa: Default::default(),
        };
        res.sentinel = res.flows.add_node(Default::default());
        res
    }
}
// TODO: need to model all dataflow, see
//   'function unused() { print x; } BEGIN { x = "hi"; x=ARGV[0]; print("h" ~ x); } '
// unused forces x to be global.
// This folds to "h" ~ "hi" currently
//
// Similar bug occurs with used fields:
//  head ../frawk-scratch/scratch/Data8277.csv | target/debug/frawk -icsv 'function unused() { print x; } { x=2; x=NF; print $x; }'
//
// prints nothing; even though the last field is populated.

impl<'a> StringConstantAnalysis<'a> {
    pub(crate) fn visit_ll(&mut self, inst: &Instr<'a>, Config { query_regex, .. }: &Config) {
        use Instr::*;
        match inst {
            StoreConstStr(dst, s) => {
                let id = self.get_id(s.literal_bytes());
                self.dfa.add_src(dst, ApproximateSet::singleton(id));
                let node = self.get_node(dst);
                self.flows.node_weight_mut(node).unwrap().insert(id)
            }
            Mov(ty, dst, src) => {
                if ty.is_array() {
                    let dst_node_k = self.get_node(Key::MapKey(*dst, *ty));
                    let dst_node_v = self.get_node(Key::MapVal(*dst, *ty));
                    let src_node_k = self.get_node(Key::MapKey(*src, *ty));
                    let src_node_v = self.get_node(Key::MapVal(*src, *ty));
                    self.flows.add_edge(dst_node_k, src_node_k, ());
                    self.flows.add_edge(dst_node_v, src_node_v, ());
                    self.flows.add_edge(src_node_k, dst_node_k, ());
                    self.flows.add_edge(src_node_v, dst_node_v, ());
                } else {
                    let dst_node = self.get_node(Key::Reg(*dst, *ty));
                    let src_node = self.get_node(Key::Reg(*src, *ty));
                    // NB: we add dependencies "in reverse"
                    self.flows.add_edge(dst_node, src_node, ());
                }
            }
            LoadVarInt(dst, src) => {
                let dst_node = self.get_node(dst);
                // By convention, scalar variables are keys
                let src_node = self.get_node(Key::VarKey(*src));
                self.flows.add_edge(dst_node, src_node, ());
            }
            LoadVarStr(dst, src) => {
                let dst_node = self.get_node(dst);
                let src_node = self.get_node(Key::VarKey(*src));
                self.flows.add_edge(dst_node, src_node, ());
            }
            LoadVarIntMap(dst, src) => {
                let (reg, ty) = dst.reflect();
                let dst_node_k = self.get_node(Key::MapKey(reg, ty));
                let dst_node_v = self.get_node(Key::MapVal(reg, ty));
                let src_node_k = self.get_node(Key::VarKey(*src));
                let src_node_v = self.get_node(Key::VarVal(*src));
                self.flows.add_edge(dst_node_k, src_node_k, ());
                self.flows.add_edge(dst_node_v, src_node_v, ());
                self.flows.add_edge(src_node_k, dst_node_k, ());
                self.flows.add_edge(src_node_v, dst_node_v, ());
            }
            LoadVarStrMap(dst, src) => {
                let (reg, ty) = dst.reflect();
                let dst_node_k = self.get_node(Key::MapKey(reg, ty));
                let dst_node_v = self.get_node(Key::MapVal(reg, ty));
                let src_node_k = self.get_node(Key::VarKey(*src));
                let src_node_v = self.get_node(Key::VarVal(*src));
                self.flows.add_edge(dst_node_k, src_node_k, ());
                self.flows.add_edge(dst_node_v, src_node_v, ());
                self.flows.add_edge(src_node_k, dst_node_k, ());
                self.flows.add_edge(src_node_v, dst_node_v, ());
            }
            // For now, we aren't going to be doing any rules for StoreVar...
            // Why? Don't we want to poison them?

            // TODO: anything else that stores into a string has to get added here as well?

            // TODO: Do the same for Sub, GSub, Split*
            Match(_, _, pat) | IsMatch(_, _, pat) if *query_regex => {
                self.add_query(pat);
            }

            _ => {}
        }
    }
    pub(crate) fn visit_hl(&mut self, inst: &HighLevel) {
        use HighLevel::*;
        match inst {
            Phi(dst, ty, preds) => {
                let dst_node = self.get_node(Key::Reg(*dst, *ty));
                for (_, pred_reg) in preds.iter() {
                    let pred_node = self.get_node(Key::Reg(*pred_reg, *ty));
                    self.flows.add_edge(dst_node, pred_node, ());
                }
            }
            _ => {}
        }
    }
    fn get_id(&mut self, s: &'a [u8]) -> usize {
        let len = self.intern_l.len();
        let l = &mut self.intern_l;
        let r = &mut self.intern_r;
        *l.entry(s).or_insert_with(|| {
            r.insert(len, s);
            len
        })
    }
    fn get_node(&mut self, k: impl Into<Key>) -> NodeIx {
        let flows = &mut self.flows;
        *self
            .reg_map
            .entry(k.into())
            .or_insert_with(|| flows.add_node(Default::default()))
    }

    pub fn possible_strings(&mut self, k: impl Into<Key>, res: &mut Vec<&'a [u8]>) {
        self.solve();
        res.extend(
            self.flows
                .node_weight(self.reg_map[&k.into()])
                .unwrap()
                .iter()
                .map(|x| self.intern_r[x].clone()),
        )
    }

    pub fn add_query(&mut self, k: impl Into<Key>) {
        let ix = self.get_node(k.into());
        self.wl.insert(ix);
    }

    fn populate(&mut self) {
        let sentinel = self.sentinel;
        for node in self.wl.iter() {
            self.flows.add_edge(sentinel, node, ());
        }
        let mut dfs = Dfs::new(&self.flows, sentinel);
        while let Some(ix) = dfs.next(&self.flows) {
            self.wl.insert(ix);
            // check if there are any neighbors?
            if self.flows.neighbors(ix).next().is_none() {
                // If we're at a leaf, then "sour" it to ensure it poisons any downstream
                // definitions. If this leaf node has some information in it, souring does nothing.
                // I don't think it's the best metaphor to be honest, but it's an API that doesn't
                // extend beyond this module.
                self.flows.node_weight_mut(ix).unwrap().sour();
            }
        }
    }

    fn solve(&mut self) {
        self.populate();
        while let Some(n) = self.wl.pop() {
            let mut cur = mem::replace(self.flows.node_weight_mut(n).unwrap(), Default::default());
            let mut changed = false;
            for n in self.flows.neighbors_directed(n, Direction::Outgoing) {
                changed |= cur.merge(self.flows.node_weight(n).unwrap());
            }
            if changed {
                for n in self.flows.neighbors_directed(n, Direction::Incoming) {
                    self.wl.insert(n)
                }
            }
            mem::swap(&mut cur, self.flows.node_weight_mut(n).unwrap());
        }
    }
}
