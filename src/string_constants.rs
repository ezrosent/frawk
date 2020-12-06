use crate::bytecode::{Accum, Instr};
use crate::common::{Graph, NodeIx, NumTy, WorkList};
use crate::compile::{HighLevel, Ty};
use hashbrown::{HashMap, HashSet};
use petgraph::{visit::Dfs, Direction};

use std::mem;

// TODO: nodes are going to go in "reverse order" here, because we want to use Dfs to figure out
// which nodes need to be visited when doing the analysis.

struct ApproximateSet(Option<HashSet<usize>>);

impl ApproximateSet {
    fn iter(&self) -> impl Iterator<Item = &usize> + '_ {
        self.0.as_ref().into_iter().flat_map(|x| x.iter())
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
impl Default for ApproximateSet {
    fn default() -> Self {
        ApproximateSet(Some(Default::default()))
    }
}

// TODO: implement hash set operations
// TODO: implement "extract" operation
// TODO: implement public "extract" function
// TODO: implement visitor.

pub struct StringConstantAnalysis<'a> {
    intern_l: HashMap<&'a [u8], usize>,
    intern_r: HashMap<usize, &'a [u8]>,
    sentinel: NodeIx,
    flows: Graph<ApproximateSet, ()>,
    reg_map: HashMap<NumTy, NodeIx>,
    wl: WorkList<NodeIx>,
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
        };
        res.sentinel = res.flows.add_node(Default::default());
        res
    }
}

impl<'a> StringConstantAnalysis<'a> {
    pub(crate) fn visit_ll(&mut self, inst: &Instr<'a>, query_regex: bool) {
        use Instr::*;
        match inst {
            StoreConstStr(dst, s) => {
                let id = self.get_id(s.literal_bytes());
                let node = self.get_node(dst.reflect().0);
                self.flows.node_weight_mut(node).unwrap().insert(id)
            }
            Mov(Ty::Str, dst, src) => {
                let dst_node = self.get_node(*dst);
                let src_node = self.get_node(*src);
                // NB: we add dependencies "in reverse"
                self.flows.add_edge(dst_node, src_node, ());
            }
            // TODO: Do the same for Sub, GSub, Split*
            Match(_, _, pat) | IsMatch(_, _, pat) if query_regex => {
                self.add_query(pat.reflect().0);
            }
            _ => {}
        }
    }
    pub(crate) fn visit_hl(&mut self, inst: &HighLevel) {
        use HighLevel::*;
        if let Phi(dst, Ty::Str, preds) = inst {
            let dst_node = self.get_node(*dst);
            for (_, pred_reg) in preds.iter() {
                let pred_node = self.get_node(*pred_reg);
                self.flows.add_edge(dst_node, pred_node, ());
            }
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
    fn get_node(&mut self, reg: NumTy) -> NodeIx {
        let flows = &mut self.flows;
        *self
            .reg_map
            .entry(reg)
            .or_insert_with(|| flows.add_node(Default::default()))
    }

    pub fn possible_strings(&mut self, reg: NumTy, res: &mut Vec<&'a [u8]>) {
        self.solve();
        res.extend(
            self.flows
                .node_weight(self.reg_map[&reg])
                .unwrap()
                .iter()
                .map(|x| self.intern_r[x].clone()),
        )
    }

    pub fn add_query(&mut self, reg: NumTy) {
        let ix = self.get_node(reg);
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
