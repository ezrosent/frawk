//! A simple data-flow analysis for finding string constants.
//!
//! This analysis is currently used to perform constant folding on regular expressions, and
//! tracking accesses to the `FI` builtin variable for help in increasing the precision in the
//! used-field analysis when passing the -H flag.
use crate::builtins::Variable;
use crate::bytecode::Instr;
use crate::common::NumTy;
use crate::compile::HighLevel;
use crate::dataflow::{self, JoinSemiLattice, Key};
use hashbrown::{HashMap, HashSet};

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

pub(crate) struct StringConstantAnalysis<'a> {
    intern_l: HashMap<&'a [u8], usize>,
    intern_r: HashMap<usize, &'a [u8]>,
    dfa: dataflow::Analysis<ApproximateSet>,
    cfg: Config,
}

pub(crate) struct Config {
    // Collect possible regexes, with the purpose of constant-folding them
    pub query_regex: bool,
    // Collect the strings used to query FI, for the purpose of doing pushdown on named columns
    pub fi_refs: bool,
}

impl<'a> StringConstantAnalysis<'a> {
    pub(crate) fn cfg(&self) -> &Config {
        &self.cfg
    }
    pub(crate) fn from_config(cfg: Config) -> Self {
        let mut res = StringConstantAnalysis {
            intern_l: Default::default(),
            intern_r: Default::default(),
            dfa: Default::default(),
            cfg,
        };
        res.dfa.add_src(Key::Rng, ApproximateSet::unknown());
        if res.cfg.fi_refs {
            res.dfa.add_query(Key::VarKey(Variable::FI));
            res.dfa.add_query(Key::VarVal(Variable::FI));
        }
        res
    }
    pub(crate) fn visit_ll(&mut self, inst: &Instr<'a>) {
        use Instr::*;
        if self.cfg.query_regex {
            // TODO: Do the same for Sub, GSub, Split*
            if let Match(_, _, pat) | IsMatch(_, _, pat) = inst {
                self.dfa.add_query(pat)
            }
        }
        match inst {
            StoreConstStr(dst, s) => {
                let id = self.get_id(s.literal_bytes());
                self.dfa.add_src(dst, ApproximateSet::singleton(id));
            }
            // Note that variables can be set "out of band", so by default we aren't treating them
            // as standard registers.
            StoreVarStrIntMap(Variable::FI, _)
            | LoadVarStrIntMap(_, Variable::FI)
            | Lookup { .. }
            | Store { .. }
            | IterBegin { .. }
            | IterGetNext { .. }
            | Mov(..) => dataflow::boilerplate::visit_ll(inst, |dst, src| {
                if let Some(src) = src {
                    self.dfa.add_dep(dst, src, ())
                } else {
                    // insert a sentinel for the inserts that occur due to a Lookup
                    let id = 0;
                    self.dfa.add_src(dst, ApproximateSet::singleton(id));
                }
            }),
            _ => dataflow::boilerplate::visit_ll(inst, |dst, _| {
                self.dfa.add_src(dst, ApproximateSet::unknown());
            }),
        }
    }
    pub(crate) fn visit_hl(&mut self, cur_fn_id: NumTy, inst: &HighLevel) {
        dataflow::boilerplate::visit_hl(inst, cur_fn_id, |dst, src| {
            self.dfa.add_dep(dst, src.unwrap(), ())
        })
    }
    fn get_id(&mut self, s: &'a [u8]) -> usize {
        // 0 is a sentinel value
        let next_id = self.intern_l.len() + 1;
        let l = &mut self.intern_l;
        let r = &mut self.intern_r;
        *l.entry(s).or_insert_with(|| {
            r.insert(next_id, s);
            next_id
        })
    }

    fn possible_strings_inner(&mut self, k: impl Into<Key>, res: &mut Vec<&'a [u8]>) -> bool /* known */
    {
        let intern_r = &self.intern_r;
        let q = self.dfa.query(k);
        if q.0.is_none() {
            return false;
        }
        res.extend(q.iter().map(|x| intern_r.get(x).cloned().unwrap_or(&[])));
        true
    }

    pub fn possible_strings(&mut self, k: impl Into<Key>, res: &mut Vec<&'a [u8]>) {
        self.possible_strings_inner(k, res);
    }

    pub fn fi_info(&mut self, cols: &mut Vec<&'a [u8]>) -> bool /* known */ {
        match &self.dfa.query(Key::VarVal(Variable::FI)).0 {
            Some(ids) => {
                if ids.is_empty() || (ids.len() == 1 && ids.contains(&0 /*sentinel*/)) {
                    self.possible_strings_inner(Key::VarKey(Variable::FI), cols)
                } else {
                    false
                }
            }
            // The analysis detected nontrivial writes into FI during execution. That's legal, but
            // we don't try and model it for the purposes of this analysis.
            None => false,
        }
    }
}
