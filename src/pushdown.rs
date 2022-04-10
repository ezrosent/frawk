//! Support for basic "projection pushdown" for LineReader types.
//!
//! **NB** this is not "pushdown" in the sense of  "pushdown control flow analysis", just in the
//! sense of pushing down projections of relevant fields from the input storage.
//!
//! Short scripts can spend a surprising amount of time just slicing and escaping strings for each
//! input record; this module provides the core components of a static analysis that constructs a
//! conservative representation of all fields referenced in a given program. By "conservative" we
//! mean that it will sometimes return more fields than a program actually uses, but it will never
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
//! ```text
//! StoreConstInt(dst, [c]) : [c] =>  dst
//! MovInt(dst, src) : src => dst
//! Phi(dst, Int, ..., pred_i, ...]) : ... pred_i-1 => dst, pred_i => dst ...
//! ```
//!
//! This graph corresponds to a set of (recursive) equation of the form
//!
//! ```text
//! Fields([c]) = {c}
//! Fields(node) = union_{n s.t. (n, node) is an edge} Fields(n)
//! ```
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
//! ```text
//! GetCol(1, [2])
//! StrToInt(0, 1)
//! GetCol(dst, 0)
//! ```
//!
//! Which corresponds roughly to the AWK snippet `$$2`, or "the field corresponding to the value of
//! the second column." We cannot predict this value ahead of time, for cases like this, we
//! contribute "full" sets to registers written by primitives that our analysis cannot introspect.

use std::fmt;

use crate::builtins::Variable;
use crate::bytecode::Instr;
use crate::common::NumTy;
use crate::compile::HighLevel;
use crate::dataflow::{self, JoinSemiLattice, Key};

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
const MAX_INDEX: usize = 62;
const FI_INDEX: usize = 63;
const FI_MASK: u64 = !(1 << FI_INDEX);

impl FieldSet {
    pub fn singleton(index: usize) -> FieldSet {
        if index > MAX_INDEX {
            Self::all()
        } else {
            FieldSet(1 << index)
        }
    }
    pub fn fi() -> FieldSet {
        FieldSet(1 << FI_INDEX)
    }
    pub fn has_fi(&self) -> bool {
        (self.0 != FieldSet::all().0) && ((1 << FI_INDEX) & self.0) != 0
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
        self.0 |= other.0;
    }

    /// Return a safe upper bound on the maximum integer value column represented by this set.
    ///
    /// The `fi` value is ignored. for full sets we return u32::max_value()
    pub fn max_value(&self) -> u32 {
        if self == &FieldSet::all() {
            u32::max_value()
        } else {
            self.max_bit()
        }
    }
    fn min_bit(&self) -> u32 {
        (FI_MASK & self.0).trailing_zeros()
    }
    fn max_bit(&self) -> u32 {
        64 - (FI_MASK & self.0).leading_zeros()
    }
    // Fill is used for `join` constructions, it fills all bits (inclusive) from the minimum bit in
    // self to the maximum bit in rhs.
    pub fn fill(&mut self, rhs: &FieldSet) {
        if rhs == &FieldSet::all() {
            *self = FieldSet::all();
            return;
        }

        let left = self.min_bit();
        let right = rhs.max_bit();
        if left >= right {
            return;
        }
        let rest = if right == 64 {
            !0
        } else {
            1u64.wrapping_shl(right).wrapping_sub(1)
        };
        let mask = if left == 64 {
            !0
        } else {
            1u64.wrapping_shl(left).wrapping_sub(1)
        };
        self.0 |= rest ^ mask;
    }
    pub fn get(&self, index: usize) -> bool {
        if index > MAX_INDEX {
            self.0 == Self::all().0
        } else {
            (1u64 << (index as u32)) & self.0 != 0
        }
    }
    pub fn set(&mut self, index: usize) {
        if index <= MAX_INDEX {
            self.0 |= 1u64 << index;
        } else {
            *self = Self::all();
        }
    }
}

impl JoinSemiLattice for FieldSet {
    type Func = ();
    fn bottom() -> Self {
        FieldSet::empty()
    }
    fn invoke(&mut self, other: &FieldSet, (): &()) -> bool /*changed*/ {
        let old = self.clone();
        self.union(other);
        self != &old
    }
}

impl fmt::Debug for FieldSet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self == &FieldSet::all() {
            return write!(f, "<ALL>");
        }
        let mut v: Vec<_> = (0..=MAX_INDEX)
            .filter(|i| self.get(*i))
            .map(|x| format!("{}", x))
            .collect();
        if self.has_fi() {
            v.push("FI[..]".into());
        }
        write!(f, "{:?}", v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fieldset_of_range(elts: impl Iterator<Item = usize>) -> FieldSet {
        let mut fs = FieldSet::empty();
        for e in elts {
            fs.set(e);
        }
        fs
    }
    fn fieldset_of_slice(elts: &[usize]) -> FieldSet {
        fieldset_of_range(elts.iter().cloned())
    }
    #[test]
    fn fill_test() {
        let mut fs1 = FieldSet::singleton(2);
        let fs2 = FieldSet::singleton(7);
        fs1.fill(&fs2);
        assert_eq!(fs1, fieldset_of_slice(&[2, 3, 4, 5, 6, 7]));

        let mut fs3 = fieldset_of_slice(&[3, 5, 7, 15]);
        let fs4 = fieldset_of_slice(&[6, 13, 23]);
        fs3.fill(&fs4);
        assert_eq!(fs3, fieldset_of_range(3usize..=23));

        let mut fs5 = FieldSet::singleton(1);
        let fs6 = FieldSet::singleton(62);
        fs5.fill(&fs6);
        assert_eq!(fs5, fieldset_of_range(1usize..=62));

        let mut fs7 = FieldSet::all();
        fs7.fill(&fs2);
        assert_eq!(fs7, FieldSet::all());
        let mut fs8 = FieldSet::singleton(3);
        fs8.fill(&fs7);
        assert_eq!(fs8, FieldSet::all());
    }

    #[test]
    fn max_value() {
        assert_eq!(FieldSet::singleton(23).max_value(), 24);
        assert_eq!(FieldSet::singleton(1024).max_value(), u32::max_value());
        assert_eq!(FieldSet::singleton(0).max_value(), 1);
    }
}

pub struct UsedFieldAnalysis {
    dfa: dataflow::Analysis<FieldSet>,
    // We could make the Join operation a member of FieldSet::Func but, while it is monotone, it
    // does not commute with union. The most general option here is probably to make Funcs
    // Semilattices themselves, and when solving to take the join of the functions before reading
    // the variables in question.  We can always add it in the future, but since join nodes are
    // always "leaves" we will just add the missing columns as a postprocessing step.
    joins: Vec<(Key /*lhs*/, Key /*rhs*/)>,
}

impl Default for UsedFieldAnalysis {
    fn default() -> UsedFieldAnalysis {
        let mut res = UsedFieldAnalysis {
            dfa: Default::default(),
            joins: Default::default(),
        };
        res.dfa.add_src(Key::Rng, FieldSet::all());
        res.dfa.add_src(Key::VarVal(Variable::FI), FieldSet::fi());
        res.dfa.add_src(Key::VarKey(Variable::FI), FieldSet::all());
        res
    }
}

impl UsedFieldAnalysis {
    pub(crate) fn visit_hl(&mut self, cur_fn_id: NumTy, inst: &HighLevel) {
        dataflow::boilerplate::visit_hl(inst, cur_fn_id, |dst, src| {
            self.dfa.add_dep(dst, src.unwrap(), ())
        })
    }
    pub(crate) fn visit_ll(&mut self, inst: &Instr) {
        use Instr::*;
        match inst {
            StoreConstInt(dst, i) if *i >= 0 => {
                self.dfa.add_src(dst, FieldSet::singleton(*i as usize))
            }

            LoadVarStrMap(_, Variable::FI)
            | StoreVarStrMap(Variable::FI, _)
            | Lookup { .. }
            | Store { .. }
            | IterBegin { .. }
            | IterGetNext { .. }
            | Mov(..) => dataflow::boilerplate::visit_ll(inst, |dst, src| {
                if let Some(src) = src {
                    self.dfa.add_dep(dst, src, ())
                } else {
                    self.dfa.add_src(dst, FieldSet::singleton(0))
                }
            }),
            GetColumn(dst, col_reg) => {
                self.dfa.add_query(col_reg);
                self.dfa.add_src(dst, FieldSet::all());
            }
            JoinCSV(dst, start, end)
            | JoinTSV(dst, start, end)
            | JoinColumns(dst, start, end, _) => {
                self.dfa.add_query(start);
                self.dfa.add_query(end);
                self.dfa.add_src(dst, FieldSet::all());
                self.joins.push((start.into(), end.into()));
            }
            _ => dataflow::boilerplate::visit_ll(inst, |dst, _| {
                self.dfa.add_src(dst, FieldSet::all())
            }),
        }
    }

    /// Return the set of all fields mentioned by column nodes.
    pub fn solve(mut self) -> FieldSet {
        let mut res = self.dfa.root().clone();
        for (l, r) in self.joins.iter().cloned() {
            let mut l_flds = self.dfa.query(l).clone();
            let r_flds = self.dfa.query(r);
            l_flds.fill(r_flds);
            res.union(&l_flds);
        }
        res
    }
}
