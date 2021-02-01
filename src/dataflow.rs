//! A common library for constructing dataflow analyses on frawk programs.
use crate::builtins::Variable;
use crate::bytecode::{Accum, Instr, Reg};
use crate::common::{Graph, NodeIx, NumTy, WorkList};
use crate::compile::{HighLevel, Ty};

use hashbrown::{HashMap, HashSet};
use petgraph::{
    visit::{Dfs, EdgeRef},
    Direction,
};

use std::convert::TryFrom;
use std::hash::Hash;
use std::mem;

/// A trait used for implementing Join Semilattices with a type for custom (monotone) binary
/// functions. We assume that Func::default() is the "join" operation on the Semilattice.
pub trait JoinSemiLattice {
    type Func: Default;
    fn bottom() -> Self;
    // invoke(other, &Default::default()) ~ *self = join(self, other);
    fn invoke(&mut self, other: &Self, f: &Self::Func) -> bool /* changed */;
}

pub(crate) struct Analysis<J: JoinSemiLattice, K = Key> {
    sentinel: NodeIx,
    nodes: HashMap<K, NodeIx>,
    graph: Graph<J, J::Func>,
    queries: HashSet<NodeIx>,
    solved: bool,
}

impl<K, J: JoinSemiLattice> Default for Analysis<J, K> {
    fn default() -> Analysis<J, K> {
        let mut res = Analysis {
            sentinel: Default::default(),
            nodes: Default::default(),
            graph: Default::default(),
            queries: Default::default(),
            solved: false,
        };
        res.sentinel = res.graph.add_node(J::bottom());
        res
    }
}

impl<K: Eq + Hash, J: JoinSemiLattice> Analysis<J, K> {
    pub(crate) fn add_src(&mut self, k: impl Into<K>, mut v: J) {
        let ix = self.get_node(k);
        let weight = self.graph.node_weight_mut(ix).unwrap();
        v.invoke(weight, &Default::default());
        *weight = v;
    }
    pub(crate) fn add_dep(&mut self, dst: impl Into<K>, src: impl Into<K>, edge: J::Func) {
        // We add dependencies in reverse order, so we can dfs for relevant nodes later on
        let dst_ix = self.get_node(dst);
        let src_ix = self.get_node(src);
        self.graph.add_edge(dst_ix, src_ix, edge);
    }
    fn get_node(&mut self, k: impl Into<K>) -> NodeIx {
        let graph = &mut self.graph;
        self.nodes
            .entry(k.into())
            .or_insert_with(|| graph.add_node(J::bottom()))
            .clone()
    }
    pub(crate) fn add_query(&mut self, k: impl Into<K>) {
        let ix = self.get_node(k);
        self.queries.insert(ix);
    }

    /// Call "solve" ahead of time to get a stable value here.
    pub(crate) fn query(&mut self, k: impl Into<K>) -> &J {
        self.solve();
        let ix = self.get_node(k);
        assert!(self.queries.contains(&ix));
        self.graph.node_weight(ix).unwrap()
    }

    /// Solves the constraints, then returns the join of all the queries.
    pub(crate) fn root(&mut self) -> &J {
        self.solve();
        self.graph.node_weight(self.sentinel).unwrap()
    }
    fn populate(&mut self, wl: &mut WorkList<NodeIx>) {
        let sentinel = self.sentinel;
        for node in self.queries.iter().cloned() {
            self.graph.add_edge(sentinel, node, Default::default());
        }
        let mut dfs = Dfs::new(&self.graph, sentinel);
        while let Some(ix) = dfs.next(&self.graph) {
            wl.insert(ix);
        }
    }
    fn solve(&mut self) {
        if self.solved {
            return;
        }
        self.solved = true;
        let mut wl = WorkList::default();
        self.populate(&mut wl);
        while let Some(n) = wl.pop() {
            let mut start = mem::replace(self.graph.node_weight_mut(n).unwrap(), J::bottom());
            let mut changed = false;
            for edge in self.graph.edges_directed(n, Direction::Outgoing) {
                let neigh = edge.target();
                let func = edge.weight();
                changed |= start.invoke(self.graph.node_weight(neigh).unwrap(), func);
            }
            mem::swap(&mut start, self.graph.node_weight_mut(n).unwrap());
            if !changed {
                continue;
            }
            for neigh in self.graph.neighbors_directed(n, Direction::Incoming) {
                wl.insert(neigh)
            }
        }
    }
}

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub(crate) enum Key {
    Reg(NumTy, Ty),
    MapKey(NumTy, Ty),
    MapVal(NumTy, Ty),
    Rng,
    Var(Variable),
    VarKey(Variable),
    VarVal(Variable),
    Slot(NumTy, Ty),
    Func(NumTy),
}

impl<T> From<&Reg<T>> for Key
where
    Reg<T>: Accum,
{
    fn from(r: &Reg<T>) -> Key {
        let (reg, ty) = r.reflect();
        Key::Reg(reg, ty)
    }
}

// TODO: loads/stores _of maps_ need to be bidirectional, because maps are referencey.
// TODO: wire into string constants

pub(crate) mod boilerplate {
    //! Some utility functions for discovering reads and writes in various parts of the IR.
    //! TODO: more precise tracking of function arguments.
    use super::*;

    pub(crate) fn visit_hl(
        inst: &HighLevel,
        cur_fn_id: NumTy,
        mut f: impl FnMut(/*dst*/ Key, /*src*/ Option<Key>),
    ) {
        use HighLevel::*;
        match inst {
            Call {
                func_id,
                dst_reg,
                dst_ty,
                args,
            } => {
                let dst_key = Key::Reg(*dst_reg, *dst_ty);
                f(dst_key.clone(), Some(Key::Func(*func_id)));
                for (reg, ty) in args.iter().cloned() {
                    f(dst_key.clone(), Some(Key::Reg(reg, ty)));
                }
            }
            Ret(reg, ty) => {
                f(Key::Func(cur_fn_id), Some(Key::Reg(*reg, *ty)));
            }
            Phi(reg, ty, preds) => {
                for (_, pred_reg) in preds.iter() {
                    f(Key::Reg(*reg, *ty), Some(Key::Reg(*pred_reg, *ty)));
                }
            }
            DropIter(..) => {}
        }
    }

    pub(crate) fn visit_ll(inst: &Instr, mut f: impl FnMut(/*dst*/ Key, /*src*/ Option<Key>)) {
        use Instr::*;
        match inst {
            StoreConstStr(dst, _) => f(dst.into(), None),
            StoreConstInt(dst, _) => f(dst.into(), None),
            StoreConstFloat(dst, _) => f(dst.into(), None),

            IntToStr(dst, src) => f(dst.into(), Some(src.into())),
            IntToFloat(dst, src) => f(dst.into(), Some(src.into())),
            FloatToStr(dst, src) => f(dst.into(), Some(src.into())),
            FloatToInt(dst, src) => f(dst.into(), Some(src.into())),
            StrToFloat(dst, src) => f(dst.into(), Some(src.into())),
            LenStr(dst, src) | StrToInt(dst, src) | HexStrToInt(dst, src) => f(dst.into(), Some(src.into())),

            Mov(ty, dst, src) => if !ty.is_array() {
                f(Key::Reg(*dst, *ty), Some(Key::Reg(*src, *ty)))
            } else {
                f(Key::MapKey(*dst, *ty), Some(Key::MapKey(*src, *ty)));
                f(Key::MapVal(*dst, *ty), Some(Key::MapVal(*src, *ty)));
                f(Key::MapKey(*src, *ty), Some(Key::MapKey(*dst, *ty)));
                f(Key::MapVal(*src, *ty), Some(Key::MapVal(*dst, *ty)));
            },
            AddInt(dst, x, y)
            | MulInt(dst, x, y)
            | MinusInt(dst, x, y)
            | ModInt(dst, x, y)
            | Int2(_, dst, x, y) => {
                f(dst.into(), Some(x.into()));
                f(dst.into(), Some(y.into()));
            }
            AddFloat(dst, x, y)
            | MulFloat(dst, x, y)
            | MinusFloat(dst, x, y)
            | ModFloat(dst, x, y)
            | Div(dst, x, y)
            | Pow(dst, x, y)
            | Float2(_, dst, x, y) => {
                f(dst.into(), Some(x.into()));
                f(dst.into(), Some(y.into()));
            }
            Not(dst, src) | NegInt(dst, src) | Int1(_, dst, src) => f(dst.into(), Some(src.into())),
            NegFloat(dst, src) | Float1(_, dst, src) => f(dst.into(), Some(src.into())),
            NotStr(dst, src) => f(dst.into(), Some(src.into())),
            Rand(dst) => f(dst.into(), Some(Key::Rng)),
            Srand(old, new) => {
                f(old.into(), Some(Key::Rng));
                f(Key::Rng, Some(new.into()));
            }
            ReseedRng(new) => f(Key::Rng, Some(new.into())),
            Concat(dst, x, y) => {
                f(dst.into(), Some(x.into()));
                f(dst.into(), Some(y.into()));
            }
            StartsWithConst(dst, x, _) => f(dst.into(), Some(x.into())),

            // NB: this assumes that regexes that have been constant-folded are not tainted by
            // user-input. That is certainly true today, but any kind of dynamic simplification or
            // inlining could change that.
            MatchConst(dst, x, _) | IsMatchConst(dst, x, _) => f(dst.into(), Some(x.into())),
            IsMatch(dst, x, y) | Match(dst, x, y) | SubstrIndex(dst, x, y) => {
                f(dst.into(), Some(x.into()));
                f(dst.into(), Some(y.into()));
            }
            GSub(dst, x, y, dstin) | Sub(dst, x, y, dstin) => {
                f(dst.into(), Some(x.into()));
                f(dst.into(), Some(y.into()));
                f(dstin.into(), Some(x.into()));
                f(dstin.into(), Some(y.into()));
            }
            EscapeTSV(dst, src) | EscapeCSV(dst, src) => f(dst.into(), Some(src.into())),
            Substr(dst, x, y, z) => {
                f(dst.into(), Some(x.into()));
                f(dst.into(), Some(y.into()));
                f(dst.into(), Some(z.into()));
            }
            LTFloat(dst, x, y)
            | GTFloat(dst, x, y)
            | LTEFloat(dst, x, y)
            | GTEFloat(dst, x, y)
            | EQFloat(dst, x, y) => {
                    f(dst.into(), Some(x.into()));
                    f(dst.into(), Some(y.into()));
                }
            LTInt(dst, x, y)
            | GTInt(dst, x, y)
            | LTEInt(dst, x, y)
            | GTEInt(dst, x, y)
            | EQInt(dst, x, y) => {
                f(dst.into(), Some(x.into()));
                f(dst.into(), Some(y.into()));
            }
            LTStr(dst, x, y)
            | GTStr(dst, x, y)
            | LTEStr(dst, x, y)
            | GTEStr(dst, x, y)
            | EQStr(dst, x, y) => {
                f(dst.into(), Some(x.into()));
                f(dst.into(), Some(y.into()));
            }
            GetColumn(dst, _) => f(dst.into(), None),
            JoinTSV(dst, start, end) | JoinCSV(dst, start, end) => {
                f(dst.into(), Some(start.into()));
                f(dst.into(), Some(end.into()));
            }
            JoinColumns(dst, x, y, z) => {
                f(dst.into(), Some(x.into()));
                f(dst.into(), Some(y.into()));
                f(dst.into(), Some(z.into()));
            }
            ReadErr(dst, _cmd, _) => f(dst.into(), None),
            NextLine(dst, _cmd, _) => f(dst.into(), None),
            ReadErrStdin(dst) => f(dst.into(), None),
            NextLineStdin(dst) => f(dst.into(), None),
            SplitInt(dst1, src1, dst2, src2) => {
                f(dst1.into(), Some(src1.into()));
                f(dst1.into(), Some(src2.into()));
                let (dst2_reg, dst2_ty) = dst2.reflect();
                debug_assert!(dst2_ty.is_array());
                f(Key::MapVal(dst2_reg, dst2_ty), Some(src1.into()));
                f(Key::MapVal(dst2_reg, dst2_ty), Some(src2.into()));
            }
            SplitStr(dst1, src1, dst2, src2) => {
                f(dst1.into(), Some(src1.into()));
                f(dst1.into(), Some(src2.into()));
                f(dst2.into(), Some(src1.into()));
                f(dst2.into(), Some(src2.into()));
            }
            Sprintf { dst, fmt, args } => {
                f(dst.into(), Some(fmt.into()));
                for (reg, ty) in args.iter() {
                    f(dst.into(), Some(Key::Reg(*reg, *ty)));
                }
            }
            RunCmd(dst, _) => f(dst.into(), None),
            Lookup {
                map_ty,
                dst,
                map,
                key,
            } => {
                // lookups are also writes to the keys
                f(Key::MapKey(*map, *map_ty), Some(Key::Reg(*key, map_ty.key().unwrap())));
                // a null value will be inserted as a value into the map
                f(Key::MapVal(*map, *map_ty), None);
                f(Key::Reg(*dst, map_ty.val().unwrap()), Some(Key::MapVal(*map, *map_ty)))
            },
            Len { map_ty, dst, map } => f(Key::Reg(*dst, Ty::Int), Some(Key::Reg(*map, *map_ty))),
            Store { map_ty, map, key, val } => {
                f(Key::MapKey(*map, *map_ty), Some(Key::Reg(*key, map_ty.key().unwrap())));
                f(Key::MapVal(*map, *map_ty), Some(Key::Reg(*val, map_ty.val().unwrap())));
            }
            IterBegin { map_ty, dst, map } => {
                f(Key::Reg(*dst, map_ty.key_iter().unwrap()), Some(Key::MapKey(*map, *map_ty)))
            }
            IterGetNext{iter_ty, dst, iter} => {
                f(Key::Reg(*dst, iter_ty.iter().unwrap()), Some(Key::Reg(*iter, *iter_ty)));
            }
            LoadVarStr(dst, v) => f(dst.into(), Some(Key::Var(*v))),
            LoadVarInt(dst, v) => f(dst.into(), Some(Key::Var(*v))),
            StoreVarIntMap(v, reg) | LoadVarIntMap(reg, v) => {
                let (reg, ty) = reg.reflect();

                f(Key::MapKey(reg, ty), Some(Key::VarKey(*v)));
                f(Key::MapVal(reg, ty), Some(Key::VarVal(*v)));
                f(Key::VarKey(*v), Some(Key::MapKey(reg, ty)));
                f(Key::VarVal(*v), Some(Key::MapVal(reg, ty)));
            },
            StoreVarStrMap(v, reg) | LoadVarStrMap(reg, v) => {
                let (reg, ty) = reg.reflect();

                f(Key::MapKey(reg, ty), Some(Key::VarKey(*v)));
                f(Key::MapVal(reg, ty), Some(Key::VarVal(*v)));
                f(Key::VarKey(*v), Some(Key::MapKey(reg, ty)));
                f(Key::VarVal(*v), Some(Key::MapVal(reg, ty)));
            },
            StoreVarStr(v, src) => f(Key::Var(*v), Some(src.into())),
            StoreVarInt(v, src) => f(Key::Var(*v), Some(src.into())),
            LoadSlot{ty,slot,dst} =>
                f(Key::Reg(*dst, *ty), Some(Key::Slot(u32::try_from(*slot).expect("slot too large"), *ty))),
            StoreSlot{ty,slot,src} =>
                f(Key::Slot(u32::try_from(*slot).expect("slot too large"), *ty), Some(Key::Reg(*src, *ty))),
            Delete{..}
            | UpdateUsedFields()
            | SetFI(..)
            | PrintAll{..}
            | Contains{..} // 0 or 1
            | IterHasNext{..}
            | JmpIf(..)
            | Jmp(_)
            | Halt
            | Push(..)
            | Pop(..)
            // We consume high-level instructions, so calls and returns are handled by visit_hl
            // above
            | Call(_)
            | Ret
            | Printf { .. }
            | Close(_)
            | NextLineStdinFused()
            | NextFile()
            | SetColumn(_, _)
            | AllocMap(_, _) => {}
        }
    }
}
