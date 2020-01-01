use crate::builtins;
use crate::cfg::{self, Function, Ident, ProgramContext};
use crate::common::{self, NodeIx, NumTy, Result};
use crate::compile;
use hashbrown::{HashMap, HashSet};

use std::ops::{Deref, DerefMut};

pub(crate) type SmallVec<T> = smallvec::SmallVec<[T; 2]>;
// TODO: rationalize null handling. We want to infer that all 0-subscripted variables are null, and
// use that information when doing type deduction. Conversions at he bytecode level just insert
// constants. We want to make sure that
// { SUM += 1} END {print SUM}
// has SUM as an integer though.
//  start:
//  body:
//  SUM_1 = phi [ SUM_0: start, SUM_2: body]
//  SUM_2 = SUM_1 + 1;
//  jmp body;
// Seems like SUM_0 cannot be treated as a string here... That makes this difficult because we want
// to also handle
// function x(y) { if (y) { return 1;} }
// print x(0),x(1) # " 1"
// Which implies that we can't just coerce this to a phi node. What matters is the "use" of a
// variable down the line. Perhaps we can encode that as some kind of constraint. For now, let's
// proceed with the Integer semantics (breaking some uses of `x` for the moment).
//
// Yes. I think we will eventually have to compute an inductive IsPrinted relation on certain
// types, and treat them differently when doing phi [ NULL, INT ]. Distinguishing Null from None
// will help here, at least form a debugability perspective (None means we just haven't gotten to
// the variable yet, Null means it was really never written to).
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub(crate) enum BaseTy {
    Null,
    Int,
    Float,
    Str,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub(crate) enum TVar<T> {
    Iter(T),
    Scalar(T),
    Map { key: T, val: T },
}

impl<T> TVar<T> {
    // deriving Functor ...
    fn map<S>(&self, mut f: impl FnMut(&T) -> S) -> TVar<S> {
        use TVar::*;
        match self {
            Iter(t) => Iter(f(t)),
            Scalar(t) => Scalar(f(t)),
            Map { key, val } => Map {
                key: f(key),
                val: f(val),
            },
        }
    }
}

impl<T: Clone> TVar<T> {
    pub(crate) fn abs(&self) -> Option<TVar<Option<T>>> {
        Some(self.map(|t| Some(t.clone())))
    }
}

#[derive(Clone, Eq, PartialEq)]
pub(crate) enum Constraint<T> {
    // TODO(ezr): Better names for Key/KeyIn etc.
    KeyIn(T),
    Key(T),
    ValIn(T),
    Val(T),
    IterVal(T),
    IterValIn(T),
    Flows(T),
    // TODO(ezr): have a shared Vec and just store a slice here?
    CallBuiltin(SmallVec<NodeIx>, builtins::Function),
    CallUDF(SmallVec<NodeIx>, NumTy),
}

impl<T> Constraint<T> {
    fn sub<S>(&self, s: S) -> Constraint<S> {
        match self {
            Constraint::KeyIn(_) => Constraint::KeyIn(s),
            Constraint::Key(_) => Constraint::Key(s),
            Constraint::ValIn(_) => Constraint::ValIn(s),
            Constraint::Val(_) => Constraint::Val(s),
            Constraint::IterValIn(_) => Constraint::IterValIn(s),
            Constraint::IterVal(_) => Constraint::IterVal(s),
            Constraint::Flows(_) => Constraint::Flows(s),
            Constraint::CallBuiltin(args, f) => Constraint::CallBuiltin(args.clone(), f.clone()),
            Constraint::CallUDF(args, f) => Constraint::CallUDF(args.clone(), f.clone()),
        }
    }
    fn is_flow(&self) -> bool {
        if let Constraint::Flows(_) = self {
            true
        } else {
            false
        }
    }
}

impl Constraint<State> {
    fn eval<'a, 'b>(&self, tc: &mut TypeContext<'a, 'b>) -> Result<State> {
        match self {
            Constraint::KeyIn(None) => Ok(None),
            Constraint::KeyIn(Some(TVar::Scalar(k))) => Ok(Some(TVar::Map {
                key: k.clone(),
                val: None,
            })),
            Constraint::KeyIn(op) => err!("Non-scalar KeyIn constraint: {:?}", op),

            Constraint::Key(None) => Ok(None),
            Constraint::Key(Some(TVar::Map { key: s, .. })) => Ok(Some(TVar::Scalar(s.clone()))),
            Constraint::Key(op) => {
                err!("invalid operand for Key constraint: {:?} (must be map)", op)
            }

            Constraint::ValIn(None) => Ok(None),
            Constraint::ValIn(Some(TVar::Scalar(v))) => Ok(Some(TVar::Map {
                key: None,
                val: v.clone(),
            })),
            Constraint::ValIn(op) => err!("Non-scalar ValIn constraint: {:?}", op),

            Constraint::Val(None) => Ok(None),
            Constraint::Val(Some(TVar::Map { val: s, .. })) => Ok(Some(TVar::Scalar(s.clone()))),
            Constraint::Val(op) => {
                err!("invalid operand for Val constraint: {:?} (must be map)", op)
            }

            Constraint::IterValIn(None) => Ok(None),
            Constraint::IterValIn(Some(TVar::Scalar(v))) => Ok(Some(TVar::Iter(v.clone()))),
            Constraint::IterValIn(op) => err!("Non-scalar IterValIn constraint: {:?}", op),

            Constraint::IterVal(None) => Ok(None),
            Constraint::IterVal(Some(TVar::Iter(s))) => Ok(Some(TVar::Scalar(s.clone()))),
            Constraint::IterVal(op) => err!(
                "invalid operand for IterVal constraint: {:?} (must be iterator)",
                op
            ),

            Constraint::Flows(s) => Ok(s.clone()),
            Constraint::CallBuiltin(args, f) => {
                let arg_state: SmallVec<State> =
                    args.iter().map(|ix| tc.nw.read(*ix).clone()).collect();
                f.step(&arg_state[..])
            }
            Constraint::CallUDF(args, f) => {
                let arg_state: SmallVec<State> =
                    args.iter().map(|ix| tc.nw.read(*ix).clone()).collect();
                let ret_ix = tc.get_function(&tc.func_table[*f as usize], &arg_state);
                Ok(tc.nw.read(ret_ix).clone())
            }
        }
    }
}

// We distinguish Edge from Constraint because we want to add more data later.
#[derive(Clone)]
struct Edge {
    // TODO: do we need a separate edge type, or can it just be Constraint<()>
    constraint: Constraint<()>,
}

// encode deps as a single edge from function to Var?

#[derive(Copy, Clone)]
enum Rule {
    Var,
    Const(State),
}

pub(crate) type State = Option<TVar<Option<BaseTy>>>;

pub(crate) fn null_ty() -> compile::Ty {
    flatten(concrete(None)).unwrap()
}

impl Rule {
    // TODO(ezr): Why also have `prev`? This allows us to only hand the deps back that have changed
    // since the last update. this extra functionality is not yet implemented; it's unclear if we
    // can use it while still relying on petgraph, or if we'll have to (e.g.) store a priority
    // queue of edges per node ordered by modified timestamp.
    fn step(&self, prev: &State, deps: &[State]) -> Result<State> {
        fn value_rule(b1: BaseTy, b2: BaseTy) -> BaseTy {
            use BaseTy::*;
            match (b1, b2) {
                (Null, _) | (_, Null) | (Str, _) | (_, Str) => Str,
                (Float, _) | (_, Float) => Float,
                (Int, Int) => Int,
            }
        }
        if let Rule::Const(tv) = self {
            return Ok(tv.clone());
        }
        let mut cur = prev.clone();
        for d in deps.iter().cloned() {
            use TVar::*;
            cur = match (cur, d) {
                (None, x) | (x, None) => x,
                (Some(x), Some(y)) => match (x, y) {
                    (Iter(x), Iter(None)) | (Iter(None), Iter(x)) => Some(Iter(x)),
                    (Iter(Some(x)), Iter(Some(y))) => {
                        if x == y {
                            cur
                        } else {
                            return err!("Incompatible iterator types: {:?} vs. {:?}", x, y);
                        }
                    }
                    (Scalar(x), Scalar(None)) | (Scalar(None), Scalar(x)) => Some(Scalar(x)),
                    (Scalar(Some(x)), Scalar(Some(y))) => Some(Scalar(Some(value_rule(x, y)))),
                    (Map { key: k1, val: v1 }, Map { key: k2, val: v2 }) => {
                        fn join_key(b1: BaseTy, b2: BaseTy) -> BaseTy {
                            use BaseTy::*;
                            match (b1, b2) {
                                (Float, _)
                                | (_, Float)
                                | (Str, _)
                                | (_, Str)
                                | (Null, _)
                                | (_, Null) => Str,
                                (Int, _) => Int,
                            }
                        }
                        fn lift(
                            f: impl Fn(BaseTy, BaseTy) -> BaseTy,
                            o1: Option<BaseTy>,
                            o2: Option<BaseTy>,
                        ) -> Option<BaseTy> {
                            match (o1, o2) {
                                (Some(x), Some(y)) => Some(f(x, y)),
                                (Some(x), None) | (None, Some(x)) => Some(x),
                                (None, None) => None,
                            }
                        }
                        Some(Map {
                            key: lift(join_key, k1, k2),
                            val: lift(value_rule, v1, v2),
                        })
                    }
                    (t1, t2) => return err!("kinds do not match. {:?} vs {:?}", t1, t2),
                },
            };
        }
        Ok(cur)
    }
}

fn concrete(state: State) -> TVar<BaseTy> {
    fn concrete_scalar(o: Option<BaseTy>) -> BaseTy {
        match o {
            Some(s) => s,
            None => BaseTy::Null,
        }
    }

    {
        use TVar::*;
        match state {
            Some(Iter(i)) => Iter(concrete_scalar(i)),
            Some(Scalar(i)) => Scalar(concrete_scalar(i)),
            Some(Map { key, val }) => Map {
                key: {
                    let k = concrete_scalar(key);
                    if let BaseTy::Null = k {
                        BaseTy::Str
                    } else {
                        k
                    }
                },
                val: concrete_scalar(val),
            },
            None => Scalar(BaseTy::Null),
        }
    }
}

fn flatten(tv: TVar<BaseTy>) -> Result<compile::Ty> {
    use compile::Ty;
    use {BaseTy::*, TVar::*};
    match tv {
        Scalar(Int) => Ok(Ty::Int),
        Scalar(Float) => Ok(Ty::Float),
        // TODO(ezr): add Null to compile::Ty?
        Scalar(Null) | Scalar(Str) => Ok(Ty::Str),
        Iter(Int) => Ok(Ty::IterInt),
        Iter(Null) | Iter(Str) => Ok(Ty::IterStr),
        Iter(x) => err!("Iterator over an unsupported type: {:?}", x),
        Map { key: Int, val: Int } => Ok(Ty::MapIntInt),
        Map {
            key: Int,
            val: Float,
        } => Ok(Ty::MapIntFloat),
        Map { key: Int, val: Str } => Ok(Ty::MapIntStr),
        Map { key: Str, val: Int } => Ok(Ty::MapStrInt),
        Map {
            key: Str,
            val: Float,
        } => Ok(Ty::MapStrFloat),
        Map { key: Str, val: Str } => Ok(Ty::MapStrStr),
        Map { key, val } => err!("Map with unsupported type (key={:?} val={:?})", key, val),
    }
}

#[derive(Clone)]
struct Node {
    rule: Rule,
    cur_val: State,
    // TODO(ezr): put debugging information in here?
}

impl Node {
    fn new(rule: Rule) -> Node {
        Node {
            rule,
            cur_val: None,
        }
    }
}

pub(crate) struct Network {
    base_node: NodeIx,
    wl: common::WorkList<NodeIx>,
    call_deps: HashMap<NodeIx, SmallVec<NodeIx>>,
    graph: common::Graph<Node, Edge>,
    iso: HashSet<(NumTy, NumTy)>,
}

impl Default for Network {
    fn default() -> Network {
        let mut graph = common::Graph::default();
        let base_node = graph.add_node(Node::new(Rule::Var));
        Network {
            graph,
            base_node,
            wl: Default::default(),
            call_deps: Default::default(),
            iso: Default::default(),
        }
    }
}

fn is_iso(set: &HashSet<(NumTy, NumTy)>, n1: NodeIx, n2: NodeIx) -> bool {
    use std::cmp::{max, min};
    let i1 = n1.index() as NumTy;
    let i2 = n2.index() as NumTy;
    set.contains(&(min(i1, i2), max(i1, i2)))
}

fn make_iso(set: &mut HashSet<(NumTy, NumTy)>, n1: NodeIx, n2: NodeIx) -> bool {
    use std::cmp::{max, min};
    let i1 = n1.index() as NumTy;
    let i2 = n2.index() as NumTy;
    set.insert((min(i1, i2), max(i1, i2)))
}

impl Network {
    fn add_rule(&mut self, rule: Rule) -> NodeIx {
        let res = self.graph.add_node(Node::new(rule));
        self.wl.insert(res);
        res
    }

    fn read(&self, ix: NodeIx) -> &State {
        &self.graph.node_weight(ix).unwrap().cur_val
    }
    pub(crate) fn add_dep(&mut self, from: NodeIx, to: NodeIx, constraint: Constraint<()>) {
        self.graph.add_edge(from, to, Edge { constraint });
        self.wl.insert(from);
    }
}

#[derive(Default)]
pub(crate) struct TypeContext<'a, 'b> {
    pub(crate) nw: Network,
    base: HashMap<State, NodeIx>,
    env: HashMap<Args<Ident>, NodeIx>,
    funcs: HashMap<Args<NumTy>, NodeIx>,
    func_table: &'a [Function<'b, &'b str>],
}

struct View<'a, 'b, 'c> {
    tc: &'a mut TypeContext<'b, 'c>,
    frame_id: NumTy, // which function are we in?
    frame_args: SmallVec<State>,
}

impl<'a, 'b, 'c> Deref for View<'a, 'b, 'c> {
    type Target = &'a mut TypeContext<'b, 'c>;
    fn deref(&self) -> &&'a mut TypeContext<'b, 'c> {
        &self.tc
    }
}

impl<'a, 'b, 'c> DerefMut for View<'a, 'b, 'c> {
    fn deref_mut(&mut self) -> &mut &'a mut TypeContext<'b, 'c> {
        &mut self.tc
    }
}

pub(crate) fn get_types<'a>(pc: &ProgramContext<'a, &'a str>) -> Result<TypeInfo> {
    TypeContext::from_prog(pc)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Args<T> {
    id: T,
    func_id: Option<NumTy>,
    args: SmallVec<State>, // ignored when id.global
}

pub(crate) struct TypeInfo {
    // Map a particular identifier in a function to a type.
    pub var_tys: HashMap<(Ident, NumTy, SmallVec<compile::Ty>), compile::Ty>,
    // Map a particular function invocation to a return type.
    pub func_tys: HashMap<(NumTy, SmallVec<compile::Ty>), compile::Ty>,
}

impl<'b, 'c> TypeContext<'b, 'c> {
    pub(crate) fn from_prog<'a>(pc: &ProgramContext<'a, &'a str>) -> Result<TypeInfo> {
        let mut tc = TypeContext::default();
        tc.func_table = &pc.funcs[..];
        let main = &pc.funcs[pc.main_offset];
        let empty: SmallVec<State> = Default::default();
        tc.get_function(main, &empty);
        tc.solve()?;
        let mut var_tys = HashMap::new();
        let mut func_tys = HashMap::new();
        for (Args { id, args, .. }, ix) in tc.funcs.iter() {
            let mut flat_args = SmallVec::new();
            for a in args.iter().cloned() {
                flat_args.push(flatten(concrete(a))?);
            }
            if let hashbrown::hash_map::Entry::Vacant(v) = func_tys.entry((*id, flat_args)) {
                v.insert(flatten(concrete(*tc.nw.read(*ix)))?);
            }
        }
        for (Args { id, func_id, args }, ix) in tc.env.iter() {
            let mut flat_args = SmallVec::new();
            for a in args.iter().cloned() {
                flat_args.push(flatten(concrete(a))?);
            }
            let v = flatten(concrete(*tc.nw.read(*ix)))?;

            // We won't use the function id if id.global, so setting it to 0 should be fine.
            // TODO clean up some of this to make it less misleading
            if let Some(prev) = var_tys.insert((*id, func_id.unwrap_or(0), flat_args), v) {
                if prev != v {
                    return err!(
                        "coherence violation! {:?} in args {:?}, we get both {:?} and {:?}",
                        id,
                        args,
                        v,
                        prev
                    );
                }
            }
        }
        Ok(TypeInfo { var_tys, func_tys })
    }
    fn solve(&mut self) -> Result<()> {
        let mut dep_indices: SmallVec<NodeIx> = Default::default();
        let mut deps: SmallVec<State> = Default::default();
        while let Some(ix) = self.nw.wl.pop() {
            deps.clear();
            dep_indices.clear();
            let Node { rule, cur_val } = self.nw.graph.node_weight(ix).unwrap().clone();
            // Iterate over the incoming edges; read their current values and evaluate the
            // constraints.
            use petgraph::Direction::Incoming;
            let mut walker = self.nw.graph.neighbors_directed(ix, Incoming).detach();
            while let Some((e_ix, node_ix)) = walker.next(&self.nw.graph) {
                let edge = self.nw.graph.edge_weight(e_ix).unwrap().clone();
                let node_val = self.nw.graph.node_weight(node_ix).unwrap().cur_val.clone();
                deps.push(edge.constraint.sub(node_val).eval(self)?);
                if edge.constraint.is_flow() {
                    dep_indices.push(node_ix);
                }
            }
            // Compute an update value based on the newly-evaluated constraints.
            let next = rule.step(&cur_val, &deps[..])?;
            if next == cur_val {
                continue;
            }
            if let Some(TVar::Map { .. }) = next {
                // If we have a map node, then we need to make sure that anything that assigns to
                // it winds up with the same type. Assignments are marked as `Flows` constraints;
                // and we recorded the nodes that flow into `ix` in `dep_indices`. Now we just add
                // another arrow back, unless we have done so before. We figure that last part out
                // by using the `iso` functions, which take care to sort the indices before before
                // inserting them into the hash set so we only store them once.
                for d in dep_indices.iter().cloned() {
                    if is_iso(&self.nw.iso, ix, d) {
                        continue;
                    }
                    self.nw.graph.add_edge(
                        ix,
                        d,
                        Edge {
                            constraint: Constraint::Flows(()),
                        },
                    );
                    make_iso(&mut self.nw.iso, ix, d);
                }
            }
            self.nw.graph.node_weight_mut(ix).unwrap().cur_val = next;
            for n in self.nw.graph.neighbors(ix) {
                self.nw.wl.insert(n);
            }
            for c in self.nw.call_deps.get(&ix).iter().flat_map(|c| c.iter()) {
                self.nw.wl.insert(*c);
            }
        }
        Ok(())
    }
    pub(crate) fn set_key(&mut self, arr: NodeIx, key: NodeIx) {
        self.nw.add_dep(key, arr, Constraint::KeyIn(()));
        self.nw.add_dep(arr, key, Constraint::Key(()));
    }

    fn set_val(&mut self, arr: NodeIx, val: NodeIx) {
        self.nw.add_dep(val, arr, Constraint::ValIn(()));
        self.nw.add_dep(arr, val, Constraint::Val(()));
    }

    fn set_iter_val(&mut self, iter: NodeIx, val: NodeIx) {
        self.nw.add_dep(val, iter, Constraint::IterValIn(()));
        self.nw.add_dep(iter, val, Constraint::IterVal(()));
    }
    pub(crate) fn constant(&mut self, tv: State) -> NodeIx {
        use hashbrown::hash_map::Entry::*;
        match self.base.entry(tv) {
            Occupied(o) => *o.get(),
            Vacant(v) => {
                let res = self.nw.add_rule(Rule::Const(tv));
                v.insert(res);
                res
            }
        }
    }
    pub(crate) fn get_node(&mut self, key: Args<Ident>) -> NodeIx {
        self.env
            .entry(key)
            .or_insert(self.nw.add_rule(Rule::Var))
            .clone()
    }

    fn get_function<'a>(
        &mut self,
        Function {
            ident, cfg, args, ..
        }: &Function<'a, &'a str>,
        arg_states: &SmallVec<State>,
    ) -> NodeIx {
        let mut key = Args {
            id: *ident,
            func_id: None,
            args: arg_states.clone(),
        };
        // First we want to normalize the provided arguments. If we provide too few arguments, the
        // rest are filled with nulls. If we provide too many arguments, we throw away the extras.
        if key.args.len() < args.len() {
            for _ in 0..(args.len() - key.args.len()) {
                key.args.push(None);
            }
        }
        if args.len() < key.args.len() {
            key.args.truncate(args.len());
        }

        // Check if we have already created the function
        if let Some(ix) = self.funcs.get(&key) {
            return *ix;
        }
        // Just return a new node for the return value. We will infer return types by looking up
        // this node later and adding dependencies when we encounter a `Return` stmt.
        //
        // TODO: this means we do some duplicate work in rewriting returns in the cfg module.

        // Create a new function.
        let res = self.nw.add_rule(Rule::Var);
        self.funcs.insert(key.clone(), res);
        let mut view = View {
            tc: self,
            frame_id: *ident,
            frame_args: key.args.clone(),
        };

        // Apply the arguments appropriately:
        for (cfg::Arg { id, .. }, state) in args.iter().zip(key.args.iter()) {
            let ix = view.ident_node(id);
            let cnode = view.constant(*state);
            view.nw.add_dep(cnode, ix, Constraint::Flows(()));
        }
        let nodes = cfg.raw_nodes();
        for bb in nodes {
            for stmt in bb.weight.q.iter() {
                view.constrain_stmt(stmt);
            }
        }
        res
    }
}

impl<'b, 'c, 'd> View<'b, 'c, 'd> {
    fn add_builtin_call(&mut self, f: builtins::Function, args: SmallVec<NodeIx>, to: NodeIx) {
        f.feedback(&args[..], self);
        for arg in args.iter() {
            self.nw
                .call_deps
                .entry(*arg)
                .or_insert(Default::default())
                .push(to);
        }
        let from = self.nw.base_node;
        self.nw.add_dep(from, to, Constraint::CallBuiltin(args, f));
        self.nw.wl.insert(to);
    }
    fn add_udf_call(&mut self, f: NumTy, args: SmallVec<NodeIx>, to: NodeIx) {
        for arg in args.iter() {
            self.nw
                .call_deps
                .entry(*arg)
                .or_insert(Default::default())
                .push(to);
        }
        let from = self.nw.base_node;
        self.nw.add_dep(from, to, Constraint::CallUDF(args, f));
        self.nw.wl.insert(to);
    }

    fn constrain_stmt<'a>(&mut self, stmt: &cfg::PrimStmt<'a>) {
        use cfg::PrimStmt::*;
        match stmt {
            AsgnIndex(arr, ix, v) => {
                let arr_ix = self.ident_node(arr);
                let ix_ix = self.val_node(ix);
                // TODO(ezr): set up caching for keys, values of maps and iterators?
                self.set_key(arr_ix, ix_ix);
                let val_ix = self.nw.add_rule(Rule::Var);
                self.set_val(arr_ix, val_ix);
                self.constrain_expr(v, val_ix);
            }
            AsgnVar(v, e) => {
                let v_ix = self.ident_node(v);
                self.constrain_expr(e, v_ix);
            }
            Return(v) => {
                let v_ix = self.val_node(v);
                let cur_func_key = Args {
                    id: self.frame_id,
                    func_id: None,
                    args: self.frame_args.clone(),
                };
                let ret_ix = self.tc.funcs[&cur_func_key];
                self.nw.add_dep(v_ix, ret_ix, Constraint::Flows(()));
            }
            // Builtins have fixed types; no constraint generation is necessary.
            SetBuiltin(_, _) => {}
        }
    }

    fn constrain_expr<'a>(&mut self, expr: &cfg::PrimExpr<'a>, to: NodeIx) {
        use cfg::PrimExpr::*;
        match expr {
            Val(pv) => {
                let pv_ix = self.val_node(pv);
                self.nw.add_dep(pv_ix, to, Constraint::Flows(()));
            }
            Phi(preds) => {
                for (_, id) in preds.iter() {
                    let id_ix = self.ident_node(id);
                    self.nw.add_dep(id_ix, to, Constraint::Flows(()));
                }
            }
            CallBuiltin(f, args) => {
                let args: SmallVec<NodeIx> = args.iter().map(|arg| self.val_node(arg)).collect();
                self.add_builtin_call(*f, args, to);
            }
            CallUDF(f, args) => {
                let args: SmallVec<NodeIx> = args.iter().map(|arg| self.val_node(arg)).collect();
                self.add_udf_call(*f, args, to);
            }
            Index(arr, ix) => {
                let arr_ix = self.val_node(arr);
                let ix_ix = self.val_node(ix);
                self.set_key(arr_ix, ix_ix);
                self.set_val(arr_ix, to);
            }
            IterBegin(arr) => {
                let arr_ix = self.val_node(arr);
                let iter_ix = to;
                let key_ix = self.nw.add_rule(Rule::Var);

                // The key of the map and the iterator must be the same.
                self.set_key(arr_ix, key_ix);
                self.set_iter_val(iter_ix, key_ix);
            }
            HasNext(_) => {
                let int = self.constant(TVar::Scalar(BaseTy::Int).abs());
                self.nw.add_dep(int, to, Constraint::Flows(()))
            }
            Next(iter) => {
                let iter_ix = self.val_node(iter);
                self.set_iter_val(iter_ix, to);
            }
            LoadBuiltin(bv) => {
                let bv_ix = self.constant(bv.ty().abs());
                self.nw.add_dep(bv_ix, to, Constraint::Flows(()));
            }
        };
    }

    fn val_node<'a>(&mut self, val: &cfg::PrimVal<'a>) -> NodeIx {
        use cfg::PrimVal::*;
        match val {
            Var(id) => self.ident_node(id),
            ILit(_) => self.constant(TVar::Scalar(BaseTy::Int).abs()),
            FLit(_) => self.constant(TVar::Scalar(BaseTy::Float).abs()),
            StrLit(_) => self.constant(TVar::Scalar(BaseTy::Str).abs()),
        }
    }

    fn ident_node(&mut self, id: &Ident) -> NodeIx {
        let key = Args {
            id: *id,
            func_id: if id.global { None } else { Some(self.frame_id) },
            args: if id.global {
                Default::default()
            } else {
                self.frame_args.clone()
            },
        };
        self.get_node(key)
    }
}
