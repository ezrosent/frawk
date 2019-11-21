//! Algorithms and types pertaining to type deduction and converion.
//!
//! TODO: update this with more documentation when the algorithms are more fully baked.
use std::iter::once;

use crate::builtins::Function;
use crate::cfg::{self, Ident, PrimExpr, PrimStmt, PrimVal};
use crate::common::{Either, Graph, NodeIx, NumTy, Result};
use hashbrown::{hash_map::Entry, HashMap};

pub(crate) type SmallVec<T> = smallvec::SmallVec<[T; 2]>;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum Scalar {
    Str,
    Int,
    Float,
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
// TODO: migrate Kind to TVar<()> ?
pub(crate) enum Kind {
    Scalar,
    Iter,
    Map,
}

/// A propagator is a monotone function that receives "partially done" inputs and produces
/// "partially done" outputs. This is a very restricted API to simplify the implementation of a
/// propagator network.
///
/// TODO: Link to more information on propagators.
pub(crate) trait Propagator {
    type Item;
    /// Propagator rules can either have a fixed arity, or they can consume an arbitrary number
    /// of inputs. The inputs to a propagator rule must have the same arity as `arity`, if
    /// there is one.
    fn arity(&self) -> Option<usize>;

    /// Ingest new information from `incoming`, produce a new output and indicate if the output has
    /// been saturated (i.e. regardless of what new inputs it will produce, the output will not
    /// change)
    fn step(&self, incoming: &[Option<Self::Item>]) -> (bool, Option<Self::Item>);
}

pub(crate) mod prop {
    use super::{once, Function, Graph, NodeIx, Propagator, Result, Scalar, SmallVec};
    use std::fmt::{self, Debug};
    fn fold_option<'a, T: Clone + 'a>(
        incoming: impl Iterator<Item = &'a T>,
        f: impl Fn(Option<T>, &T) -> (bool, Option<T>),
    ) -> (bool, Option<T>) {
        let mut start = None;
        let mut done = false;
        for i in incoming {
            let (stop, cur) = f(start, i);
            start = cur;
            done = stop;
            if stop {
                break;
            }
        }
        (done, start)
    }
    fn map_key<'a>(incoming: impl Iterator<Item = &'a Scalar>) -> (bool, Option<Scalar>) {
        fold_option(incoming, |o1, o2| {
            use Scalar::*;
            match (o1, *o2) {
                (Some(Str), _) | (Some(Float), _) | (_, Str) | (_, Float) => (true, Some(Str)),
                (_, _) => (false, Some(Int)),
            }
        })
    }
    fn map_val<'a>(incoming: impl Iterator<Item = &'a Scalar>) -> (bool, Option<Scalar>) {
        fold_option(incoming, |o1, o2| {
            use Scalar::*;
            match (o1, *o2) {
                (Some(Str), _) | (_, Str) => (true, Some(Str)),
                (Some(Float), _) | (_, Float) => (false, Some(Float)),
                (Some(Int), _) | (_, Int) => (false, Some(Int)),
            }
        })
    }

    #[derive(Clone, PartialEq, Eq, Debug)]
    pub(crate) enum Rule {
        Const(Scalar),
        Builtin(Function),
        MapKey,
        Val,
    }

    impl Propagator for Rule {
        type Item = Scalar;
        fn arity(&self) -> Option<usize> {
            use Rule::*;
            match self {
                Const(_) => Some(0),
                Builtin(b) => b.arity(),
                MapKey | Val => None,
            }
        }
        fn step(&self, incoming: &[Option<Scalar>]) -> (bool, Option<Scalar>) {
            use Rule::*;
            let set = incoming.iter().flat_map(|o| o.as_ref().into_iter());
            match self {
                Const(s) => (true, Some(*s)),
                Builtin(b) => b.step(incoming),
                MapKey => map_key(set),
                Val => map_val(set),
            }
        }
    }

    struct Node<P: Propagator> {
        rule: Option<P>,
        item: Option<P::Item>,
        deps: SmallVec<NodeIx>,
        done: bool,
        active: bool,
    }

    impl<P: Propagator + Debug> Debug for Node<P>
    where
        P::Item: Debug,
    {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "{:?},{:?}", self.rule, self.item)
        }
    }

    pub(crate) struct Network<P: Propagator> {
        graph: Graph<Node<P>, ()>,
        worklist: Vec<NodeIx>,
    }

    impl<P: Propagator + Debug> Debug for Network<P>
    where
        P::Item: Debug,
    {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "{:?}", petgraph::dot::Dot::new(&self.graph))
        }
    }

    impl<P: Propagator> Default for Network<P> {
        fn default() -> Network<P> {
            Network {
                graph: Default::default(),
                worklist: Default::default(),
            }
        }
    }

    impl<P: Propagator + Clone + Eq + Debug> Network<P>
    where
        P::Item: Clone + Eq + Debug,
    {
        pub(crate) fn solve(&mut self) {
            while let Some(id) = self.worklist.pop() {
                // We have to do some { ... }-ing to avoid taking a mut reference that lives for
                // the rest of the iteration.
                let (rule, deps) = {
                    // Pop off the worklist and then copy rule and deps out of the graph.
                    let Node {
                        rule, deps, active, ..
                    } = self
                        .graph
                        .node_weight_mut(id)
                        .expect("all nodes in worklist must be valid");
                    *active = false;
                    (rule.clone(), deps.clone())
                };

                if rule.is_none() {
                    continue;
                }

                // Convert SmallVec<NodeIx> => SmallVec<Option<P::Item>>
                let read_deps: SmallVec<_> = deps.iter().map(|d| self.read(*d).cloned()).collect();
                let (done_now, next) = rule.as_ref().unwrap().step(&read_deps[..]);
                {
                    let Node { item, done, .. } = self.graph.node_weight_mut(id).unwrap();
                    *done = done_now;
                    if &next != item {
                        *item = next;
                    } else {
                        // Don't re-evaluate dependencies if item did not change.
                        continue;
                    }
                }

                // We have a new value; add all nodes that depend on us to the worklist.
                let mut walker = self
                    .graph
                    .neighbors_directed(id, petgraph::Direction::Outgoing)
                    .detach();
                while let Some(neigh) = walker.next_node(&self.graph) {
                    let Node { active, done, .. } = self.graph.node_weight_mut(neigh).unwrap();
                    if *done || *active {
                        continue;
                    }
                    *active = true;
                    self.worklist.push(neigh)
                }
            }
        }
        pub(crate) fn read(&self, id: NodeIx) -> Option<&P::Item> {
            let Node { item, .. } = self
                .graph
                .node_weight(id)
                .expect("read must get a valid node index");
            item.as_ref()
        }
        pub(crate) fn add_rule(&mut self, rule: Option<P>, deps: &[NodeIx]) -> NodeIx {
            let res = self.graph.add_node(Node {
                rule,
                item: None,
                deps: deps.iter().cloned().collect(),
                done: false,
                active: true,
            });
            for d in deps.iter() {
                self.graph.add_edge(*d, res, ());
            }
            self.worklist.push(res);
            res
        }

        // TODO refactor tests so that they do not require this method.
        #[allow(unused)]
        pub(crate) fn update_rule(&mut self, id: NodeIx, new_rule: P) -> Result<()> {
            if let Some(Node {
                rule, deps, active, ..
            }) = self.graph.node_weight_mut(id)
            {
                if let Some(r) = rule {
                    if &new_rule != r {
                        return err!("attempt to replace rule {:?} with {:?}", r, new_rule);
                    }
                    if let Some(a) = new_rule.arity() {
                        if deps.len() != a {
                            return err!(
                                "attempt to assign node {:?} to rule {:?} with wrong number of dependencies ({} vs {})",
                                id, new_rule, deps.len(), a);
                        }
                    }
                }
                if !*active {
                    *active = true;
                    self.worklist.push(id);
                }
                *rule = Some(new_rule);
                Ok(())
            } else {
                return err!("invalid node id {:?}", id);
            }
        }
        pub(crate) fn add_dep(&mut self, id: NodeIx, dep: NodeIx) -> Result<()> {
            self.add_deps(id, once(dep))
        }
        pub(crate) fn add_deps(
            &mut self,
            id: NodeIx,
            new_deps: impl Iterator<Item = NodeIx>,
        ) -> Result<()> {
            let new_deps: SmallVec<_> = new_deps.collect();
            if let Some(Node {
                rule, deps, active, ..
            }) = self.graph.node_weight_mut(id)
            {
                if let Some(a) = rule.as_ref().and_then(Propagator::arity) {
                    return err!(
                        "Attempt to add dependencies to rule {:?} with arity {:?}",
                        rule,
                        a
                    );
                }
                deps.extend(new_deps.iter().cloned());
                if !*active {
                    *active = true;
                    self.worklist.push(id);
                }
            } else {
                return err!("invalid node id {:?}", id);
            }
            for d in new_deps.into_iter() {
                self.graph.add_edge(d, id, ());
            }
            Ok(())
        }
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub(crate) enum TVar<T = NodeIx> {
    Scalar(T),
    Iter(T),
    Map { key: T, val: T },
}

impl<T> TVar<T> {
    fn kind(&self) -> Kind {
        use TVar::*;
        match self {
            Scalar(_) => Kind::Scalar,
            Iter(_) => Kind::Iter,
            Map { .. } => Kind::Map,
        }
    }
    fn scalar(self) -> Result<T> {
        use TVar::*;
        match self {
            Scalar(ix) => Ok(ix),
            Iter(_) => err!("expected scalar, got iterator"),
            Map { .. } => err!("expected scalar, got map"),
        }
    }
    fn map(self) -> Result<(T, T)> {
        use TVar::*;
        match self {
            Scalar(_) => err!("expected map, got scalar"),
            Iter(_) => err!("expected map, got iterator"),
            Map { key, val } => Ok((key, val)),
        }
    }
    fn iterator(self) -> Result<T> {
        use TVar::*;
        match self {
            Scalar(_) => err!("expected iterator, got scalar"),
            Iter(ix) => Ok(ix),
            Map { .. } => err!("expected iterator, got map"),
        }
    }
}

pub(crate) fn get_types<'a>(
    cfg: &cfg::CFG<'a>,
    num_idents: usize,
) -> Result<HashMap<Ident, TVar<Option<Scalar>>>> {
    let mut cs = Constraints::new(num_idents);
    cs.build(cfg)
}

pub(crate) struct Constants {
    pub str_node: NodeIx,
    pub int_node: NodeIx,
    pub float_node: NodeIx,
    pub nil_node: NodeIx,
}

pub(crate) struct Constraints {
    network: prop::Network<prop::Rule>,
    ident_map: HashMap<Ident, TVar>,
    constants: Constants,
    canonical_ident: HashMap<Ident, NumTy>,
    uf: petgraph::unionfind::UnionFind<NumTy>,
    kind_map: HashMap<NumTy, Kind>,
    max_ident: NumTy,
}

use prop::Rule;

impl Constraints {
    fn new(n: usize) -> Constraints {
        let mut network = prop::Network::<Rule>::default();
        let nil_node = network.add_rule(None, &[]);
        let str_node = network.add_rule(Some(Rule::Const(Scalar::Str)), &[]);
        let int_node = network.add_rule(Some(Rule::Const(Scalar::Int)), &[]);
        let float_node = network.add_rule(Some(Rule::Const(Scalar::Float)), &[]);
        Constraints {
            network,
            ident_map: Default::default(),
            canonical_ident: Default::default(),
            kind_map: Default::default(),
            uf: petgraph::unionfind::UnionFind::new(n),
            constants: Constants {
                nil_node,
                str_node,
                int_node,
                float_node,
            },
            max_ident: 0,
        }
    }

    fn build<'a>(&mut self, cfg: &cfg::CFG<'a>) -> Result<HashMap<Ident, TVar<Option<Scalar>>>> {
        let nodes = cfg.raw_nodes();

        // First, build kinds by unification.
        for bb in nodes {
            for stmt in bb.weight.0.iter() {
                self.assign_kinds(stmt)?;
            }
        }

        // Then, build propgator network for type inference.
        for bb in nodes {
            for stmt in bb.weight.0.iter() {
                self.gen_stmt_constraints(stmt)?;
            }
        }
        self.network.solve();
        Ok(self
            .ident_map
            .iter()
            .map(|(k, v)| {
                use TVar::*;
                (
                    *k,
                    match v {
                        Iter(i) => Iter(self.network.read(*i).cloned()),
                        Scalar(i) => Scalar(self.network.read(*i).cloned()),
                        Map { key, val } => Map {
                            key: self.network.read(*key).cloned(),
                            val: self.network.read(*val).cloned(),
                        },
                    },
                )
            })
            .collect())
    }

    fn merge_val<'a>(&mut self, v: &PrimVal<'a>, id: Ident) -> Result<()> {
        use PrimVal::*;
        match v {
            ILit(_) | FLit(_) | StrLit(_) => self.set_kind(id, Kind::Scalar),
            Var(id2) => self.merge_idents(id, *id2),
        }
    }

    fn assert_val_kind<'a>(&mut self, v: &PrimVal<'a>, k: Kind) -> Result<()> {
        use PrimVal::*;
        match v {
            ILit(_) | FLit(_) | StrLit(_) => {
                if let Kind::Scalar = k {
                    Ok(())
                } else {
                    err!("Scalar literal used in non-scalar context {:?}", k)
                }
            }
            Var(id) => self.set_kind(*id, k),
        }
    }

    fn merge_expr<'a>(&mut self, expr: &PrimExpr<'a>, with: Either<Kind, Ident>) -> Result<()> {
        use {Either::*, PrimExpr::*};
        match (expr, &with) {
            (Val(pv), Left(k)) => self.assert_val_kind(pv, *k),
            (Val(pv), Right(id)) => self.merge_val(pv, *id),
            (Phi(preds), e) => {
                for (_, id) in preds.iter() {
                    match e {
                        Left(k) => self.set_kind(*id, *k)?,
                        Right(inp) => self.merge_idents(*inp, *id)?,
                    };
                }
                Ok(())
            }
            (CallBuiltin(b, args), e) => {
                let arg_ks = b.signature();
                let ret_k = Kind::Scalar;
                if args.len() > arg_ks.len() {
                    return err!(
                        "Calling builtin {} with {} args; expected {}",
                        b,
                        args.len(),
                        arg_ks.len()
                    );
                }
                // N.B: Awk lets you pass fewer variables than declared in the function.
                for (a, k) in args.iter().zip(arg_ks.iter()) {
                    self.assert_val_kind(a, *k)?;
                }
                match e {
                    Left(k) => {
                        if *k == ret_k {
                            Ok(())
                        } else {
                            err!("Result of builtin {} used in context of kind {:?}, expected kind {:?}",
                             b, k, ret_k)
                        }
                    }
                    Right(id) => self.set_kind(*id, ret_k),
                }
            }
            (Index(map, ix), e) => {
                self.assert_val_kind(map, Kind::Map)?;
                self.assert_val_kind(ix, Kind::Scalar)?;
                match e {
                    Left(Kind::Scalar) => Ok(()),
                    Left(k) => err!("result of map lookup used in {:?} context", k),
                    Right(id) => self.set_kind(*id, Kind::Scalar),
                }
            }
            (IterBegin(map), e) => {
                self.assert_val_kind(map, Kind::Map)?;
                match e {
                    Left(Kind::Iter) => Ok(()),
                    Left(k) => err!(
                        "[internal error] taking iterator in non-map context {:?}",
                        k
                    ),
                    Right(id) => self.set_kind(*id, Kind::Iter),
                }
            }
            (HasNext(iter), e) => {
                self.assert_val_kind(iter, Kind::Iter)?;
                match e {
                    Left(Kind::Scalar) => Ok(()),
                    Left(k) => err!(
                        "[internal error] checking next in non-scalar context {:?}",
                        k
                    ),
                    Right(id) => self.set_kind(*id, Kind::Scalar),
                }
            }
            (Next(iter), e) => {
                self.assert_val_kind(iter, Kind::Iter)?;
                match e {
                    Left(Kind::Iter) => Ok(()),
                    Left(k) => err!(
                        "[internal error] getting next from interator in non-iterator context {:?}",
                        k
                    ),
                    Right(id) => self.set_kind(*id, Kind::Scalar),
                }
            }
            (LoadBuiltin(b), Left(k)) => {
                let b_k = b.ty().kind();
                if b_k == *k {
                    Ok(())
                } else {
                    err!(
                        "using builtin {} with kind {:?} in context of {:?}",
                        b,
                        b_k,
                        k
                    )
                }
            }
            (LoadBuiltin(b), Right(id)) => self.set_kind(*id, b.ty().kind()),
        }
    }

    fn assign_kinds<'a>(&mut self, stmt: &PrimStmt<'a>) -> Result<()> {
        use Either::*;
        use PrimStmt::*;
        match stmt {
            AsgnIndex(map, k, v) => {
                self.set_kind(*map, Kind::Map)?;
                self.assert_val_kind(k, Kind::Scalar)?;
                self.merge_expr(v, Left(Kind::Scalar))
            }
            AsgnVar(id, v) => self.merge_expr(v, Right(*id)),
            SetBuiltin(var, exp) => self.merge_expr(exp, Left(var.ty().kind())),
        }
    }

    fn get_canonical(&mut self, id: Ident) -> NumTy {
        match self.canonical_ident.entry(id) {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(v) => {
                let res = self.max_ident;
                v.insert(res);
                self.max_ident += 1;
                res
            }
        }
    }

    fn set_kind(&mut self, id: Ident, k: Kind) -> Result<()> {
        let c = self.get_canonical(id);
        let rep = self.uf.find_mut(c);
        match self.kind_map.entry(rep) {
            Entry::Occupied(o) => {
                let cur = o.get();
                if *cur != k {
                    return err!(
                        "Identifier {:?} used in both {:?} and {:?} contexts",
                        id,
                        cur,
                        k
                    );
                }
            }
            Entry::Vacant(v) => {
                v.insert(k);
            }
        };
        Ok(())
    }

    fn merge_idents(&mut self, id1: Ident, id2: Ident) -> Result<()> {
        // Get canonical identifiers for id1 and id2, then map them to their representative in
        // the disjoint-set datastructure.
        let i1 = self.get_canonical(id1);
        let i2 = self.get_canonical(id2);
        let c1 = self.uf.find_mut(i1);
        let c2 = self.uf.find_mut(i2);
        if self.uf.union(i1, i2) {
            // We changed the data-structure. First, get the current mappings of the previous
            // two canonical representatives, then see what the "merged kind" is, and insert it
            // into the new map if necessary.
            let k1 = self.kind_map.get(&c1).clone();
            let k2 = self.kind_map.get(&c2).clone();
            let merged = match (k1, k2) {
                (Some(k), None) | (None, Some(k)) => *k,
                (None, None) => return Ok(()),
                (Some(k1), Some(k2)) => {
                    if k1 == k2 {
                        *k1
                    } else {
                        return err!(
                            "Identifiers {:?} and {:?} have incompatible kinds ({:?} and {:?})",
                            id1,
                            id2,
                            k1,
                            k2
                        );
                    }
                }
            };
            let c = self.uf.find_mut(i1);
            self.kind_map.insert(c, merged);
        }
        Ok(())
    }

    fn get_kind(&mut self, id: Ident) -> Option<Kind> {
        let c = self.get_canonical(id);
        self.kind_map.get(&self.uf.find_mut(c)).cloned()
    }

    fn get_expr_constraints<'a>(&mut self, expr: &PrimExpr<'a>) -> Result<TVar> {
        use PrimExpr::*;
        match expr {
            Val(pv) => self.get_val(pv),
            Phi(preds) => {
                assert!(preds.len() > 0);
                let k = self.get_kind(preds[0].1).or(Some(Kind::Scalar)).unwrap();
                match k {
                    Kind::Scalar => {
                        let mut deps = SmallVec::new();
                        for (_, id) in preds.iter() {
                            deps.push(self.get_scalar(*id)?);
                        }
                        Ok(TVar::Scalar(
                            self.network.add_rule(Some(Rule::Val), &deps[..]),
                        ))
                    }
                    Kind::Map => {
                        let mut ks = SmallVec::new();
                        let mut vs = SmallVec::new();
                        for (_, id) in preds.iter() {
                            let (k, v) = self.get_map(*id)?;
                            ks.push(k);
                            vs.push(v);
                        }
                        let key = self.network.add_rule(Some(Rule::MapKey), &ks[..]);
                        let val = self.network.add_rule(Some(Rule::Val), &vs[..]);
                        Ok(TVar::Map { key, val })
                    }
                    Kind::Iter => {
                        // Unlikely that this will be an issue (deps should all have the same
                        // type), but we do need something here.
                        let mut deps = SmallVec::new();
                        for (_, id) in preds.iter() {
                            deps.push(self.get_scalar(*id)?);
                        }
                        Ok(TVar::Iter(
                            self.network.add_rule(Some(Rule::Val), &deps[..]),
                        ))
                    }
                }
            }
            CallBuiltin(b, args) => {
                let mut args = args.clone();
                let arg_ks = b.signature();
                // optimize for the all-scalar case; if maps are involved we will just grow the
                // vector.
                let mut deps = SmallVec::with_capacity(args.len());
                for k in arg_ks.iter() {
                    match args.pop() {
                        Some(PrimVal::StrLit(_)) => deps.push(self.constants.str_node),
                        Some(PrimVal::ILit(_)) => deps.push(self.constants.int_node),
                        Some(PrimVal::FLit(_)) => deps.push(self.constants.float_node),
                        Some(PrimVal::Var(id)) => match self.get_var(id)? {
                            TVar::Scalar(v) | TVar::Iter(v) => deps.push(v),
                            TVar::Map { key, val } => {
                                deps.push(key);
                                deps.push(val)
                            }
                        },
                        None => match k {
                            Kind::Scalar | Kind::Iter => deps.push(self.constants.nil_node),
                            Kind::Map => {
                                deps.push(self.constants.nil_node);
                                deps.push(self.constants.nil_node);
                            }
                        },
                    }
                }
                b.feedback(&mut self.network, &self.constants, &deps[..])?;
                Ok(TVar::Scalar(
                    self.network.add_rule(Some(Rule::Builtin(*b)), &deps[..]),
                ))
            }
            Index(map, ix) => {
                let (key, val) = self.get_val(map)?.map()?;
                let ix_node = self.get_scalar_val(ix)?;
                // We want to add ix_node as a dependency of key
                self.network.add_deps(key, once(ix_node))?;
                // Then we want to yield val, the result of this expression.
                Ok(TVar::Scalar(val))
            }
            IterBegin(map) => {
                let (key, _) = self.get_val(map)?.map()?;
                Ok(TVar::Iter(key))
            }
            HasNext(iter) => {
                let _it = self.get_val(iter)?.iterator()?;
                Ok(TVar::Scalar(self.constants.int_node))
            }
            Next(iter) => {
                let it = self.get_val(iter)?.iterator()?;
                Ok(TVar::Scalar(it))
            }
            LoadBuiltin(b) => Ok(match b.ty() {
                TVar::Scalar(s) => TVar::Scalar(self.const_node(s)),
                TVar::Iter(s) => TVar::Iter(self.const_node(s)),
                TVar::Map { key, val } => TVar::Map {
                    key: self.const_node(key),
                    val: self.const_node(val),
                },
            }),
        }
    }

    fn const_node(&self, scalar: Scalar) -> NodeIx {
        use Scalar::*;
        match scalar {
            Int => self.constants.int_node,
            Float => self.constants.float_node,
            Str => self.constants.str_node,
        }
    }

    fn gen_stmt_constraints<'a>(&mut self, stmt: &PrimStmt<'a>) -> Result<()> {
        use PrimStmt::*;
        match stmt {
            AsgnIndex(map_ident, key, val) => {
                let (k, v) = self.get_map(*map_ident)?;
                let k_node = self.get_scalar_val(key)?;
                self.network.add_dep(k, k_node)?;
                let v_node = self.get_expr_constraints(val)?.scalar()?;
                self.network.add_dep(v, v_node)
            }
            AsgnVar(ident, exp) => {
                let ident_v = self.get_var(*ident)?;
                let exp_v = self.get_expr_constraints(exp)?;
                match (ident_v, exp_v) {
                    (TVar::Iter(i1), TVar::Iter(i2)) | (TVar::Scalar(i1), TVar::Scalar(i2)) => {
                        self.network.add_dep(i1, i2)
                    }
                    (TVar::Map { key: k1, val: v1 }, TVar::Map { key: k2, val: v2 }) => {
                        // These two get bidirectional constraints, as maps do not get implicit
                        // conversions.
                        self.network.add_dep(k1, k2)?;
                        self.network.add_dep(k2, k1)?;
                        self.network.add_dep(v1, v2)?;
                        self.network.add_dep(v2, v1)
                    }
                    (k1, k2) => err!(
                        "assigning variables of mismatched kinds: {:?} and {:?}",
                        k1,
                        k2
                    ),
                }
            }
            // Builtins have fixed types, no constraint generation necessary.
            SetBuiltin(_, _) => Ok(()),
        }
    }
    pub(crate) fn get_iter(&mut self, ident: Ident) -> Result<NodeIx /* item */> {
        match self.ident_map.entry(ident) {
            Entry::Occupied(o) => match o.get() {
                TVar::Map { key: _, val: _ } => err!(
                    "Identifier {:?} is used in both map and iterator context",
                    ident
                ),
                TVar::Iter(n) => Ok(*n),
                TVar::Scalar(_) => err!(
                    "Identifier {:?} is used in both scalar and iterator context",
                    ident
                ),
            },
            Entry::Vacant(v) => {
                let item = self.network.add_rule(Some(Rule::Val), &[]);
                v.insert(TVar::Iter(item));
                Ok(item)
            }
        }
    }

    pub(crate) fn get_map(&mut self, ident: Ident) -> Result<(NodeIx /* key */, NodeIx /* val */)> {
        match self.ident_map.entry(ident) {
            Entry::Occupied(o) => match o.get() {
                TVar::Map { key, val } => Ok((*key, *val)),
                TVar::Iter(_) => err!(
                    "found iterator in map context (this indicates a bug in the implementation)"
                ),
                TVar::Scalar(_) => err!(
                    "Identifier {:?} is used in both scalar and map context",
                    ident
                ),
            },
            Entry::Vacant(v) => {
                let key = self.network.add_rule(Some(Rule::MapKey), &[]);
                let val = self.network.add_rule(Some(Rule::Val), &[]);
                v.insert(TVar::Map { key, val });
                Ok((key, val))
            }
        }
    }

    pub(crate) fn get_scalar(&mut self, ident: Ident) -> Result<NodeIx> {
        match self.ident_map.entry(ident.clone()) {
            Entry::Occupied(o) => match o.get() {
                TVar::Map { key: _, val: _ } => {
                    err!("identfier {:?} is used elsewhere as a map (insert)", ident)
                }
                TVar::Iter(_) => err!(
                    "found iterator in scalar context (this indicates a bug in the implementation)"
                ),
                TVar::Scalar(n) => Ok(*n),
            },
            Entry::Vacant(v) => {
                let n = self.network.add_rule(Some(Rule::Val), &[]);
                v.insert(TVar::Scalar(n));
                Ok(n)
            }
        }
    }

    fn get_scalar_val<'a>(&mut self, v: &PrimVal<'a>) -> Result<NodeIx> {
        use PrimVal::*;
        match v {
            ILit(_) => Ok(self.constants.int_node),
            FLit(_) => Ok(self.constants.float_node),
            StrLit(_) => Ok(self.constants.str_node),
            Var(id) => self.get_scalar(*id),
        }
    }

    fn get_var(&mut self, id: Ident) -> Result<TVar> {
        Ok(match self.get_kind(id) {
            Some(Kind::Scalar) | None => TVar::Scalar(self.get_scalar(id)?),
            Some(Kind::Map) => {
                let (key, val) = self.get_map(id)?;
                TVar::Map { key, val }
            }
            Some(Kind::Iter) => TVar::Iter(self.get_iter(id)?),
        })
    }

    fn get_val<'a>(&mut self, v: &PrimVal<'a>) -> Result<TVar> {
        use PrimVal::*;
        match v {
            ILit(_) => Ok(TVar::Scalar(self.constants.int_node)),
            FLit(_) => Ok(TVar::Scalar(self.constants.float_node)),
            StrLit(_) => Ok(TVar::Scalar(self.constants.str_node)),
            Var(id) => self.get_var(*id),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ast::Binop;
    use crate::builtins::Function;

    #[test]
    fn plus_key_int() {
        let mut n = prop::Network::default();
        use {Binop::*, Scalar::*};
        let i1 = n.add_rule(Some(Rule::Const(Int)), &[]);
        let i2 = n.add_rule(None, &[]);
        let addi12 = n.add_rule(Some(Rule::Builtin(Function::Binop(Plus))), &[i1, i2]);
        assert!(n.update_rule(i2, Rule::MapKey).is_ok());
        assert!(n.add_deps(i2, once(addi12)).is_ok());
        n.solve();
        assert_eq!(n.read(i1), Some(&Int));
        assert_eq!(n.read(i2), Some(&Int));
        assert_eq!(n.read(addi12), Some(&Int));
    }

    #[test]
    fn plus_key_float() {
        let mut n = prop::Network::default();
        use {Binop::*, Scalar::*};
        let i1 = n.add_rule(Some(Rule::Const(Int)), &[]);
        let f1 = n.add_rule(Some(Rule::MapKey), &[]);
        let add12 = n.add_rule(Some(Rule::Builtin(Function::Binop(Plus))), &[i1, f1]);
        let f2 = n.add_rule(Some(Rule::Const(Float)), &[]);
        assert!(n.add_deps(f1, vec![add12, f2].into_iter()).is_ok());
        n.solve();
        assert_eq!(n.read(i1), Some(&Int));
        assert_eq!(n.read(f1), Some(&Str));
        assert_eq!(n.read(f2), Some(&Float));
        assert_eq!(n.read(add12), Some(&Float));
    }
}
