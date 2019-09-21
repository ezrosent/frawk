//! Algorithms and types pertaining to type deduction and converion.
//!
//! TODO: update this with more documentation when the algorithms are more fully baked.
use crate::ast;
use crate::cfg::{self, Ident, PrimExpr, PrimStmt, PrimVal};
use crate::common::{Either, Graph, NodeIx, NumTy, Result};
use hashbrown::{hash_map::Entry, HashMap};
use smallvec::smallvec;

type SmallVec<T> = smallvec::SmallVec<[T; 2]>;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum Scalar {
    Str,
    Int,
    Float,
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

    fn inputs(&self, incoming: &[Option<Self::Item>]) -> SmallVec<Option<Self::Item>>;
}

mod prop {
    use super::{smallvec, Graph, NodeIx, Propagator, Result, Scalar, SmallVec};
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
        ArithUnop,
        ArithOp,
        CompareOp,
        Div,
        MapKey,
        Val,
    }

    impl Propagator for Rule {
        type Item = Scalar;
        fn arity(&self) -> Option<usize> {
            use Rule::*;
            match self {
                Const(_) => Some(0),
                ArithUnop => Some(1),
                Div | CompareOp | ArithOp => Some(2),
                MapKey | Val => None,
            }
        }
        fn step(&self, incoming: &[Option<Scalar>]) -> (bool, Option<Scalar>) {
            use Rule::*;
            use Scalar::*;
            let set = incoming.iter().flat_map(|o| o.as_ref().into_iter());
            match self {
                Const(s) => (true, Some(*s)),
                CompareOp => (true, Some(Scalar::Int)),
                ArithUnop => match &incoming[0] {
                    Some(Str) | Some(Float) => (true, Some(Float)),
                    x => (false, *x),
                },
                Div => (true, Some(Float)),
                ArithOp => match (&incoming[0], &incoming[1]) {
                    (Some(Str), _) | (_, Some(Str)) | (Some(Float), _) | (_, Some(Float)) => {
                        (true, Some(Float))
                    }
                    (_, _) => (false, Some(Int)),
                },
                MapKey => map_key(set),
                Val => map_val(set),
            }
        }
        fn inputs(&self, incoming: &[Option<Scalar>]) -> SmallVec<Option<Scalar>> {
            use Rule::*;
            use Scalar::*;
            match self {
                Const(_) => Default::default(),
                Div => smallvec![Some(Float), Some(Float)],
                ArithUnop => match &incoming[0] {
                    Some(Str) | Some(Float) => smallvec![Some(Float)],
                    x => smallvec![*x],
                },
                CompareOp => {
                    let max = match (incoming[0], incoming[1]) {
                        (Some(Str), _) | (_, Some(Str)) => Some(Str),
                        (Some(Float), _) | (_, Some(Float)) => Some(Float),
                        (Some(Int), _) | (_, Some(Int)) => Some(Int),
                        (None, None) => None,
                    };
                    smallvec![max; 2]
                }
                ArithOp => match (incoming[0], incoming[1]) {
                    (Some(Str), _) | (_, Some(Str)) | (Some(Float), _) | (_, Some(Float)) => {
                        smallvec![Some(Float); 2]
                    }
                    (_, _) => smallvec![Some(Int); 2],
                },
                MapKey => {
                    let set = incoming.iter().flat_map(|o| o.as_ref().into_iter());
                    let len = incoming.len();
                    smallvec![map_key(set).1; len]
                }
                Val => {
                    let set = incoming.iter().flat_map(|o| o.as_ref().into_iter());
                    let len = incoming.len();
                    smallvec![map_val(set).1; len]
                }
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

    pub(crate) struct Network<P: Propagator> {
        graph: Graph<Node<P>, ()>,
        worklist: Vec<NodeIx>,
    }

    impl<P: Propagator> Default for Network<P> {
        fn default() -> Network<P> {
            Network {
                graph: Default::default(),
                worklist: Default::default(),
            }
        }
    }

    impl<P: Propagator + Clone + Eq + std::fmt::Debug> Network<P>
    where
        P::Item: Clone + Eq,
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
        pub(crate) fn read_inputs(&self, id: NodeIx) -> SmallVec<Option<P::Item>> {
            let Node { deps, rule, .. } = self
                .graph
                .node_weight(id)
                .expect("read must get a valid node index");
            match rule {
                Some(r) => {
                    let inputs: SmallVec<_> = deps
                        .iter()
                        // for each dependency
                        .map(|n| {
                            // get the Option<Node>
                            self.graph
                                .node_weight(*n)
                                // unwrap it
                                .expect("node deps must be valid nodes")
                                // copy out the current `item`
                                .item
                                .clone()
                        })
                        .collect();
                    r.inputs(&inputs[..])
                }
                None => smallvec![None; deps.len()],
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
                    if !*active {
                        *active = true;
                        self.worklist.push(id);
                    }
                    *rule = Some(new_rule);
                }
                Ok(())
            } else {
                return err!("invalid node id {:?}", id);
            }
        }
        pub(crate) fn add_deps(
            &mut self,
            id: NodeIx,
            new_deps: impl Iterator<Item = NodeIx>,
        ) -> Result<()> {
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
                deps.extend(new_deps);
                if !*active {
                    *active = true;
                    self.worklist.push(id);
                }
                Ok(())
            } else {
                return err!("invalid node id {:?}", id);
            }
        }
    }
}

#[derive(PartialEq, Eq, Debug)]
pub(crate) enum TVar<T = NodeIx> {
    Scalar(T),
    Iter(T),
    Map { key: T, val: T },
}

impl TVar {
    fn scalar(self) -> Result<NodeIx> {
        use TVar::*;
        match self {
            Scalar(ix) => Ok(ix),
            Iter(_) => err!("expected scalar, got iterator"),
            Map { .. } => err!("expected scalar, got map"),
        }
    }
    fn map(self) -> Result<(NodeIx, NodeIx)> {
        use TVar::*;
        match self {
            Scalar(_) => err!("expected map, got scalar"),
            Iter(_) => err!("expected map, got iterator"),
            Map { key, val } => Ok((key, val)),
        }
    }
    fn iterator(self) -> Result<NodeIx> {
        use TVar::*;
        match self {
            Scalar(_) => err!("expected iterator, got scalar"),
            Iter(ix) => Ok(ix),
            Map { .. } => err!("expected iterator, got map"),
        }
    }
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
enum Kind {
    Scalar,
    Iter,
    Map,
}

pub(crate) fn get_types<'a>(
    cfg: &cfg::CFG<'a>,
    num_idents: usize,
) -> Result<HashMap<Ident, TVar<Option<Scalar>>>> {
    let mut cs = Constraints::new(num_idents);
    cs.build(cfg)
}

pub(crate) struct Constraints {
    network: prop::Network<prop::Rule>,
    ident_map: HashMap<Ident, TVar>,
    str_node: NodeIx,
    int_node: NodeIx,
    float_node: NodeIx,

    canonical_ident: HashMap<Ident, NumTy>,
    uf: petgraph::unionfind::UnionFind<NumTy>,
    kind_map: HashMap<NumTy, Kind>,
    max_ident: NumTy,
}

use prop::Rule;

impl Constraints {
    fn new(n: usize) -> Constraints {
        let mut network = prop::Network::<Rule>::default();
        let str_node = network.add_rule(Some(Rule::Const(Scalar::Str)), &[]);
        let int_node = network.add_rule(Some(Rule::Const(Scalar::Int)), &[]);
        let float_node = network.add_rule(Some(Rule::Const(Scalar::Float)), &[]);
        Constraints {
            network,
            ident_map: Default::default(),
            canonical_ident: Default::default(),
            kind_map: Default::default(),
            uf: petgraph::unionfind::UnionFind::new(n),
            max_ident: 0,
            str_node,
            int_node,
            float_node,
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
        use Either::*;
        use PrimExpr::*;
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
            (Binop(_, o1, o2), e) => {
                self.assert_val_kind(o1, Kind::Scalar)?;
                self.assert_val_kind(o2, Kind::Scalar)?;
                match e {
                    Left(Kind::Scalar) => Ok(()),
                    Left(k) => err!("result of scalar binary operation used in {:?} context", k),
                    Right(id) => self.set_kind(*id, Kind::Scalar),
                }
            }
            (Unop(_, o), e) => {
                self.assert_val_kind(o, Kind::Scalar)?;
                match e {
                    Left(Kind::Scalar) => Ok(()),
                    Left(k) => err!("result of scalar unary operation used in {:?} context", k),
                    Right(id) => self.set_kind(*id, Kind::Scalar),
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
        }
    }

    fn assign_kinds<'a>(&mut self, stmt: &PrimStmt<'a>) -> Result<()> {
        use Either::*;
        use PrimStmt::*;
        match stmt {
            Print(vs, out) => {
                for v in vs.iter().chain(out.iter()) {
                    self.assert_val_kind(v, Kind::Scalar)?;
                }
                Ok(())
            }
            AsgnIndex(map, k, v) => {
                self.set_kind(*map, Kind::Map)?;
                self.assert_val_kind(k, Kind::Scalar)?;
                self.merge_expr(v, Left(Kind::Scalar))
            }
            AsgnVar(id, v) => self.merge_expr(v, Right(*id)),
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
            Unop(op, o) => {
                use ast::Unop::*;
                Ok(TVar::Scalar(match op {
                    // TODO: give these two inputs
                    Column => self.str_node,
                    Not => self.int_node,
                    Neg | Pos => {
                        let inp = self.get_val(o)?.scalar()?;
                        self.network.add_rule(Some(Rule::ArithUnop), &[inp])
                    }
                }))
            }
            Binop(op, o1, o2) => {
                use ast::Binop::*;
                let i1 = self.get_val(o1)?.scalar()?;
                let i2 = self.get_val(o2)?.scalar()?;
                Ok(TVar::Scalar(match op {
                    Plus | Minus | Mult | Mod => {
                        self.network.add_rule(Some(Rule::ArithOp), &[i1, i2])
                    }
                    Div => self.network.add_rule(Some(Rule::Div), &[i1, i2]),
                    // TODO: give these inputs
                    Concat => self.str_node,
                    Match => self.int_node,
                    LT | GT | LTE | GTE | EQ => self.int_node,
                }))
            }
            Index(map, ix) => {
                let (key, val) = self.get_val(map)?.map()?;
                let ix_node = self.get_scalar_val(ix)?;
                // We want to add ix_node as a dependency of key
                self.network.add_deps(key, Some(ix_node).into_iter())?;
                // Then we want to yield val, the result of this expression.
                Ok(TVar::Scalar(val))
            }
            IterBegin(map) => {
                let (key, _) = self.get_val(map)?.map()?;
                Ok(TVar::Iter(key))
            }
            HasNext(iter) => {
                let _it = self.get_val(iter)?.iterator()?;
                Ok(TVar::Scalar(self.int_node))
            }
            Next(iter) => {
                let it = self.get_val(iter)?.iterator()?;
                Ok(TVar::Scalar(it))
            }
        }
    }
    fn gen_stmt_constraints<'a>(&mut self, stmt: &PrimStmt<'a>) -> Result<()> {
        use PrimStmt::*;
        match stmt {
            Print(_, _) => {
                // TODO: anything to do here?
                Ok(())
            }
            AsgnIndex(map_ident, key, val) => {
                let (k, v) = self.get_map(*map_ident)?;
                self.merge_rule(k, Rule::MapKey, Some(key).into_iter())?;
                let v_node = self.get_expr_constraints(val)?.scalar()?;
                self.merge_rule_id(k, Rule::Val, Some(v_node).into_iter())
            }
            AsgnVar(ident, exp) => {
                let ident_v = self.get_var(*ident)?;
                let exp_v = self.get_expr_constraints(exp)?;
                match (ident_v, exp_v) {
                    (TVar::Iter(i1), TVar::Iter(i2)) | (TVar::Scalar(i1), TVar::Scalar(i2)) => {
                        self.merge_rule_id(i1, Rule::Val, Some(i2).into_iter())
                    }
                    (TVar::Map { key: k1, val: v1 }, TVar::Map { key: k2, val: v2 }) => {
                        // These two get bidirectional constraints, as maps do not get implicit
                        // conversions.
                        self.merge_rule_id(k1, Rule::MapKey, Some(k2).into_iter())?;
                        self.merge_rule_id(k2, Rule::MapKey, Some(k1).into_iter())?;
                        self.merge_rule_id(v1, Rule::Val, Some(v2).into_iter())?;
                        self.merge_rule_id(v2, Rule::Val, Some(v1).into_iter())
                    }
                    (k1, k2) => err!(
                        "assigning variables of mismatched kinds: {:?} and {:?}",
                        k1,
                        k2
                    ),
                }
            }
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
                let item = self.network.add_rule(None, &[]);
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
                // TODO: can we initialize this to Val and then get rid of the merges?
                let key = self.network.add_rule(None, &[]);
                let val = self.network.add_rule(None, &[]);
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
                TVar::Scalar(n) => {
                    // TODO: what was this doing?
                    // self.network
                    //     .update(*n, TypeRule::Placeholder, None.into_iter())?;
                    Ok(*n)
                }
            },
            Entry::Vacant(v) => {
                let n = self.network.add_rule(None, &[]);
                v.insert(TVar::Scalar(n));
                Ok(n)
            }
        }
    }

    fn get_scalar_val<'a>(&mut self, v: &PrimVal<'a>) -> Result<NodeIx> {
        use PrimVal::*;
        match v {
            ILit(_) => Ok(self.int_node),
            FLit(_) => Ok(self.float_node),
            StrLit(_) => Ok(self.str_node),
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
            ILit(_) => Ok(TVar::Scalar(self.int_node)),
            FLit(_) => Ok(TVar::Scalar(self.float_node)),
            StrLit(_) => Ok(TVar::Scalar(self.str_node)),
            Var(id) => self.get_var(*id),
        }
    }

    fn merge_rule_id(
        &mut self,
        id: NodeIx,
        rule: Rule,
        deps: impl Iterator<Item = NodeIx>,
    ) -> Result<()> {
        self.network.update_rule(id, rule)?;
        self.network.add_deps(id, deps)
    }
    fn merge_rule<'b, 'a: 'b>(
        &mut self,
        id: NodeIx,
        rule: Rule,
        deps: impl Iterator<Item = &'b PrimVal<'a>>,
    ) -> Result<()> {
        let mut dep_nodes = SmallVec::<NodeIx>::new();
        // Build a vector of NodeIxs from a vector of PrimVals. For literals, we map those to
        // our special-cased literal nodes. For Vars, first check that the var is not currently
        // a Map (N.B. no scalar expressions contain maps, at least for now), then either grab
        // the existing node or insert a scalar Placeholder at that identifier.
        //
        //
        // TODO: fix these rules for when functions are supported.
        for i in deps {
            dep_nodes.push(self.get_scalar_val(i)?);
        }
        self.merge_rule_id(id, rule, dep_nodes.into_iter())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn plus_key_int() {
        let mut n = prop::Network::default();
        use Rule::*;
        use Scalar::*;
        let i1 = n.add_rule(Some(Const(Int)), &[]);
        let i2 = n.add_rule(None, &[]);
        let addi12 = n.add_rule(Some(ArithOp), &[i1, i2]);
        assert!(n.update_rule(i2, MapKey).is_ok());
        assert!(n.add_deps(i2, Some(addi12).into_iter()).is_ok());
        n.solve();
        assert_eq!(n.read(i1), Some(&Int));
        assert_eq!(n.read(i2), Some(&Int));
        assert_eq!(n.read(addi12), Some(&Int));
    }

    // #[test]
    // fn plus_key_float() {
    //     let mut n = Network::default();
    //     use Scalar::*;
    //     use TypeRule::*;
    //     let i1 = n.insert(Const(Int), None.into_iter());
    //     let f1 = n.insert(Placeholder, None.into_iter());
    //     let add12 = n.insert(ArithOp(None), vec![i1, f1].into_iter());
    //     let f2 = n.insert(Const(Float), None.into_iter());
    //     n.update(f1, MapKey(None), vec![add12, f2].into_iter());
    //     n.solve();
    //     assert_eq!(n.read(i1), &Some(Int));
    //     assert_eq!(n.read(f1), &Some(Str));
    //     assert_eq!(n.read(f2), &Some(Float));
    //     assert_eq!(n.read(add12), &Some(Float));
    // }
}
