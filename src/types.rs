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
pub(crate) trait Propagator: Default + Clone {
    type Item;
    /// While rules (instances of Self) themselves will propgate information in a monotone fashion,
    /// we may run into cases where we want to change from one rule to another. This too has to be
    /// "monotone" in the same way.
    ///
    /// Using TypeRule as an example, we cannot change a variable from an Int to a Str. But we might
    /// be able to refine a Placeholder into either one of those. Updating to an operation with
    /// strictly less information (e.g. ArithOp(..) => Placeholder) has no effect; in this way,
    /// try_replace behaves like a lub operation.
    ///
    /// N.B In a more general propagator implementation, nodes in a network could have different;
    /// and then rules could just be nodes whose values were functions. It may be worth implementing
    /// such an abstraction depending on how this implementation grows in complexity.
    fn try_replace(&mut self, other: Self) -> Option<Self>;
    /// Ingest new information from `incoming`, produce a new output and indicate if the output has
    /// been saturated (i.e. regardless of what new inputs it will produce, the output will not
    /// change).
    fn step(&mut self, incoming: impl Iterator<Item = Self::Item>)
        -> (bool /* done */, Self::Item);
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum TypeRule {
    Placeholder,
    // For literals, and also operators like '/' (which will always coerce to float) or bitwise
    // operations (which always coerce to integers).
    Const(Scalar),
    // For things like +,% and *
    ArithOp(Option<Scalar>),
    // CompareOp(Option<Scalar>),
    MapKey(Option<Scalar>),
    MapVal(Option<Scalar>),
}

impl Default for TypeRule {
    fn default() -> TypeRule {
        TypeRule::Placeholder
    }
}

fn apply_binary_rule<T>(
    mut start: T,
    incoming: impl Iterator<Item = T>,
    f: impl Fn(&T, &T) -> (bool, T),
) -> (bool, T) {
    let mut done = false;
    for i in incoming {
        let (stop, cur) = f(&start, &i);
        start = cur;
        done = stop;
        if stop {
            break;
        }
    }
    (done, start)
}

impl Propagator for TypeRule {
    type Item = Option<Scalar>;
    fn try_replace(&mut self, other: TypeRule) -> Option<TypeRule> {
        use TypeRule::*;
        if let Placeholder = self {
            *self = other;
            return None;
        }
        match (self, other) {
            (Placeholder, _) => unreachable!(),
            (ArithOp(Some(_)), ArithOp(None))
            // | (CompareOp(Some(_)), CompareOp(None))
            | (MapKey(Some(_)), MapKey(None))
            | (MapVal(Some(_)), MapVal(None))
            | (_, Placeholder) => None,
            (_, other) => Some(other),
        }
    }
    fn step(
        &mut self,
        incoming: impl Iterator<Item = Option<Scalar>>,
    ) -> (bool /* done */, Option<Scalar>) {
        fn op_helper(o1: &Option<Scalar>, o2: &Option<Scalar>) -> (bool, Option<Scalar>) {
            use Scalar::*;
            match (o1, o2) {
                // No information to propagate
                (None, None) => (false, None),
                (Some(Str), _) | (_, Some(Str)) | (Some(Float), _) | (_, Some(Float)) => {
                    (true, Some(Float))
                }
                (Some(Int), _) | (_, Some(Int)) => (false, Some(Int)),
            }
        }
        use TypeRule::*;
        match self {
            Placeholder => (false, None),
            Const(s) => (true, Some(*s)),
            ArithOp(ty) => {
                let (done, res) = apply_binary_rule(*ty, incoming, op_helper);
                *ty = res;
                (done, res)
            }
            // CompareOp(ty) => {
            //     let (done, res) = apply_binary_rule(*ty, incoming, op_helper);
            //     *ty = res;
            //     (done, Some(Scalar::Int))
            // }
            MapKey(ty) => {
                let (done, res) = apply_binary_rule(*ty, incoming, |t1, t2| {
                    use Scalar::*;
                    match (t1, t2) {
                        (None, None) => (false, None),
                        (Some(Str), _) | (_, Some(Str)) | (Some(Float), _) | (_, Some(Float)) => {
                            (true, Some(Str))
                        }
                        (Some(Int), _) | (_, Some(Int)) => (false, Some(Int)),
                    }
                });
                *ty = res;
                (done, res)
            }
            MapVal(ty) => {
                let (done, res) = apply_binary_rule(*ty, incoming, |t1, t2| {
                    use Scalar::*;
                    match (t1, t2) {
                        (None, None) => (false, None),
                        (Some(Str), _) | (_, Some(Str)) => (true, Some(Str)),
                        (Some(Float), _) | (_, Some(Float)) => (true, Some(Float)),
                        (Some(Int), _) | (_, Some(Int)) => (false, Some(Int)),
                    }
                });
                *ty = res;
                (done, res)
            }
        }
    }
}

struct Node<P: Propagator> {
    // The propagator rule.
    prop: P,
    // The last value returned by the rule, or the default value.
    item: P::Item,
    // Will the propagator return new values?
    done: bool,
    // Is this node in the worklist?
    in_wl: bool,
}

#[derive(Default)]
pub(crate) struct Network<P: Propagator> {
    g: Graph<Node<P>, ()>,
    wl: Vec<NodeIx>,
}

impl<P: Propagator + std::fmt::Debug> Network<P>
where
    P::Item: Eq + Clone + Default,
{
    pub(crate) fn insert(&mut self, p: P, deps: impl Iterator<Item = NodeIx>) -> NodeIx {
        let ix = self.g.add_node(Node {
            prop: p,
            item: Default::default(),
            done: false,
            in_wl: true,
        });
        for d in deps {
            self.g.add_edge(d, ix, ());
        }
        self.wl.push(ix);
        ix
    }

    pub(crate) fn update(
        &mut self,
        ix: NodeIx,
        p: P,
        deps: impl Iterator<Item = NodeIx>,
    ) -> Result<()> {
        {
            let Node {
                prop,
                item: _,
                done: _,
                in_wl,
            } = self.g.node_weight_mut(ix).unwrap();
            if let Some(p) = prop.try_replace(p) {
                // Return a result here if we think this would represent a malformed program, not
                // just a bug in SSA conversion.
                return err!(
                    "internal error: tried to overwrite {:?} rule with {:?}",
                    prop,
                    p
                );
            }
            *in_wl = true;
        }
        for d in deps {
            self.g.add_edge(d, ix, ());
        }
        self.wl.push(ix);
        Ok(())
    }
    pub(crate) fn read(&self, ix: NodeIx) -> &P::Item {
        &self.g.node_weight(ix).unwrap().item
    }
    fn solve(&mut self) {
        let mut incoming: SmallVec<P::Item> = Default::default();
        let mut neighs: SmallVec<NodeIx> = Default::default();
        while let Some(node) = self.wl.pop() {
            use petgraph::Direction::*;
            let mut p = {
                let Node {
                    prop,
                    item: _,
                    done,
                    in_wl,
                } = self.g.node_weight_mut(node).unwrap();
                *in_wl = false;
                if *done {
                    continue;
                }
                prop.clone()
            };
            // TODO: Add support for something like timestamps so that we could filter for nodes
            // that have changed since the last iteration. Probably not a super high-priority
            // feature, but could be useful if we made this more general. (But for full generality
            // we could also probably just use timely/differential).
            incoming.extend(
                self.g
                    .neighbors_directed(node, Incoming)
                    .map(|ix| self.g.node_weight(ix).unwrap().item.clone()),
            );
            let (done_now, ty) = p.step(incoming.drain());
            let Node {
                prop,
                item,
                done,
                in_wl: _,
            } = self.g.node_weight_mut(node).unwrap();
            *done = done_now;
            *prop = p;
            if item != &ty {
                *item = ty;
                neighs.extend(self.g.neighbors_directed(node, Outgoing));
                for n in neighs.drain() {
                    self.wl.push(n);
                    *(&mut self.g.node_weight_mut(n).unwrap().in_wl) = true;
                }
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
            Map { key: _, val: _ } => err!("expected scalar, got map"),
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
            Map { key: _, val: _ } => err!("expected iterator, got map"),
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
    network: Network<TypeRule>,
    ident_map: HashMap<Ident, TVar>,
    str_node: NodeIx,
    int_node: NodeIx,
    float_node: NodeIx,

    canonical_ident: HashMap<Ident, NumTy>,
    uf: petgraph::unionfind::UnionFind<NumTy>,
    kind_map: HashMap<NumTy, Kind>,
    max_ident: NumTy,
}

impl Constraints {
    fn new(n: usize) -> Constraints {
        let mut network = Network::<TypeRule>::default();
        let str_node = network.insert(TypeRule::Const(Scalar::Str), None.into_iter());
        let int_node = network.insert(TypeRule::Const(Scalar::Int), None.into_iter());
        let float_node = network.insert(TypeRule::Const(Scalar::Float), None.into_iter());
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
                        Iter(i) => Iter(self.network.read(*i).clone()),
                        Scalar(i) => Scalar(self.network.read(*i).clone()),
                        Map { key, val } => Map {
                            key: self.network.read(*key).clone(),
                            val: self.network.read(*val).clone(),
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
            (StrBinop(_, o1, o2), e) | (NumBinop(_, o1, o2), e) => {
                self.assert_val_kind(o1, Kind::Scalar)?;
                self.assert_val_kind(o2, Kind::Scalar)?;
                match e {
                    Left(Kind::Scalar) => Ok(()),
                    Left(k) => err!("result of scalar binary operation used in {:?} context", k),
                    Right(id) => self.set_kind(*id, Kind::Scalar),
                }
            }
            (StrUnop(_, o), e) | (NumUnop(_, o), e) => {
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
        self.kind_map.get(&self.uf.find_mut(c)).map(Clone::clone)
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
                            self.network
                                .insert(TypeRule::MapVal(None), deps.into_iter()),
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
                        let key = self.network.insert(TypeRule::MapKey(None), ks.into_iter());
                        let val = self.network.insert(TypeRule::MapVal(None), vs.into_iter());
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
                            self.network
                                .insert(TypeRule::MapVal(None), deps.into_iter()),
                        ))
                    }
                }
            }
            StrUnop(_op, _pv) => err!("no string unops supported"),
            StrBinop(ast::StrBinop::Concat, _o1, _o2) => Ok(TVar::Scalar(self.str_node)),
            StrBinop(ast::StrBinop::Match, _o1, _o2) => Ok(TVar::Scalar(self.int_node)),
            NumUnop(op, o) => {
                use ast::NumUnop::*;
                Ok(TVar::Scalar(match op {
                    Column => self.str_node,
                    Not => self.int_node,
                    Neg | Pos => {
                        let inp = self.get_val(o)?.scalar()?;
                        self.network
                            .insert(TypeRule::ArithOp(None), Some(inp).into_iter())
                    }
                }))
            }
            NumBinop(op, o1, o2) => {
                use ast::NumBinop::*;
                Ok(TVar::Scalar(match op {
                    Plus | Minus | Mult | Mod => {
                        let i1 = self.get_val(o1)?.scalar()?;
                        let i2 = self.get_val(o2)?.scalar()?;
                        let deps: SmallVec<NodeIx> = smallvec![i1, i2];
                        self.network
                            .insert(TypeRule::ArithOp(None), deps.into_iter())
                    }
                    Div => self
                        .network
                        .insert(TypeRule::Const(Scalar::Float), None.into_iter()),
                }))
            }
            Index(map, ix) => {
                let (key, val) = self.get_val(map)?.map()?;
                let ix_node = self.get_scalar_val(ix)?;
                // We want to add ix_node as a dependency of key
                self.network
                    .update(key, TypeRule::Placeholder, Some(ix_node).into_iter())?;
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
                self.insert_rule(k, TypeRule::MapKey(None), Some(key).into_iter())?;
                let v_node = self.get_expr_constraints(val)?.scalar()?;
                self.network
                    .update(v, TypeRule::MapVal(None), Some(v_node).into_iter())?;
                Ok(())
            }
            AsgnVar(ident, exp) => {
                let ident_v = self.get_var(*ident)?;
                let exp_v = self.get_expr_constraints(exp)?;
                match (ident_v, exp_v) {
                    (TVar::Iter(i1), TVar::Iter(i2)) | (TVar::Scalar(i1), TVar::Scalar(i2)) => self
                        .network
                        .update(i1, TypeRule::MapVal(None), Some(i2).into_iter()),
                    (TVar::Map { key: k1, val: v1 }, TVar::Map { key: k2, val: v2 }) => {
                        self.network
                            .update(k1, TypeRule::MapKey(None), Some(k2).into_iter())?;
                        self.network
                            .update(v1, TypeRule::MapVal(None), Some(v2).into_iter())
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
                let item = self.network.insert(TypeRule::Placeholder, None.into_iter());
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
                let key = self.network.insert(TypeRule::Placeholder, None.into_iter());
                let val = self.network.insert(TypeRule::Placeholder, None.into_iter());
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
                    self.network
                        .update(*n, TypeRule::Placeholder, None.into_iter())?;
                    Ok(*n)
                }
            },
            Entry::Vacant(v) => {
                let n = self.network.insert(TypeRule::Placeholder, None.into_iter());
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
    pub(crate) fn insert_rule<'b, 'a: 'b>(
        &mut self,
        ident: NodeIx,
        rule: TypeRule,
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
        self.network.update(ident, rule, dep_nodes.into_iter())?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn plus_key_int() {
        let mut n = Network::default();
        use Scalar::*;
        use TypeRule::*;
        let i1 = n.insert(Const(Int), None.into_iter());
        let i2 = n.insert(Placeholder, None.into_iter());
        let addi12 = n.insert(ArithOp(None), vec![i1, i2].into_iter());
        n.update(i2, MapKey(None), vec![addi12].into_iter());
        n.solve();
        assert_eq!(n.read(i1), &Some(Int));
        assert_eq!(n.read(i2), &Some(Int));
        assert_eq!(n.read(addi12), &Some(Int));
    }

    #[test]
    fn plus_key_float() {
        let mut n = Network::default();
        use Scalar::*;
        use TypeRule::*;
        let i1 = n.insert(Const(Int), None.into_iter());
        let f1 = n.insert(Placeholder, None.into_iter());
        let add12 = n.insert(ArithOp(None), vec![i1, f1].into_iter());
        let f2 = n.insert(Const(Float), None.into_iter());
        n.update(f1, MapKey(None), vec![add12, f2].into_iter());
        n.solve();
        assert_eq!(n.read(i1), &Some(Int));
        assert_eq!(n.read(f1), &Some(Str));
        assert_eq!(n.read(f2), &Some(Float));
        assert_eq!(n.read(add12), &Some(Float));
    }
}
