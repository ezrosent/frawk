//! Algorithms and types pertaining to type deduction and converion.
//!
//! TODO: update this with more documentation when the algorithms are more fully baked.
use crate::common::{Graph, NodeIx};
use crate::hashbrown::HashSet;
type SmallVec<T> = crate::smallvec::SmallVec<[T; 2]>;
#[derive(Clone, Copy)]
pub(crate) struct Var(NodeIx);
pub(crate) struct Target {
    var: Var,
    ty: Option<Scalar>,
}

impl Target {
    fn from_var(v: Var) -> Self {
        Target { var: v, ty: None }
    }
}

#[derive(Copy, Clone)]
pub(crate) enum Scalar {
    Str,
    Int,
    Float,
}

pub(crate) struct MapTy {
    key: Option<Scalar>,
    val: Option<Scalar>,
}

pub(crate) trait Propagator: Default + Clone {
    type Item;
    fn step(&mut self, incoming: impl Iterator<Item = Self::Item>)
        -> (bool /* done */, Self::Item);
}

#[derive(Clone)]
pub(crate) enum TypeRule {
    Placeholder,
    // For literals, and also operators like '/' (which will always coerce to float) or bitwise
    // operations (which always coerce to integers).
    Const(Scalar),
    // For things like +,% and *
    ArithOp(Option<Scalar>),
    CompareOp(Option<Scalar>),
    MapKey(Option<Scalar>),
    MapVal(Option<Scalar>),
}

impl Default for TypeRule {
    fn default() -> TypeRule {
        TypeRule::Placeholder
    }
}

fn op_helper(o1: &Option<Scalar>, o2: &Option<Scalar>) -> (bool, Option<Scalar>) {
    use Scalar::*;
    match (o1, o2) {
        // No informmation to propagate
        (None, None) => (false, None),
        (Some(Str), _) | (_, Some(Str)) | (Some(Float), _) | (_, Some(Float)) => {
            (true, Some(Float))
        }
        (Some(Int), _) | (_, Some(Int)) => (false, Some(Int)),
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
    fn step(
        &mut self,
        incoming: impl Iterator<Item = Option<Scalar>>,
    ) -> (bool /* done */, Option<Scalar>) {
        use TypeRule::*;
        match self {
            Placeholder => (false, None),
            Const(s) => (true, Some(*s)),
            ArithOp(ty) => {
                let (done, res) = apply_binary_rule(*ty, incoming, op_helper);
                *ty = res;
                (done, res)
            }
            CompareOp(ty) => {
                let (done, res) = apply_binary_rule(*ty, incoming, op_helper);
                *ty = res;
                (done, Some(Scalar::Int))
            }
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

pub(crate) struct Network<P: Propagator> {
    g: Graph<(bool, P::Item, P), ()>,
}

impl<P: Propagator> Network<P>
where
    P::Item: Eq + Clone,
{
    fn solve(&mut self) {
        let mut worklist: HashSet<NodeIx> = self.g.node_indices().collect();
        let mut incoming: SmallVec<P::Item> = Default::default();
        while worklist.len() > 0 {
            use petgraph::Direction::*;
            let node = {
                let fst = worklist
                    .iter()
                    .next()
                    .expect("worklist cannot be empty")
                    .clone();
                worklist
                    .take(&fst)
                    .expect("worklist must yield elements from the set")
            };
            let (done, _ty, mut p) = self.g.node_weight(node).unwrap().clone();
            if done {
                continue;
            }
            incoming.extend(
                self.g
                    .neighbors_directed(node, Incoming)
                    .map(|ix| self.g.node_weight(ix).unwrap().1.clone()),
            );
            let (done_now, next) = p.step(incoming.drain());
            let (ref mut done_ref, ref mut ty_ref, ref mut p_ref) =
                self.g.node_weight_mut(node).unwrap();
            *done_ref = done_now;
            *ty_ref = next;
            *p_ref = p;
            worklist.extend(self.g.neighbors_directed(node, Outgoing));
        }
    }
}
