//! Algorithms and types pertaining to type deduction and converion.
//!
//! TODO: update this with more documentation when the algorithms are more fully baked.
use crate::common::{Graph, NodeIx};
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

pub(crate) trait Propagator: Default {
    type Item;
    fn step(&mut self, incoming: impl Iterator<Item = Self::Item>)
        -> (bool /* done */, Self::Item);
}

pub(crate) enum TypeRule {
    Placeholder,
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
    g: Graph<(P::Item, P), ()>,
}

impl<P: Propagator> Network<P> {
    fn read(&self, v: Var) -> &P::Item {
        &self.g.node_weight(v.0).unwrap().0
    }
}
