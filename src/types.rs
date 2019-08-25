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

pub(crate) trait Propagator<T>: Default {
    fn step(
        &mut self,
        incoming: impl Iterator<Item = T>,
        network: &Network<T, Self>,
    ) -> (bool /* done */, T);
}

pub(crate) enum TypeRule {
    Placeholder,
    // For things like +,% and *
    ArithBinop {
        ty: Option<Scalar>,
        lhs: Var,
        rhs: Var,
    },
    CompareOp {
        ty: Option<Scalar>,
        lhs: Var,
        rhs: Var,
    },
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

impl Propagator<Option<Scalar>> for TypeRule {
    fn step(
        &mut self,
        incoming: impl Iterator<Item = Option<Scalar>>,
        network: &Network<Option<Scalar>, Self>,
    ) -> (bool, Option<Scalar>) {
        use Scalar::*;
        use TypeRule::*;
        match self {
            PlaceHolder => (false, None),
            ArithBinop { ty, lhs, rhs } => {
                let (done, res) = op_helper(network.read(*lhs), network.read(*rhs));
                *ty = res;
                (done, res)
            }
            CompareOp { ty, lhs, rhs } => {
                let (done, res) = op_helper(network.read(*lhs), network.read(*rhs));
                *ty = res;
                (done, Some(Int))
            }
            MapKey(ty) => unimplemented!(),
            MapVal(ty) => unimplemented!(),
        }
    }
}

pub(crate) struct Network<T, P> {
    g: Graph<(T, P), ()>,
}

impl<T, P: Propagator<T>> Network<T, P> {
    fn read(&self, v: Var) -> &T {
        &self.g.node_weight(v.0).unwrap().0
    }
}
