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
    fn step(&mut self, network: &Network<T, Self>) -> T;
}

pub(crate) enum TypeRule {
    Placeholder,
    // For things like +,% and *
    ArithBinop {
        ty: Option<Scalar>,
        lhs: Var,
        rhs: Var,
    },
    MapIndex {
        map_key: Target,
        map_val: Target,
    },
    CompareOp {
        lhs: Target,
        rhs: Target,
    },
}

impl Default for TypeRule {
    fn default() -> TypeRule {
        TypeRule::Placeholder
    }
}

impl Propagator<Option<Scalar>> for TypeRule {
    fn step(&mut self, network: &Network<Option<Scalar>, Self>) -> Option<Scalar> {
        use Scalar::*;
        use TypeRule::*;
        match self {
            PlaceHolder => None,
            ArithBinop { ty, lhs, rhs } => match (network.read(*lhs), network.read(*rhs)) {
                // No informmation to propagate
                (None, None) => None,
                (Some(Str), _) | (_, Some(Str)) | (Some(Float), _) | (_, Some(Float)) => {
                    *ty = Some(Float);
                    Some(Float)
                }
                (Some(Int), _) | (_, Some(Int)) => {
                    *ty = Some(Int);
                    Some(Int)
                }
            },
            MapIndex { map_key, map_val } => unimplemented!(),
            CompareOp { lhs, rhs } => unimplemented!(),
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
