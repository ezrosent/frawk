use crate::ast;
use crate::types::{Propagator, Scalar, SmallVec};
use smallvec::smallvec;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Builtin {
    Unop(ast::Unop),
    Binop(ast::Binop),
    Print,
}

impl Propagator for Builtin {
    type Item = Scalar;
    fn arity(&self) -> Option<usize> {
        use Builtin::*;
        match self {
            Unop(_) => Some(1),
            Binop(_) => Some(2),
            Print => None,
        }
    }
    fn step(&self, incoming: &[Option<Scalar>]) -> (bool, Option<Scalar>) {
        use {
            ast::{Binop::*, Unop::*},
            Builtin::*,
            Scalar::*,
        };
        match self {
            Unop(Neg) | Unop(Pos) => match &incoming[0] {
                Some(Str) | Some(Float) => (true, Some(Float)),
                x => (false, *x),
            },
            // TODO: Column this should get desugared before making CFG? We can desugar it when
            // building bytecode as well, though.
            Unop(Column) | Binop(Concat) => (true, Some(Str)),
            Unop(Not) | Binop(Match) | Binop(LT) | Binop(GT) | Binop(LTE) | Binop(GTE)
            | Binop(EQ) => (true, Some(Int)),
            Binop(Plus) | Binop(Minus) | Binop(Mod) | Binop(Mult) => {
                match (&incoming[0], &incoming[1]) {
                    (Some(Str), _) | (_, Some(Str)) | (Some(Float), _) | (_, Some(Float)) => {
                        (true, Some(Float))
                    }
                    (_, _) => (false, Some(Int)),
                }
            }
            Binop(Div) => (true, Some(Float)),
            Print => (true, None),
        }
    }
    fn inputs(&self, incoming: &[Option<Scalar>]) -> SmallVec<Option<Scalar>> {
        use {
            ast::{Binop::*, Unop::*},
            Builtin::*,
            Scalar::*,
        };
        match self {
            Unop(Neg) | Unop(Pos) => match &incoming[0] {
                Some(Str) | Some(Float) => smallvec![Some(Float)],
                x => smallvec![*x],
            },
            Unop(Column) => smallvec![Some(Int)],
            Binop(Concat) | Binop(Match) => smallvec![Some(Str); 2],
            // Not doesn't unconditionally convert to integers before negating it. Nonempty strings
            // are considered "truthy". Floating point numbers are converted beforehand:
            //    !5 == !1 == 0
            //    !0 == 1
            //    !"hi" == 0
            //    !(0.25) == 1
            Unop(Not) => smallvec![match &incoming[0] {
                None | Some(Float) => Some(Int),
                other => *other,
            }],
            Binop(EQ) => unimplemented!(),
            Binop(LT) | Binop(GT) | Binop(LTE) | Binop(GTE) | Binop(Plus) | Binop(Minus)
            | Binop(Mod) | Binop(Mult) => match (incoming[0], incoming[1]) {
                (Some(Str), _) | (_, Some(Str)) | (Some(Float), _) | (_, Some(Float)) => {
                    smallvec![Some(Float); 2]
                }
                (_, _) => smallvec![Some(Int); 2],
            },
            Binop(Div) => smallvec![Some(Float);2],
            Print => smallvec![Some(Str);incoming.len()],
        }
    }
}
