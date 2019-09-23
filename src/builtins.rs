use crate::ast;
use crate::types::{Kind, Propagator, Scalar, SmallVec};
use smallvec::smallvec;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Builtin {
    Unop(ast::Unop),
    Binop(ast::Binop),
    Print,
    Getline,
}

// TODO: need to incode the "kind signature" of different builtins. It may be as simple as
// returning a (SmallVec<Kind>, Option<Kind>); as I think we will only support builtins with fixed
// kinds to start.

impl Builtin {
    // All builtins are fixed-arity.
    fn arity_inner(&self) -> usize {
        use Builtin::*;
        match self {
            Getline | Unop(_) => 1,
            Binop(_) | Print => 2,
        }
    }

    // Return kind is alway scalar. AWK lets you just assign into a provided map.
    pub(crate) fn signature(&self) -> SmallVec<Kind> {
        smallvec![Kind::Scalar; self.arity_inner()]
    }
}

impl Propagator for Builtin {
    type Item = Scalar;
    fn arity(&self) -> Option<usize> {
        Some(self.arity_inner())
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
            // TODO: Column should get desugared before making CFG? We can desugar it when
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
            Getline => (true, Some(Str)),
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
            Binop(EQ) => match (incoming[0], incoming[1]) {
                (Some(Str), _) | (_, Some(Str)) => smallvec![Some(Str); 2],
                (Some(Float), _) | (_, Some(Float)) => smallvec![Some(Float); 2],
                (Some(Int), _) | (_, Some(Int)) => smallvec![Some(Int); 2],
                (None, None) => smallvec![None; 2],
            },
            Binop(LT) | Binop(GT) | Binop(LTE) | Binop(GTE) | Binop(Plus) | Binop(Minus)
            | Binop(Mod) | Binop(Mult) => match (incoming[0], incoming[1]) {
                (Some(Str), _) | (_, Some(Str)) | (Some(Float), _) | (_, Some(Float)) => {
                    smallvec![Some(Float); 2]
                }
                (_, _) => smallvec![Some(Int); 2],
            },
            Binop(Div) => smallvec![Some(Float);2],
            Print => smallvec![Some(Str);incoming.len()],
            Getline => smallvec![Some(Str)],
        }
    }
}
