//! This module contains definitions and metadata for builtin functions.
use crate::ast;
use crate::types::{Kind, Propagator, Scalar, SmallVec, TVar};
use smallvec::smallvec;

use std::convert::TryFrom;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Function {
    Unop(ast::Unop),
    Binop(ast::Binop),
    Print,
    Hasline,
    Nextline,
    Setcol,
    Split,
}

static_map!(
    FUNCTIONS<&'static str, Function>,
    ["print", Function::Print],
    ["split", Function::Split]
);

impl<'a> TryFrom<&'a str> for Function {
    type Error = (); // error means not found
    fn try_from(value: &'a str) -> Result<Function, ()> {
        match FUNCTIONS.get(value) {
            Some(v) => Ok(*v),
            None => Err(()),
        }
    }
}

impl Function {
    // All builtins are fixed-arity.
    fn arity_inner(&self) -> usize {
        use Function::*;
        match self {
            Hasline | Nextline | Unop(_) => 1,
            Setcol | Binop(_) | Print => 2,
            Split => 4, // 3?
        }
    }

    // Return kind is alway scalar. AWK lets you just assign into a provided map.
    pub(crate) fn signature(&self) -> SmallVec<Kind> {
        match self {
            Function::Split => smallvec![Kind::Scalar, Kind::Map, Kind::Scalar],
            _ => smallvec![Kind::Scalar; self.arity_inner()],
        }
    }
}

impl Propagator for Function {
    type Item = Scalar;
    fn arity(&self) -> Option<usize> {
        Some(self.arity_inner())
    }
    fn step(&self, incoming: &[Option<Scalar>]) -> (bool, Option<Scalar>) {
        use {
            ast::{Binop::*, Unop::*},
            Function::*,
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
            Hasline => (true, Some(Int)),
            Nextline => (true, Some(Str)),
            Setcol => (true, Some(Str)),
            Split => (true, Some(Int)),
        }
    }
    fn inputs(&self, incoming: &[Option<Scalar>]) -> SmallVec<Option<Scalar>> {
        use {
            ast::{Binop::*, Unop::*},
            Function::*,
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
            Hasline | Nextline => smallvec![Some(Str)],
            Setcol => smallvec![Some(Int), Some(Str)],
            Split => match (&incoming[1], &incoming[2]) {
                (Some(Int), _) | (None, _) => smallvec![Some(Str), Some(Int), Some(Str), Some(Str)],
                (_, _) => smallvec![Some(Str), Some(Str), Some(Str), Some(Str)],
            },
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Variable {
    ARGC,
    ARGV,
    FS,
    RS,
    NF,
    NR,
    FILENAME,
}

impl Variable {
    pub(crate) fn ty(&self) -> TVar<Scalar> {
        use Variable::*;
        match self {
            ARGC | NF | NR => TVar::Scalar(Scalar::Int),
            ARGV => TVar::Map {
                key: Scalar::Int,
                val: Scalar::Str,
            },
            FS | RS | FILENAME => TVar::Scalar(Scalar::Str),
        }
    }
}

impl<'a> TryFrom<&'a str> for Variable {
    type Error = (); // error means not found
    fn try_from(value: &'a str) -> Result<Variable, ()> {
        match VARIABLES.get(value) {
            Some(v) => Ok(*v),
            None => Err(()),
        }
    }
}

static_map!(
    VARIABLES<&'static str, Variable>,
    ["ARGC", Variable::ARGC],
    ["ARGV", Variable::ARGV],
    ["FS", Variable::FS],
    ["RS", Variable::RS],
    ["NF", Variable::NF],
    ["NR", Variable::NR],
    ["FILENAME", Variable::FILENAME]
);
