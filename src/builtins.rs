//! This module contains definitions and metadata for builtin functions.
use crate::ast;
use crate::common::{NodeIx, Result};
use crate::compile;
use crate::types::{self, prop, Kind, Propagator, Scalar, SmallVec, TVar};
use smallvec::smallvec;

use std::convert::TryFrom;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Function {
    Unop(ast::Unop),
    Binop(ast::Binop),
    Print,
    PrintStdout,
    ReadErr,
    Nextline,
    ReadErrStdin,
    NextlineStdin,
    Setcol,
    Split,
    Length,
    Contains,
    Delete,
}

static_map!(
    FUNCTIONS<&'static str, Function>,
    ["print", Function::Print],
    ["split", Function::Split],
    ["length", Function::Length]
);

impl<'a> TryFrom<&'a str> for Function {
    type Error = (); // error means not found
    fn try_from(value: &'a str) -> std::result::Result<Function, ()> {
        match FUNCTIONS.get(value) {
            Some(v) => Ok(*v),
            None => Err(()),
        }
    }
}

impl Function {
    // All builtins are fixed-arity.
    pub(crate) fn fixed_arity(&self) -> usize {
        use Function::*;
        match self {
            ReadErrStdin | NextlineStdin => 0,
            Length | ReadErr | Nextline | PrintStdout | Unop(_) => 1,
            Setcol | Binop(_) => 2,
            // is this right?
            Delete | Contains => 2,
            Print | Split => 3,
        }
    }

    // feedback allows for certain functions to propagate type information back to their arguments.
    pub(crate) fn feedback(
        &self,
        nw: &mut prop::Network<prop::Rule>,
        cs: &types::Constants,
        deps: &[NodeIx],
    ) -> Result<()> {
        use std::iter::once;
        if let Function::Split = self {
            // Split acts to assign into a variable, that means we need to propagate information
            // back out. In this case those are going to be variables 1 and 2, the key and value
            // nodes.

            debug_assert_eq!(deps.len(), 4);
            let key = deps[1];
            let val = deps[2];
            nw.add_deps(key, once(cs.int_node))?;
            nw.add_deps(val, once(cs.str_node))?;
        } else if let Function::Contains = self {
            // Attempting to look up in a map informs its type.
            debug_assert_eq!(deps.len(), 3);
            let key = deps[0];
            let query = deps[2];
            nw.add_deps(key, once(query))?;
        }
        Ok(())
    }
    pub(crate) fn type_sig(
        &self,
        incoming: &[compile::Ty],
        // TODO make the return type optional?
    ) -> Result<(SmallVec<compile::Ty>, compile::Ty)> {
        use {
            ast::{Binop::*, Unop::*},
            compile::Ty::*,
            Function::*,
        };
        if incoming.len() != self.fixed_arity() {
            return err!(
                "function {} expected {} inputs but got {}",
                self,
                self.fixed_arity(),
                incoming.len()
            );
        }
        Ok(match self {
            Unop(Neg) | Unop(Pos) => match &incoming[0] {
                Str | Float => (smallvec![Float], Float),
                _ => (smallvec![Int], Int),
            },
            Unop(Column) => (smallvec![Int], Str),
            Binop(Concat) => (smallvec![Str; 2], Str),
            Binop(Match) => (smallvec![Str; 2], Int),
            // Not doesn't unconditionally convert to integers before negating it. Nonempty strings
            // are considered "truthy". Floating point numbers are converted beforehand:
            //    !5 == !1 == 0
            //    !0 == 1
            //    !"hi" == 0
            //    !(0.25) == 1
            Unop(Not) => match &incoming[0] {
                Float | Int => (smallvec![Int], Int),
                Str => (smallvec![Str], Int),
                _ => return err!("unexpected input to Not: {:?}", incoming),
            },
            Binop(LT) | Binop(GT) | Binop(LTE) | Binop(GTE) | Binop(EQ) => (
                match (incoming[0], incoming[1]) {
                    (Str, _) | (_, Str) => smallvec![Str; 2],
                    (Float, _) | (_, Float) => smallvec![Float; 2],
                    (Int, _) | (_, Int) => smallvec![Int; 2],
                    _ => return err!("invalid input spec for comparison op: {:?}", &incoming[..]),
                },
                Int,
            ),
            Binop(Plus) | Binop(Minus) | Binop(Mod) | Binop(Mult) => {
                match (incoming[0], incoming[1]) {
                    (Str, _) | (_, Str) | (Float, _) | (_, Float) => (smallvec![Float; 2], Float),
                    (_, _) => (smallvec![Int; 2], Int),
                }
            }
            Binop(Div) => (smallvec![Float;2], Float),
            Contains => match incoming[0] {
                MapIntInt | MapIntStr | MapIntFloat => (smallvec![incoming[0], Int], Int),
                MapStrInt | MapStrStr | MapStrFloat => (smallvec![incoming[0], Str], Int),
                _ => return err!("invalid input spec fo Contains: {:?}", &incoming[..]),
            },
            Delete => match incoming[0] {
                MapIntInt | MapIntStr | MapIntFloat => (smallvec![incoming[0], Int], Int),
                MapStrInt | MapStrStr | MapStrFloat => (smallvec![incoming[0], Str], Int),
                _ => return err!("invalid input spec fo Delete: {:?}", &incoming[..]),
            },
            Print => (smallvec![Str, Str, Int], Int),
            PrintStdout => (smallvec![Str], Int),
            Nextline => (smallvec![Str], Str),
            ReadErr => (smallvec![Str], Int),
            NextlineStdin => (smallvec![], Str),
            ReadErrStdin => (smallvec![], Int),
            // irrelevant return type
            Setcol => (smallvec![Int, Str], Int),
            Length => (smallvec![incoming[0]], Int),
            // Split's second input can be a map of either type
            Split => {
                if let MapIntStr | MapStrStr = incoming[1] {
                    (smallvec![Str, incoming[1], Str], Int)
                } else {
                    return err!("invalid input spec for split: {:?}", &incoming[..]);
                }
            }
        })
    }

    // Return kind is alway scalar. AWK lets you just assign into a provided map.
    pub(crate) fn signature(&self) -> SmallVec<Option<Kind>> {
        match self {
            Function::Split => smallvec![Some(Kind::Scalar), Some(Kind::Map), Some(Kind::Scalar)],
            Function::Contains | Function::Delete => smallvec![Some(Kind::Map), Some(Kind::Scalar)],
            Function::Length => smallvec![None],
            _ => smallvec![Some(Kind::Scalar); self.fixed_arity()],
        }
    }
}

impl Propagator for Function {
    type Item = Scalar;
    fn arity(&self) -> Option<usize> {
        Some(self.fixed_arity())
    }
    // TODO unify more of this with `input_ty`
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
            Contains => (true, Some(Int)),
            Delete => (true, Some(Int)),
            PrintStdout => (true, None),
            ReadErr | ReadErrStdin => (true, Some(Int)),
            Nextline | NextlineStdin => (true, Some(Str)),
            Setcol => (true, Some(Int)), // no result
            Split => (true, Some(Int)),
            Length => (true, Some(Int)),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Variable {
    ARGC,
    ARGV,
    OFS,
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
            OFS | FS | RS | FILENAME => TVar::Scalar(Scalar::Str),
        }
    }
}

impl<'a> TryFrom<&'a str> for Variable {
    type Error = (); // error means not found
    fn try_from(value: &'a str) -> std::result::Result<Variable, ()> {
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
    ["OFS", Variable::OFS],
    ["FS", Variable::FS],
    ["RS", Variable::RS],
    ["NF", Variable::NF],
    ["NR", Variable::NR],
    ["FILENAME", Variable::FILENAME]
);
