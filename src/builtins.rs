//! This module contains definitions and metadata for builtin functions.
use crate::ast;
use crate::common::{NodeIx, Result};
use crate::compile;
use crate::runtime::{Int, IntMap, Str};
use crate::types::{self, SmallVec};
use smallvec::smallvec;

use std::convert::TryFrom;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Function {
    Unop(ast::Unop),
    Binop(ast::Binop),
    Print,
    PrintStdout,
    Close,
    ReadErr,
    Nextline,
    ReadErrStdin,
    NextlineStdin,
    Setcol,
    Split,
    Length,
    Contains,
    Delete,
    Match,
}

static_map!(
    FUNCTIONS<&'static str, Function>,
    ["close", Function::Close],
    ["print", Function::Print],
    ["split", Function::Split],
    ["length", Function::Length],
    ["match", Function::Match]
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

pub(crate) trait IsSprintf {
    fn is_sprintf(&self) -> bool;
}
impl<'a> IsSprintf for &'a str {
    fn is_sprintf(&self) -> bool {
        *self == "sprintf"
    }
}

impl Function {
    // feedback allows for certain functions to propagate type information back to their arguments.
    pub(crate) fn feedback(&self, args: &[NodeIx], ctx: &mut types::TypeContext) {
        use types::{BaseTy, Constraint, TVar::*};
        match self {
            Function::Split => {
                let arg1 = ctx.constant(
                    Map {
                        key: BaseTy::Int,
                        val: BaseTy::Str,
                    }
                    .abs(),
                );
                ctx.nw.add_dep(arg1, args[1], Constraint::Flows(()));
            }
            Function::Contains | Function::Delete => {
                let arr = args[0];
                let query = args[1];
                ctx.set_key(arr, query);
            }
            _ => {}
        };
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
        if let Some(a) = self.arity() {
            if incoming.len() != a {
                return err!(
                    "function {} expected {} inputs but got {}",
                    self,
                    a,
                    incoming.len()
                );
            }
        }
        Ok(match self {
            Unop(Neg) | Unop(Pos) => match &incoming[0] {
                Str | Float => (smallvec![Float], Float),
                _ => (smallvec![Int], Int),
            },
            Unop(Column) => (smallvec![Int], Str),
            Binop(Concat) => (smallvec![Str; 2], Str),
            Binop(IsMatch) => (smallvec![Str; 2], Int),
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
            Close => (smallvec![Str], Str),
            Match => (smallvec![Str, Str], Int),
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

    pub(crate) fn arity(&self) -> Option<usize> {
        use Function::*;
        Some(match self {
            ReadErrStdin | NextlineStdin => 0,
            Close | Length | ReadErr | Nextline | PrintStdout | Unop(_) => 1,
            Match | Setcol | Binop(_) => 2,
            Delete | Contains => 2,
            Print | Split => 3,
        })
    }

    // TODO(ezr): rename this once old types module is gone
    pub(crate) fn step(&self, args: &[types::State]) -> Result<types::State> {
        use {
            ast::{Binop::*, Unop::*},
            types::{BaseTy, TVar::*},
            Function::*,
        };
        match self {
            Unop(Neg) | Unop(Pos) => match &args[0] {
                Some(Scalar(Some(BaseTy::Str))) | Some(Scalar(Some(BaseTy::Float))) => {
                    Ok(Scalar(BaseTy::Float).abs())
                }
                x => Ok(*x),
            },
            Binop(Plus) | Binop(Minus) | Binop(Mod) | Binop(Mult) => {
                use BaseTy::*;
                match (&args[0], &args[1]) {
                    (Some(Scalar(Some(Str))), _)
                    | (_, Some(Scalar(Some(Str))))
                    | (Some(Scalar(Some(Float))), _)
                    | (_, Some(Scalar(Some(Float)))) => Ok(Scalar(Float).abs()),
                    (_, _) => Ok(Scalar(Int).abs()),
                }
            }
            Binop(Div) => Ok(Scalar(BaseTy::Float).abs()),
            Setcol | Print | PrintStdout => Ok(Scalar(BaseTy::Null).abs()),
            Unop(Not) | Binop(IsMatch) | Binop(LT) | Binop(GT) | Binop(LTE) | Binop(GTE)
            | Binop(EQ) | Length | Split | ReadErr | ReadErrStdin | Contains | Delete | Match => {
                Ok(Scalar(BaseTy::Int).abs())
            }
            Unop(Column) | Binop(Concat) | Nextline | NextlineStdin => {
                Ok(Scalar(BaseTy::Str).abs())
            }
            Close => Ok(None),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Variable {
    ARGC = 0,
    ARGV = 1,
    OFS = 2,
    FS = 3,
    RS = 4,
    NF = 5,
    NR = 6,
    FILENAME = 7,
    RSTART = 8,
    RLENGTH = 9,
}

impl From<Variable> for compile::Ty {
    fn from(v: Variable) -> compile::Ty {
        use Variable::*;
        match v {
            FS | OFS | RS | FILENAME => compile::Ty::Str,
            ARGC | NF | NR | RSTART | RLENGTH => compile::Ty::Int,
            ARGV => compile::Ty::MapIntStr,
        }
    }
}

pub(crate) struct Variables<'a> {
    pub argc: Int,
    pub argv: IntMap<Str<'a>>,
    pub fs: Str<'a>,
    pub ofs: Str<'a>,
    pub rs: Str<'a>,
    pub nf: Int,
    pub nr: Int,
    pub filename: Str<'a>,
    pub rstart: Int,
    pub rlength: Int,
}

impl<'a> Default for Variables<'a> {
    fn default() -> Variables<'a> {
        Variables {
            argc: 0,
            argv: Default::default(),
            fs: "[ \t]+".into(),
            ofs: " ".into(),
            rs: "\n".into(),
            nr: 0,
            nf: 0,
            filename: Default::default(),
            rstart: 0,
            rlength: -1,
        }
    }
}
impl<'a> Variables<'a> {
    pub fn load_int(&self, var: Variable) -> Result<Int> {
        use Variable::*;
        Ok(match var {
            ARGC => self.argc,
            NF => self.nf,
            NR => self.nr,
            RSTART => self.rstart,
            RLENGTH => self.rlength,
            OFS | FS | RS | FILENAME | ARGV => return err!("var {} not an int", var),
        })
    }

    pub fn store_int(&mut self, var: Variable, i: Int) -> Result<()> {
        use Variable::*;
        Ok(match var {
            ARGC => self.argc = i,
            NF => self.nf = i,
            NR => self.nr = i,
            RSTART => self.rstart = i,
            RLENGTH => self.rlength = i,
            OFS | FS | RS | FILENAME | ARGV => return err!("var {} not an int", var),
        })
    }

    pub fn load_str(&self, var: Variable) -> Result<Str<'a>> {
        use Variable::*;
        Ok(match var {
            FS => self.fs.clone(),
            OFS => self.ofs.clone(),
            RS => self.rs.clone(),
            FILENAME => self.filename.clone(),
            ARGC | ARGV | NF | NR | RSTART | RLENGTH => return err!("var {} not a string", var),
        })
    }

    pub fn store_str(&mut self, var: Variable, s: Str<'a>) -> Result<()> {
        use Variable::*;
        Ok(match var {
            FS => self.fs = s,
            OFS => self.ofs = s,
            RS => self.rs = s,
            FILENAME => self.filename = s,
            ARGC | ARGV | NF | NR | RSTART | RLENGTH => return err!("var {} not a string", var),
        })
    }

    pub fn load_intmap(&self, var: Variable) -> Result<IntMap<Str<'a>>> {
        use Variable::*;
        match var {
            ARGV => Ok(self.argv.clone()),
            OFS | ARGC | NF | NR | FS | RS | FILENAME | RSTART | RLENGTH => {
                err!("var {} is not a map", var)
            }
        }
    }

    pub fn store_intmap(&mut self, var: Variable, m: IntMap<Str<'a>>) -> Result<()> {
        use Variable::*;
        match var {
            ARGV => Ok(self.argv = m),
            OFS | ARGC | NF | NR | FS | RS | FILENAME | RSTART | RLENGTH => {
                err!("var {} is not a map", var)
            }
        }
    }
}
impl Variable {
    pub(crate) fn ty(&self) -> types::TVar<types::BaseTy> {
        use Variable::*;
        match self {
            ARGC | NF | NR | RSTART | RLENGTH => types::TVar::Scalar(types::BaseTy::Int),
            // TODO(ezr): For full compliance, this may have to be Str -> Str
            //  If we had
            //  m["x"] = 1;
            //  if (true) {
            //      m = ARGV
            //  }
            //  I think we have SSA:
            //  L0:
            //    m0["x"] = 1;
            //    jmpif false L2
            //  L1:
            //    m1 = ARGV
            //  L2:
            //    m2 = phi [L0: m0, L1: m1]
            //
            //  And m0 and m1 have to be the same type, because we do not want to convert between map
            //  types.
            //  I think the solution here is just to have ARGV be a local variable. It doesn't
            //  actually have to be a builtin.
            ARGV => types::TVar::Map {
                key: types::BaseTy::Int,
                val: types::BaseTy::Str,
            },
            OFS | FS | RS | FILENAME => types::TVar::Scalar(types::BaseTy::Str),
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

impl<'a> TryFrom<usize> for Variable {
    type Error = (); // error means not found
    fn try_from(value: usize) -> std::result::Result<Variable, ()> {
        use Variable::*;
        match value {
            0 => Ok(ARGC),
            1 => Ok(ARGV),
            2 => Ok(OFS),
            3 => Ok(FS),
            4 => Ok(RS),
            5 => Ok(NF),
            6 => Ok(NR),
            7 => Ok(FILENAME),
            8 => Ok(RSTART),
            9 => Ok(RLENGTH),
            _ => Err(()),
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
    ["FILENAME", Variable::FILENAME],
    ["RSTART", Variable::RSTART],
    ["RLENGTH", Variable::RLENGTH]
);
