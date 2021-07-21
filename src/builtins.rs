//! This module contains definitions and metadata for builtin functions and builtin variables.
use crate::ast;
#[allow(unused_imports)]
use crate::common::Either;
use crate::common::{NodeIx, Result};
use crate::compile;
use crate::runtime::{Int, IntMap, Str, StrMap};
use crate::types::{self, SmallVec};
use smallvec::smallvec;

use std::convert::TryFrom;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Function {
    Unop(ast::Unop),
    Binop(ast::Binop),
    FloatFunc(FloatFunc),
    IntFunc(Bitwise),
    Close,
    ReadErr,
    ReadErrCmd,
    Nextline,
    ReadErrStdin,
    NextlineStdin,
    NextlineCmd,
    ReadLineStdinFused,
    NextFile,
    Setcol,
    Split,
    Length,
    Contains,
    Delete,
    Clear,
    Match,
    SubstrIndex,
    Sub,
    GSub,
    EscapeCSV,
    EscapeTSV,
    JoinCols,
    JoinCSV,
    JoinTSV,
    Substr,
    ToInt,
    HexToInt,
    Rand,
    Srand,
    ReseedRng,
    System,
    // For header-parsing logic
    UpdateUsedFields,
    SetFI,
    ToUpper,
    ToLower,
    IncMap,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Bitwise {
    Complement,
    And,
    Or,
    LogicalRightShift,
    ArithmeticRightShift,
    LeftShift,
    Xor,
}

impl Bitwise {
    pub fn func_name(&self) -> &'static str {
        use Bitwise::*;
        match self {
            Complement => "compl",
            And => "and",
            Or => "or",
            LogicalRightShift => "rshiftl",
            ArithmeticRightShift => "rshift",
            LeftShift => "lshift",
            Xor => "xor",
        }
    }
    pub fn eval1(&self, op: i64) -> i64 {
        use Bitwise::*;
        match self {
            Complement => !op,
            And | Or | LogicalRightShift | ArithmeticRightShift | LeftShift | Xor => {
                panic!("bitwise: mismatched arity!")
            }
        }
    }
    pub fn eval2(&self, lhs: i64, rhs: i64) -> i64 {
        use Bitwise::*;
        match self {
            And => lhs & rhs,
            Or => lhs | rhs,
            LogicalRightShift => (lhs as usize).wrapping_shr(rhs as u32) as i64,
            ArithmeticRightShift => lhs.wrapping_shr(rhs as u32),
            LeftShift => lhs.wrapping_shl(rhs as u32),
            Xor => lhs ^ rhs,
            Complement => panic!("bitwise: mismatched arity!"),
        }
    }
    pub fn arity(&self) -> usize {
        use Bitwise::*;
        match self {
            Complement => 1,
            And | Or | LogicalRightShift | ArithmeticRightShift | LeftShift | Xor => 2,
        }
    }
    fn sig(&self) -> (SmallVec<compile::Ty>, compile::Ty) {
        use compile::Ty;
        (smallvec![Ty::Int; self.arity()], Ty::Int)
    }
    fn ret_state(&self) -> types::State {
        types::TVar::Scalar(types::BaseTy::Int).abs()
    }
}

// TODO: move the llvm-level code back into the LLVM module.

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FloatFunc {
    Cos,
    Sin,
    Atan,
    Atan2,
    // Natural log
    Log,
    // Log base 2
    Log2,
    // Log base 10
    Log10,
    Sqrt,
    // e^
    Exp,
}

impl FloatFunc {
    pub fn eval1(&self, op: f64) -> f64 {
        use FloatFunc::*;
        match self {
            Cos => op.cos(),
            Sin => op.sin(),
            Atan => op.atan(),
            Log => op.ln(),
            Log2 => op.log2(),
            Log10 => op.log10(),
            Sqrt => op.sqrt(),
            Exp => op.exp(),
            Atan2 => panic!("float: mismatched arity!"),
        }
    }
    pub fn eval2(&self, x: f64, y: f64) -> f64 {
        use FloatFunc::*;
        match self {
            Atan2 => x.atan2(y),
            Sqrt | Cos | Sin | Atan | Log | Log2 | Log10 | Exp => {
                panic!("float: mismatched arity!")
            }
        }
    }

    pub fn func_name(&self) -> &'static str {
        use FloatFunc::*;
        match self {
            Cos => "cos",
            Sin => "sin",
            Atan => "atan",
            Log => "log",
            Log2 => "log2",
            Log10 => "log10",
            Sqrt => "sqrt",
            Atan2 => "atan2",
            Exp => "exp",
        }
    }

    pub fn arity(&self) -> usize {
        use FloatFunc::*;
        match self {
            Sqrt | Cos | Sin | Atan | Log | Log2 | Log10 | Exp => 1,
            Atan2 => 2,
        }
    }
    fn sig(&self) -> (SmallVec<compile::Ty>, compile::Ty) {
        use compile::Ty;
        (smallvec![Ty::Float; self.arity()], Ty::Float)
    }
    fn ret_state(&self) -> types::State {
        types::TVar::Scalar(types::BaseTy::Float).abs()
    }
}

// This map is used to look up functions that are called in the program source and determine if
// they are builtin functions. Note that not all members of the Function enum are present here.
// This includes only the "public" functions.
static_map!(
    FUNCTIONS<&'static str, Function>,
    ["close", Function::Close],
    ["split", Function::Split],
    ["length", Function::Length],
    ["match", Function::Match],
    ["sub", Function::Sub],
    ["gsub", Function::GSub],
    ["substr", Function::Substr],
    ["int", Function::ToInt],
    ["hex", Function::HexToInt],
    ["exp", Function::FloatFunc(FloatFunc::Exp)],
    ["cos", Function::FloatFunc(FloatFunc::Cos)],
    ["sin", Function::FloatFunc(FloatFunc::Sin)],
    ["atan", Function::FloatFunc(FloatFunc::Atan)],
    ["log", Function::FloatFunc(FloatFunc::Log)],
    ["log2", Function::FloatFunc(FloatFunc::Log2)],
    ["log10", Function::FloatFunc(FloatFunc::Log10)],
    ["sqrt", Function::FloatFunc(FloatFunc::Sqrt)],
    ["atan2", Function::FloatFunc(FloatFunc::Atan2)],
    ["and", Function::IntFunc(Bitwise::And)],
    ["or", Function::IntFunc(Bitwise::Or)],
    ["compl", Function::IntFunc(Bitwise::Complement)],
    ["lshift", Function::IntFunc(Bitwise::LeftShift)],
    ["rshift", Function::IntFunc(Bitwise::ArithmeticRightShift)],
    ["rshiftl", Function::IntFunc(Bitwise::LogicalRightShift)],
    ["xor", Function::IntFunc(Bitwise::Xor)],
    ["join_fields", Function::JoinCols],
    ["join_csv", Function::JoinCSV],
    ["join_tsv", Function::JoinTSV],
    ["escape_csv", Function::EscapeCSV],
    ["escape_tsv", Function::EscapeTSV],
    ["rand", Function::Rand],
    ["srand", Function::Srand],
    ["index", Function::SubstrIndex],
    ["toupper", Function::ToUpper],
    ["tolower", Function::ToLower],
    ["system", Function::System]
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
    pub(crate) fn feedback(&self, args: &[NodeIx], res: NodeIx, ctx: &mut types::TypeContext) {
        use types::{BaseTy, Constraint, TVar::*};
        if args.len() < self.arity().unwrap_or(0) {
            return;
        }
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
            Function::Clear => {
                let is_map = ctx.constant(Some(Map {
                    key: None,
                    val: None,
                }));
                ctx.nw.add_dep(is_map, args[0], Constraint::Flows(()));
            }
            Function::Contains => {
                let arr = args[0];
                let query = args[1];
                ctx.nw.add_dep(query, arr, Constraint::KeyIn(()));
            }
            Function::Delete => {
                let arr = args[0];
                let query = args[1];
                ctx.nw.add_dep(query, arr, Constraint::KeyIn(()));
            }
            Function::IncMap => {
                let arr = args[0];
                let k = args[1];
                let v = res;
                ctx.nw.add_dep(k, arr, Constraint::KeyIn(()));
                ctx.nw.add_dep(v, arr, Constraint::ValIn(()));
                ctx.nw.add_dep(arr, v, Constraint::Val(()));
            }
            Function::Sub | Function::GSub => {
                let out_str = args[2];
                let str_const = ctx.constant(Scalar(BaseTy::Str).abs());
                ctx.nw.add_dep(str_const, out_str, Constraint::Flows(()));
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
        fn arith_sig(x: compile::Ty, y: compile::Ty) -> (SmallVec<compile::Ty>, compile::Ty) {
            use compile::Ty::*;
            match (x, y) {
                (Str, _) | (_, Str) | (Float, _) | (_, Float) => (smallvec![Float; 2], Float),
                (_, _) => (smallvec![Int; 2], Int),
            }
        }
        Ok(match self {
            FloatFunc(ff) => ff.sig(),
            IntFunc(bw) => bw.sig(),
            Unop(Neg) | Unop(Pos) => match &incoming[0] {
                Str | Float => (smallvec![Float], Float),
                _ => (smallvec![Int], Int),
            },
            Unop(Column) => (smallvec![Int], Str),
            Binop(Concat) => (smallvec![Str; 2], Str),
            SubstrIndex | Binop(IsMatch) => (smallvec![Str; 2], Int),
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
                    (Str, Str) => smallvec![Str; 2],
                    (Int, Int) | (Null, Int) | (Int, Null) | (Null, Null) => smallvec![Int; 2],
                    (_, Str) | (Str, _) | (Float, _) | (_, Float) => smallvec![Float; 2],
                    _ => return err!("invalid input spec for comparison op: {:?}", &incoming[..]),
                },
                Int,
            ),
            Binop(Plus) | Binop(Minus) | Binop(Mod) | Binop(Mult) => {
                arith_sig(incoming[0], incoming[1])
            }
            Binop(Pow) | Binop(Div) => (smallvec![Float;2], Float),
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
            IncMap => {
                let map = incoming[0];
                if !map.is_array() {
                    return err!(
                        "first argument to inc_map must be an array type, got: {:?}",
                        map
                    );
                }
                let val = map.val().unwrap();
                let (args, res) = arith_sig(incoming[2], val);
                (
                    smallvec![incoming[0], incoming[0].key().unwrap(), args[0]],
                    res,
                )
            }
            Clear => {
                if incoming.len() == 1 && incoming[0].is_array() {
                    (smallvec![incoming[0]], Int)
                } else {
                    return err!(
                        "invalid input spec for delete (of a map): {:?}",
                        &incoming[..]
                    );
                }
            }
            Srand => (smallvec![Int], Int),
            System | HexToInt => (smallvec![Str], Int),
            ReseedRng => (smallvec![], Int),
            Rand => (smallvec![], Float),
            ToInt => {
                let inc = incoming[0];
                match inc {
                    Null | Int | Float | Str => (smallvec![inc], Int),
                    _ => {
                        return err!(
                            "can only convert scalar values to integers, got input with type: {:?}",
                            inc
                        )
                    }
                }
            }
            NextlineCmd | Nextline => (smallvec![Str], Str),
            ReadErrCmd | ReadErr => (smallvec![Str], Int),
            UpdateUsedFields | NextFile | ReadLineStdinFused => (smallvec![], Int),
            NextlineStdin => (smallvec![], Str),
            ReadErrStdin => (smallvec![], Int),
            // irrelevant return type
            Setcol => (smallvec![Int, Str], Int),
            Length => (smallvec![incoming[0]], Int),
            Close => (smallvec![Str], Str),
            Sub | GSub => (smallvec![Str, Str, Str], Int),
            ToUpper | ToLower | EscapeCSV | EscapeTSV => (smallvec![Str], Str),
            Substr => (smallvec![Str, Int, Int], Str),
            Match => (smallvec![Str, Str], Int),
            // Split's second input can be a map of either type
            Split => {
                if let MapIntStr | MapStrStr = incoming[1] {
                    (smallvec![Str, incoming[1], Str], Int)
                } else {
                    return err!("invalid input spec for split: {:?}", &incoming[..]);
                }
            }
            JoinCols => (smallvec![Int, Int, Str], Str),
            JoinCSV | JoinTSV => (smallvec![Int, Int], Str),
            SetFI => (smallvec![Int, Int], Int),
        })
    }

    pub(crate) fn arity(&self) -> Option<usize> {
        use Function::*;
        Some(match self {
            FloatFunc(ff) => ff.arity(),
            IntFunc(bw) => bw.arity(),
            UpdateUsedFields | Rand | ReseedRng | ReadErrStdin | NextlineStdin | NextFile
            | ReadLineStdinFused => 0,
            ToUpper | ToLower | Clear | Srand | System | HexToInt | ToInt | EscapeCSV
            | EscapeTSV | Close | Length | ReadErr | ReadErrCmd | Nextline | NextlineCmd
            | Unop(_) => 1,
            SetFI | SubstrIndex | Match | Setcol | Binop(_) => 2,
            JoinCSV | JoinTSV | Delete | Contains => 2,
            IncMap | JoinCols | Substr | Sub | GSub | Split => 3,
        })
    }

    pub(crate) fn step(&self, args: &[types::State]) -> Result<types::State> {
        use {
            ast::{Binop::*, Unop::*},
            types::{BaseTy, TVar::*},
            Function::*,
        };
        fn step_arith(x: &types::State, y: &types::State) -> types::State {
            use BaseTy::*;
            match (x, y) {
                (Some(Scalar(Some(Str | Float))), _) | (_, Some(Scalar(Some(Str | Float)))) => {
                    Scalar(Float).abs()
                }
                (_, _) => Scalar(Int).abs(),
            }
        }
        match self {
            IntFunc(bw) => Ok(bw.ret_state()),
            FloatFunc(ff) => Ok(ff.ret_state()),
            Unop(Neg) | Unop(Pos) => match &args[0] {
                Some(Scalar(Some(BaseTy::Str))) | Some(Scalar(Some(BaseTy::Float))) => {
                    Ok(Scalar(BaseTy::Float).abs())
                }
                x => Ok(*x),
            },
            Binop(Plus) | Binop(Minus) | Binop(Mod) | Binop(Mult) => {
                Ok(step_arith(&args[0], &args[1]))
            }
            Rand | Binop(Div) | Binop(Pow) => Ok(Scalar(BaseTy::Float).abs()),
            Setcol => Ok(Scalar(BaseTy::Null).abs()),
            Clear | SubstrIndex | Srand | ReseedRng | Unop(Not) | Binop(IsMatch) | Binop(LT)
            | Binop(GT) | Binop(LTE) | Binop(GTE) | Binop(EQ) | Length | Split | ReadErr
            | ReadErrCmd | ReadErrStdin | Contains | Delete | Match | Sub | GSub | ToInt
            | System | HexToInt => Ok(Scalar(BaseTy::Int).abs()),
            ToUpper | ToLower | JoinCSV | JoinTSV | JoinCols | EscapeCSV | EscapeTSV | Substr
            | Unop(Column) | Binop(Concat) | Nextline | NextlineCmd | NextlineStdin => {
                Ok(Scalar(BaseTy::Str).abs())
            }
            IncMap => Ok(step_arith(&types::val_of(&args[0])?, &args[2])),
            SetFI | UpdateUsedFields | NextFile | ReadLineStdinFused | Close => Ok(None),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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
    ORS = 10,
    FNR = 11,
    PID = 12,
    FI = 13,
}

impl From<Variable> for compile::Ty {
    fn from(v: Variable) -> compile::Ty {
        use Variable::*;
        match v {
            FS | OFS | ORS | RS | FILENAME => compile::Ty::Str,
            PID | ARGC | NF | NR | FNR | RSTART | RLENGTH => compile::Ty::Int,
            ARGV => compile::Ty::MapIntStr,
            FI => compile::Ty::MapStrInt,
        }
    }
}

pub(crate) struct Variables<'a> {
    pub argc: Int,
    pub argv: IntMap<Str<'a>>,
    pub fs: Str<'a>,
    pub ofs: Str<'a>,
    pub ors: Str<'a>,
    pub rs: Str<'a>,
    pub nf: Int,
    pub nr: Int,
    pub fnr: Int,
    pub filename: Str<'a>,
    pub rstart: Int,
    pub rlength: Int,
    pub pid: Int,
    pub fi: StrMap<'a, Int>,
}

impl<'a> Default for Variables<'a> {
    fn default() -> Variables<'a> {
        Variables {
            argc: 0,
            argv: Default::default(),
            fs: " ".into(),
            ofs: " ".into(),
            ors: "\n".into(),
            rs: "\n".into(),
            nr: 0,
            fnr: 0,
            nf: 0,
            filename: Default::default(),
            rstart: 0,
            pid: 0,
            rlength: -1,
            fi: Default::default(),
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
            FNR => self.fnr,
            RSTART => self.rstart,
            RLENGTH => self.rlength,
            PID => self.pid,
            FI | ORS | OFS | FS | RS | FILENAME | ARGV => return err!("var {} not an int", var),
        })
    }

    pub fn store_int(&mut self, var: Variable, i: Int) -> Result<()> {
        use Variable::*;
        Ok(match var {
            ARGC => self.argc = i,
            NF => self.nf = i,
            NR => self.nr = i,
            FNR => self.fnr = i,
            RSTART => self.rstart = i,
            RLENGTH => self.rlength = i,
            PID => self.pid = i,
            FI | ORS | OFS | FS | RS | FILENAME | ARGV => return err!("var {} not an int", var),
        })
    }

    pub fn load_str(&self, var: Variable) -> Result<Str<'a>> {
        use Variable::*;
        Ok(match var {
            FS => self.fs.clone(),
            OFS => self.ofs.clone(),
            ORS => self.ors.clone(),
            RS => self.rs.clone(),
            FILENAME => self.filename.clone(),
            FI | PID | ARGC | ARGV | NF | NR | FNR | RSTART | RLENGTH => {
                return err!("var {} not a string", var)
            }
        })
    }

    pub fn store_str(&mut self, var: Variable, s: Str<'a>) -> Result<()> {
        use Variable::*;
        Ok(match var {
            FS => self.fs = s,
            OFS => self.ofs = s,
            ORS => self.ors = s,
            RS => self.rs = s,
            FILENAME => self.filename = s,
            FI | PID | ARGC | ARGV | NF | NR | FNR | RSTART | RLENGTH => {
                return err!("var {} not a string", var)
            }
        })
    }

    pub fn load_intmap(&self, var: Variable) -> Result<IntMap<Str<'a>>> {
        use Variable::*;
        match var {
            ARGV => Ok(self.argv.clone()),
            FI | PID | ORS | OFS | ARGC | NF | NR | FNR | FS | RS | FILENAME | RSTART | RLENGTH => {
                err!("var {} is not an int-keyed map", var)
            }
        }
    }

    pub fn store_intmap(&mut self, var: Variable, m: IntMap<Str<'a>>) -> Result<()> {
        use Variable::*;
        match var {
            ARGV => Ok(self.argv = m),
            FI | PID | ORS | OFS | ARGC | NF | NR | FNR | FS | RS | FILENAME | RSTART | RLENGTH => {
                err!("var {} is not an int-keyed map", var)
            }
        }
    }
    pub fn load_strmap(&self, var: Variable) -> Result<StrMap<'a, Int>> {
        use Variable::*;
        match var {
            FI => Ok(self.fi.clone()),
            ARGV | PID | ORS | OFS | ARGC | NF | NR | FNR | FS | RS | FILENAME | RSTART
            | RLENGTH => {
                err!("var {} is not a string-keyed map", var)
            }
        }
    }

    pub fn store_strmap(&mut self, var: Variable, m: StrMap<'a, Int>) -> Result<()> {
        use Variable::*;
        match var {
            FI => Ok(self.fi = m),
            ARGV | PID | ORS | OFS | ARGC | NF | NR | FNR | FS | RS | FILENAME | RSTART
            | RLENGTH => {
                err!("var {} is not a string-keyed map", var)
            }
        }
    }
}

impl Variable {
    pub(crate) fn ty(&self) -> types::TVar<types::BaseTy> {
        use Variable::*;
        match self {
            PID | ARGC | NF | FNR | NR | RSTART | RLENGTH => {
                types::TVar::Scalar(types::BaseTy::Int)
            }
            // NB: For full compliance, this may have to be Str -> Str
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
            //
            //  OTOH... maybe it's not so bad that we get type errors when putting strings as keys
            //  in ARGV.
            ARGV => types::TVar::Map {
                key: types::BaseTy::Int,
                val: types::BaseTy::Str,
            },
            FI => types::TVar::Map {
                key: types::BaseTy::Str,
                val: types::BaseTy::Int,
            },
            ORS | OFS | FS | RS | FILENAME => types::TVar::Scalar(types::BaseTy::Str),
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
            10 => Ok(ORS),
            11 => Ok(FNR),
            12 => Ok(PID),
            13 => Ok(FI),
            _ => Err(()),
        }
    }
}

static_map!(
    VARIABLES<&'static str, Variable>,
    ["ARGC", Variable::ARGC],
    ["ARGV", Variable::ARGV],
    ["OFS", Variable::OFS],
    ["ORS", Variable::ORS],
    ["FS", Variable::FS],
    ["RS", Variable::RS],
    ["NF", Variable::NF],
    ["NR", Variable::NR],
    ["FNR", Variable::FNR],
    ["FILENAME", Variable::FILENAME],
    ["RSTART", Variable::RSTART],
    ["RLENGTH", Variable::RLENGTH],
    ["PID", Variable::PID],
    ["FI", Variable::FI]
);
