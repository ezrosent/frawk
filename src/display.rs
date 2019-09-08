//! Noisey `Display` impls.
use crate::ast::{NumBinop, NumUnop, StrBinop, StrUnop};
use crate::cfg::{BasicBlock, Ident, PrimExpr, PrimStmt, PrimVal, Transition};
use std::fmt::{self, Display, Formatter};

struct Wrap(pub Ident);

impl Display for Wrap {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}-{}", (self.0).0, (self.0).1)
    }
}

impl<'a> Display for Transition<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match &self.0 {
            Some(v) => write!(f, "{}", v),
            None => write!(f, "else"),
        }
    }
}

impl<'a> Display for BasicBlock<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        for i in &self.0 {
            writeln!(f, "{}", i)?;
        }
        Ok(())
    }
}

impl<'a> Display for PrimStmt<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use PrimStmt::*;
        match self {
            Print(os, out) => {
                write!(f, "print ")?;
                for (i, o) in os.iter().enumerate() {
                    let is_last = i == os.len() - 1;
                    if is_last {
                        write!(f, "{}", o)?;
                    } else {
                        write!(f, "{}, ", o)?;
                    }
                }
                if let Some(out) = out {
                    write!(f, "> {}", out)?;
                }
                Ok(())
            }
            AsgnIndex(id, pv, pe) => write!(f, "{}[{}] = {}", Wrap(*id), pv, pe),
            AsgnVar(id, pe) => write!(f, "{} = {}", Wrap(*id), pe),
        }
    }
}

impl<'a> Display for PrimExpr<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use PrimExpr::*;
        match self {
            Val(v) => write!(f, "{}", v),
            Phi(preds) => {
                write!(f, "phi [")?;
                for (i, (pred, id)) in preds.iter().enumerate() {
                    let is_last = i == preds.len() - 1;
                    if is_last {
                        write!(f, "n@{}:{}", pred.index(), Wrap(*id))?
                    } else {
                        write!(f, "n@{}:{}, ", pred.index(), Wrap(*id))?
                    }
                }
                write!(f, "]")
            }
            StrUnop(u, o) => write!(f, "{}{}", u, o),
            StrBinop(b, o1, o2) => write!(f, "{} {} {}", o1, b, o2),
            NumUnop(u, o) => write!(f, "{}{}", u, o),
            NumBinop(b, o1, o2) => write!(f, "{} {} {}", o1, b, o2),
            Index(m, v) => write!(f, "{}[{}]", m, v),
            IterBegin(m) => write!(f, "begin({})", m),
            HasNext(i) => write!(f, "hasnext({})", i),
            Next(i) => write!(f, "next({})", i),
        }
    }
}

impl<'a> Display for PrimVal<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use PrimVal::*;
        match self {
            Var(id) => write!(f, "{}", Wrap(*id)),
            ILit(n) => write!(f, "{}@int", *n),
            FLit(n) => write!(f, "{}@float", *n),
            StrLit(s) => write!(f, "\"{}\"", s),
        }
    }
}

impl Display for NumUnop {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use NumUnop::*;
        write!(
            f,
            "{}",
            match self {
                Column => "$",
                Not => "!",
                Neg => "-",
                Pos => "+",
            }
        )
    }
}

impl Display for NumBinop {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use NumBinop::*;
        write!(
            f,
            "{}",
            match self {
                Plus => "+",
                Minus => "-",
                Mult => "*",
                Div => "/",
                Mod => "%",
            }
        )
    }
}

impl Display for StrBinop {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use StrBinop::*;
        write!(
            f,
            "{}",
            match self {
                Concat => "<concat>",
                Match => "~",
            }
        )
    }
}

impl Display for StrUnop {
    fn fmt(&self, _f: &mut Formatter) -> fmt::Result {
        match *self {}
    }
}
