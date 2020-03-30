//! Noisey `Display` impls.
use crate::ast::{Binop, Unop};
use crate::builtins::{Function, Variable};
use crate::cfg::{BasicBlock, Ident, PrimExpr, PrimStmt, PrimVal, Transition};
use std::fmt::{self, Display, Formatter};

pub(crate) struct Wrap(pub Ident);

impl Display for Wrap {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let Ident { low, sub, .. } = self.0;
        write!(f, "{}-{}", low, sub)
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
        for i in &self.q {
            writeln!(f, "{}", i)?;
        }
        Ok(())
    }
}

impl<'a> Display for PrimStmt<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use PrimStmt::*;
        match self {
            AsgnIndex(id, pv, pe) => write!(f, "{}[{}] = {}", Wrap(*id), pv, pe),
            AsgnVar(id, pe) => write!(f, "{} = {}", Wrap(*id), pe),
            SetBuiltin(v, pv) => write!(f, "{} = {}", v, pv),
            Return(v) => write!(f, "return {}", v),
            Printf(fmt, args, out) => {
                write!(f, "printf({}", fmt)?;
                for (i, a) in args.iter().enumerate() {
                    if i == args.len() - 1 {
                        write!(f, "{}", a)?;
                    } else {
                        write!(f, "{}, ", a)?;
                    }
                }
                write!(f, ")")?;
                if let Some((out, ap)) = out {
                    write!(f, " {} {}", out, if *ap { ">>" } else { ">" })?;
                }
                Ok(())
            }
            IterDrop(v) => write!(f, "drop_iter {}", v),
        }
    }
}

impl<'a> Display for PrimExpr<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use PrimExpr::*;
        fn write_func(
            f: &mut Formatter,
            func: impl fmt::Display,
            os: &[impl fmt::Display],
        ) -> fmt::Result {
            write!(f, "{}(", func)?;
            for (i, o) in os.iter().enumerate() {
                let is_last = i == os.len() - 1;
                if is_last {
                    write!(f, "{}", o)?;
                } else {
                    write!(f, "{}, ", o)?;
                }
            }
            write!(f, ")")
        }
        match self {
            Val(v) => write!(f, "{}", v),
            Phi(preds) => {
                write!(f, "phi [")?;
                for (i, (pred, id)) in preds.iter().enumerate() {
                    let is_last = i == preds.len() - 1;
                    if is_last {
                        write!(f, "←{}:{}", pred.index(), Wrap(*id))?
                    } else {
                        write!(f, "←{}:{}, ", pred.index(), Wrap(*id))?
                    }
                }
                write!(f, "]")
            }
            CallBuiltin(b, os) => write_func(f, b, &os[..]),
            CallUDF(func, os) => write_func(f, func, &os[..]),
            Sprintf(fmt, os) => write_func(f, format!("sprintf[{}]", fmt), &os[..]),
            Index(m, v) => write!(f, "{}[{}]", m, v),
            IterBegin(m) => write!(f, "begin({})", m),
            HasNext(i) => write!(f, "hasnext({})", i),
            Next(i) => write!(f, "next({})", i),
            LoadBuiltin(b) => write!(f, "{}", b),
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

impl Display for Function {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use Function::*;
        match self {
            Unop(u) => write!(f, "{}", u),
            Binop(b) => write!(f, "{}", b),
            FloatFunc(ff) => write!(f, "{}", ff.func_name()),
            Print => write!(f, "print"),
            PrintStdout => write!(f, "print(stdout)"),
            ReadErr => write!(f, "hasline"),
            Nextline => write!(f, "nextline"),
            ReadErrStdin => write!(f, "hasline(stdin)"),
            NextlineStdin => write!(f, "nextline(stdin)"),
            ReadLineStdinFused => write!(f, "stdin-fused"),
            NextFile => write!(f, "nextfile"),
            Setcol => write!(f, "$="),
            Split => write!(f, "split"),
            Length => write!(f, "length"),
            Contains => write!(f, "contains"),
            Delete => write!(f, "delete"),
            Close => write!(f, "close"),
            Match => write!(f, "match"),
            Sub => write!(f, "sub"),
            GSub => write!(f, "gsub"),
            Substr => write!(f, "substr"),
        }
    }
}

impl Display for Variable {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use Variable::*;
        write!(
            f,
            "{}",
            match self {
                ARGC => "ARGC",
                ARGV => "ARGV",
                OFS => "OFS",
                ORS => "ORS",
                FS => "FS",
                RS => "RS",
                NF => "NF",
                NR => "NR",
                FNR => "FNR",
                FILENAME => "FILENAME",
                RSTART => "RSTART",
                RLENGTH => "RLENGTH",
            }
        )
    }
}

impl Display for Unop {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use Unop::*;
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

impl Display for Binop {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        use Binop::*;
        write!(
            f,
            "{}",
            match self {
                Plus => "+",
                Minus => "-",
                Mult => "*",
                Div => "/",
                Mod => "%",
                Concat => "<concat>",
                IsMatch => "~",
                LT => "<",
                GT => ">",
                LTE => "<=",
                GTE => ">=",
                EQ => "==",
            }
        )
    }
}
