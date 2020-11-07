//! Noisey `Display` impls.
use crate::ast::{Binop, Unop};
use crate::builtins::{Function, Variable};
use crate::cfg::{BasicBlock, Ident, PrimExpr, PrimStmt, PrimVal, Transition};
use crate::common::FileSpec;
use crate::lexer;
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
                    let redirect = match ap {
                        FileSpec::Trunc => ">",
                        FileSpec::Append => ">>",
                        FileSpec::Cmd => "|",
                    };
                    write!(f, " {} {}", out, redirect)?;
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
            IntFunc(bw) => write!(f, "{}", bw.func_name()),
            Print => write!(f, "print"),
            PrintStdout => write!(f, "print(stdout)"),
            ReadErr => write!(f, "hasline"),
            ReadErrCmd => write!(f, "hasline(cmd)"),
            Nextline => write!(f, "nextline"),
            NextlineCmd => write!(f, "nextline(cmd)"),
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
            SubstrIndex => write!(f, "index"),
            Sub => write!(f, "sub"),
            GSub => write!(f, "gsub"),
            EscapeCSV => write!(f, "escape_csv"),
            EscapeTSV => write!(f, "escape_tsv"),
            JoinCSV => write!(f, "join_csv"),
            JoinTSV => write!(f, "join_tsv"),
            JoinCols => write!(f, "join_fields"),
            Substr => write!(f, "substr"),
            ToInt => write!(f, "int"),
            HexToInt => write!(f, "hex"),
            Rand => write!(f, "rand"),
            Srand => write!(f, "srand"),
            ReseedRng => write!(f, "srand_reseed"),
            System => write!(f, "system"),
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
                PID => "PID",
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
                Pow => "^",
                LT => "<",
                GT => ">",
                LTE => "<=",
                GTE => ">=",
                EQ => "==",
            }
        )
    }
}

impl Display for lexer::Loc {
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        write!(fmt, "line {}, column {}", self.line + 1, self.col + 1)
    }
}

impl Display for lexer::Error {
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        write!(fmt, "{}. {}", self.location, self.desc)
    }
}

impl<'a> Display for lexer::Tok<'a> {
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        use lexer::Tok::*;
        let rep = match self {
            Begin => "BEGIN",
            Prepare => "PREPARE",
            End => "END",
            Break => "break",
            Continue => "continue",
            Next => "next",
            NextFile => "nextfile",
            For => "for",
            If => "if",
            Else => "else",
            Print => "print",
            Printf => "printf",
            // Separate token for a "print(" and "printf(".
            PrintLP => "print(",
            PrintfLP => "printf(",
            While => "while",
            Do => "do",

            // { }
            LBrace => "{",
            RBrace => "}",
            // [ ]
            LBrack => "[",
            RBrack => "]",
            // ( )
            LParen => "(",
            RParen => ")",

            Getline => "getline",
            Pipe => "|",
            Assign => "=",
            Add => "+",
            AddAssign => "+=",
            Sub => "-",
            SubAssign => "-=",
            Mul => "*",
            MulAssign => "*=",
            Div => "/",
            DivAssign => "/=",
            Pow => "^",
            PowAssign => "^=",
            Mod => "%",
            ModAssign => "%=",
            Match => "~",
            NotMatch => "!~",

            EQ => "==",
            NEQ => "!=",
            LT => "<",
            GT => ">",
            LTE => "<=",
            GTE => ">=",
            Incr => "++",
            Decr => "--",
            Not => "!",

            AND => "&&",
            OR => "||",
            QUESTION => "?",
            COLON => ":",

            Append => ">>",

            Dollar => "$",
            Semi => ";",
            Newline => "\\n",
            Comma => ",",
            In => "in",
            Delete => "delete",
            Return => "return",

            Ident(s) => return write!(fmt, "identifier({})", s),
            StrLit(s) => return write!(fmt, "{:?}", s),
            PatLit(s) => return write!(fmt, "/{}/", s),
            CallStart(s) => return write!(fmt, "{}(", s),
            FunDec(s) => return write!(fmt, "function {}", s),

            ILit(s) | HexLit(s) | FLit(s) => return write!(fmt, "{}", s),
        };
        write!(fmt, "{}", rep)
    }
}
