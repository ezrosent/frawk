use crate::arena::Arena;
use crate::builtins::Function;
use crate::common::Either;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Unop {
    Column,
    Not,
    Neg,
    Pos,
}

static_map!(
    UNOPS<&'static str, Unop>,
    ["$", Unop::Column],
    ["!", Unop::Not],
    ["-", Unop::Neg],
    ["+", Unop::Pos]
);

pub struct FunDec<'a, 'b, I> {
    pub name: I,
    pub args: Vec<I>,
    pub body: &'a Stmt<'a, 'b, I>,
}

pub enum Pattern<'a, 'b, I> {
    Null,
    Bool(&'a Expr<'a, 'b, I>),
    Comma(&'a Expr<'a, 'b, I>, &'a Expr<'a, 'b, I>),
}

pub struct Prog<'a, 'b, I> {
    pub field_sep: Option<&'b str>,
    pub prelude_vardecs: Vec<(I, &'a Expr<'a, 'b, I>)>,
    pub decs: Vec<FunDec<'a, 'b, I>>,
    pub begin: Option<&'a Stmt<'a, 'b, I>>,
    pub end: Option<&'a Stmt<'a, 'b, I>>,
    pub pats: Vec<(Pattern<'a, 'b, I>, Option<&'a Stmt<'a, 'b, I>>)>,
}

impl<'a, 'b, I: From<&'b str> + Clone> Prog<'a, 'b, I> {
    pub(crate) fn desugar<'outer>(&self, arena: &'a Arena<'outer>) -> Stmt<'a, 'b, I> {
        use {self::Binop::*, self::Expr::*, Stmt::*};
        let mut conds = 0;
        let mut res = vec![];

        // Desugar -F flag
        if let Some(sep) = self.field_sep {
            res.push(arena.alloc_v(Expr(arena.alloc_v(Assign(
                arena.alloc_v(Var("FS".into())),
                arena.alloc_v(StrLit(sep)),
            )))));
        }
        // Desugar -v flags
        for (ident, exp) in self.prelude_vardecs.iter() {
            res.push(arena.alloc_v(Expr(
                arena.alloc_v(Assign(arena.alloc_v(Var(ident.clone())), exp)),
            )));
        }
        if let Some(begin) = self.begin {
            res.push(begin);
        }

        // Desugar patterns into if statements, with the usual desugaring for an empty action.
        let mut inner = vec![
            arena.alloc_v(Expr(arena.alloc_v(Inc {
                is_inc: true,
                is_post: false,
                x: arena.alloc_v(Var("NR".into())),
            }))),
            arena.alloc_v(Expr(arena.alloc_v(Inc {
                is_inc: true,
                is_post: false,
                x: arena.alloc_v(Var("FNR".into())),
            }))),
        ];
        let init_len = inner.len();
        for (pat, body) in self.pats.iter() {
            let body = if let Some(body) = body {
                body
            } else {
                arena.alloc_v(Print(vec![], None))
            };
            match pat {
                Pattern::Null => inner.push(body),
                Pattern::Bool(pat) => inner.push(arena.alloc_v(If(pat, body, None))),
                Pattern::Comma(l, r) => {
                    // We desugar pat1,pat2
                    // TODO: This doesn't totally work as a desugaring,
                    // If we had
                    //   /\/*/,/*\// { comment++; next; }
                    // we would never finish the comment because `next` would bail out before
                    // EndCond. Something to consider once we add `next` support.
                    inner.push(arena.alloc_v(If(l, arena.alloc_v(StartCond(conds)), None)));
                    inner.push(arena.alloc_v(If(arena.alloc_v(Cond(conds)), body, None)));
                    inner.push(arena.alloc_v(If(r, arena.alloc_v(EndCond(conds)), None)));
                    conds += 1;
                }
            }
        }

        if self.end.is_some() || inner.len() > init_len {
            // Wrap the whole thing in a while((getline) > 0) { } statement.
            res.push(arena.alloc(move || {
                While(
                    arena.alloc(|| Binop(GT, arena.alloc(|| ReadStdin), arena.alloc(|| ILit(0)))),
                    arena.alloc(move || Block(inner)),
                )
            }));
        }

        if let Some(end) = self.end {
            res.push(end);
        }

        Stmt::Block(res)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Binop {
    Plus,
    Minus,
    Mult,
    Div,
    Mod,
    Concat,
    IsMatch,
    LT,
    GT,
    LTE,
    GTE,
    EQ,
}

// Features:
// TODO printf
// TODO add support for "next"; just continue to the toplevel loop -- annotate while loop?
// TODO add "close", make cache for regexes LRU.
// TODO trig functions, !=, any missing operators.
// TODO CLI
// TODO multiple files
// TODO full /pat1/../pat2/ patterns
//
// Improvements:
// * Remove `Vec`s in ASTs. This may be hard for lalrpop for now, but we should at least be able to
//   move some of the Printf nodes onto an arena.

static_map!(
    BINOPS<&'static str, Binop>,
    ["+", Binop::Plus],
    ["-", Binop::Minus],
    ["*", Binop::Mult],
    ["/", Binop::Div],
    ["%", Binop::Mod],
    ["", Binop::Concat], // we may have to handle this one specially
    ["~", Binop::IsMatch],
    ["<", Binop::LT],
    [">", Binop::GT],
    ["<=", Binop::LTE],
    [">=", Binop::GTE],
    ["==", Binop::EQ]
);

#[derive(Debug, Clone)]
pub enum Expr<'a, 'b, I> {
    ILit(i64),
    FLit(f64),
    StrLit(&'b str),
    PatLit(&'b str),
    Unop(Unop, &'a Expr<'a, 'b, I>),
    Binop(Binop, &'a Expr<'a, 'b, I>, &'a Expr<'a, 'b, I>),
    Call(Either<I, Function>, Vec<&'a Expr<'a, 'b, I>>),
    Var(I),
    Index(&'a Expr<'a, 'b, I>, &'a Expr<'a, 'b, I>),
    Assign(
        &'a Expr<'a, 'b, I>, /*var or index expression*/
        &'a Expr<'a, 'b, I>,
    ),
    AssignOp(&'a Expr<'a, 'b, I>, Binop, &'a Expr<'a, 'b, I>),
    And(&'a Expr<'a, 'b, I>, &'a Expr<'a, 'b, I>),
    Or(&'a Expr<'a, 'b, I>, &'a Expr<'a, 'b, I>),
    ITE(
        &'a Expr<'a, 'b, I>,
        &'a Expr<'a, 'b, I>,
        &'a Expr<'a, 'b, I>,
    ),
    Inc {
        is_inc: bool,
        is_post: bool,
        x: &'a Expr<'a, 'b, I>,
    },
    Getline {
        into: Option<&'a Expr<'a, 'b, I>>,
        from: Option<&'a Expr<'a, 'b, I>>,
    },
    ReadStdin,
    // Used for comma patterns
    Cond(usize),
}

#[derive(Debug, Clone)]
pub enum Stmt<'a, 'b, I> {
    StartCond(usize),
    EndCond(usize),
    Expr(&'a Expr<'a, 'b, I>),
    Block(Vec<&'a Stmt<'a, 'b, I>>),
    Print(
        Vec<&'a Expr<'a, 'b, I>>,
        Option<(&'a Expr<'a, 'b, I>, bool /*append*/)>,
    ),
    // Unlike print, printf must have at least one argument.
    Printf(
        &'a Expr<'a, 'b, I>,
        Vec<&'a Expr<'a, 'b, I>>,
        Option<(&'a Expr<'a, 'b, I>, bool /*append*/)>,
    ),
    If(
        &'a Expr<'a, 'b, I>,
        &'a Stmt<'a, 'b, I>,
        Option<&'a Stmt<'a, 'b, I>>,
    ),
    For(
        Option<&'a Stmt<'a, 'b, I>>,
        Option<&'a Expr<'a, 'b, I>>,
        Option<&'a Stmt<'a, 'b, I>>,
        &'a Stmt<'a, 'b, I>,
    ),
    DoWhile(&'a Expr<'a, 'b, I>, &'a Stmt<'a, 'b, I>),
    While(&'a Expr<'a, 'b, I>, &'a Stmt<'a, 'b, I>),
    ForEach(I, &'a Expr<'a, 'b, I>, &'a Stmt<'a, 'b, I>),
    Break,
    Continue,
    Return(Option<&'a Expr<'a, 'b, I>>),
}
