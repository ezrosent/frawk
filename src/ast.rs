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

pub struct Prog<'a, 'b, I> {
    pub begin: Option<&'a Stmt<'a, 'b, I>>,
    pub end: Option<&'a Stmt<'a, 'b, I>>,
    pub pats: Vec<(Option<&'a Expr<'a, 'b, I>>, Option<&'a Stmt<'a, 'b, I>>)>,
}

impl<'a, 'b, I> Prog<'a, 'b, I> {
    pub(crate) fn desugar<'outer>(&self, arena: &'a Arena<'outer>) -> Stmt<'a, 'b, I> {
        use {self::Binop::*, self::Expr::*, Stmt::*};
        let mut res = vec![];

        if let Some(begin) = self.begin {
            res.push(begin);
        }

        // Desugar patterns into if statements, with the usual desugaring for an empty action.
        let mut inner = vec![];
        for (pat, body) in self.pats.iter() {
            let body = if let Some(body) = body {
                body
            } else {
                arena.alloc(|| Print(vec![], None))
            };
            if let Some(pat) = pat {
                inner.push(arena.alloc(|| If(pat, body, None)));
            } else {
                inner.push(body);
            }
        }

        if inner.len() > 0 {
            // Wrap the whole thing in a while((getline) > 0) { } statement.
            res.push(arena.alloc(move || {
                While(
                    arena.alloc(|| {
                        Binop(
                            GT,
                            arena.alloc(|| Getline {
                                into: None,
                                from: None,
                            }),
                            arena.alloc(|| ILit(0)),
                        )
                    }),
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
    Match,
    LT,
    GT,
    LTE,
    GTE,
    EQ,
}

// Once we have done this, it's time to add a parser. That will let us clean out a lot of bugs and
// get a more robust test suite as well.
//
// TODO add "length" -- works on strings and arrays, along with desugaring for length() =>
// length($0)
//   * for polymorphism, just exempt it from kind inference, and just generate separate
//   instructions
// TODO add support for "next"; just continue to the toplevel loop?
// TODO add "delete"
// TODO desugaring for "split"
// TODO add "close", make cache for regexes LRU.
// TODO add UDFs

static_map!(
    BINOPS<&'static str, Binop>,
    ["+", Binop::Plus],
    ["-", Binop::Minus],
    ["*", Binop::Mult],
    ["/", Binop::Div],
    ["%", Binop::Mod],
    ["", Binop::Concat], // we may have to handle this one specially
    ["~", Binop::Match],
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
}

#[derive(Debug, Clone)]
pub enum Stmt<'a, 'b, I> {
    Expr(&'a Expr<'a, 'b, I>),
    Block(Vec<&'a Stmt<'a, 'b, I>>),
    // of course, Print can have 0 arguments. But let's handle that up the stack.
    Print(
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
}
