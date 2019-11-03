#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum Unop {
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

pub(crate) struct Prog<'a, 'b, I> {
    begin: Option<Stmt<'a, 'b, I>>,
    end: Option<Stmt<'a, 'b, I>>,
    pats: Vec<(Option<Expr<'a, 'b, I>>, Option<Stmt<'a, 'b, I>>)>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum Binop {
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

// TODO add notion of pattern and action, which are desugared into a Vec<Stmt>
// TODO add "getline" desugaring
// TODO add c-style do-while loop
// TODO add pattern desugaring
// TODO add "length" -- works on strings and arrays.
// TODO add support for "next"; just continue to the toplevel loop?
// TODO add "delete"
// TODO add "in" -- maybe not an operator, just a builtin
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
pub(crate) enum Expr<'a, 'b, I> {
    ILit(i64),
    FLit(f64),
    StrLit(&'b str),
    Unop(Unop, &'a Expr<'a, 'b, I>),
    Binop(Binop, &'a Expr<'a, 'b, I>, &'a Expr<'a, 'b, I>),
    Call(I, Vec<&'a Expr<'a, 'b, I>>),
    Var(I),
    Index(&'a Expr<'a, 'b, I>, &'a Expr<'a, 'b, I>),
    Assign(
        &'a Expr<'a, 'b, I>, /*var or index expression*/
        &'a Expr<'a, 'b, I>,
    ),
    AssignOp(&'a Expr<'a, 'b, I>, Binop, &'a Expr<'a, 'b, I>),
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
pub(crate) enum Stmt<'a, 'b, I> {
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
    While(&'a Expr<'a, 'b, I>, &'a Stmt<'a, 'b, I>),
    ForEach(I, &'a Expr<'a, 'b, I>, &'a Stmt<'a, 'b, I>),
    Break,
    Continue,
}
