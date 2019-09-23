#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum Unop {
    Column, // $
    Not,    // !
    Neg,    // -
    Pos,    // +
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

#[derive(Debug)]
pub(crate) enum Expr<'a, 'b, I> {
    ILit(i64),
    FLit(f64),
    StrLit(&'b str),
    Unop(Unop, &'a Expr<'a, 'b, I>),
    Binop(Binop, &'a Expr<'a, 'b, I>, &'a Expr<'a, 'b, I>),
    // TODO: add Call(&'b str, SmallVec<&'a Expr>). Have a static map of all builtin function
    // names. Use that to resolve scopes, etc.
    Var(I),
    Index(&'a Expr<'a, 'b, I>, &'a Expr<'a, 'b, I>),
    Assign(
        &'a Expr<'a, 'b, I>, /*var or index expression*/
        &'a Expr<'a, 'b, I>,
    ),
    AssignOp(&'a Expr<'a, 'b, I>, Binop, &'a Expr<'a, 'b, I>),
}

#[derive(Debug)]
pub(crate) enum Stmt<'a, 'b, I> {
    Expr(&'a Expr<'a, 'b, I>),
    Block(Vec<&'a Stmt<'a, 'b, I>>),
    // of course, Print can have 0 arguments. But let's handle that up the stack.
    Print(Vec<&'a Expr<'a, 'b, I>>, Option<&'a Expr<'a, 'b, I>>),
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
