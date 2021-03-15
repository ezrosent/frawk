/// An "Abstract Syntax Tree" that fairly closely resembles the structure of an AWK program. This
/// is the representation that the parser returns. A couple of basic desugaring rules are applied
/// that translate a `Prog` into a bare `Stmt`, along with its accompanying function definitions.
/// Those materials are consumed by the `cfg` module, which produces an untyped SSA form.
///
/// A couple things to note:
///  * The Expr and Stmt types are trees supporting arbitrary nesting. With limited exceptions, we
///    do not allocate each node separately on the heap. We instead use an arena for these
///    allocations. This is a win on all fronts: it's strictly faster because allocation is
///    extremely cheap, and destructors are fairly cheap, _and_ it's much easier to program because
///    references are Copy.
///  * The common way of introducing AWK: that it is a language structured around patterns and
///    actions to execute when the input matches that pattern is desugared in this module. We do
///    not handle it specially.
///
///    TODO It is not clear that this is the right move long-term: lots of regex implementations
///    (like HyperScan, or BurntSushi's engine in use here) achieve higher throughput by matching a
///    string against several patterns at once. There is probably a transormation we could do here
///    to take advantage of that, but it would probably involve building out def-use chains (which
///    we currently don't do), and we'd want to verify that performance didn't degrade when the
///    patterns are _not sparse_ in the input.
use crate::arena::Arena;
use crate::builtins::Function;
use crate::common::{Either, FileSpec, Stage};

use bumpalo::{collections::Vec as BumpVec, Bump};

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
    // FS
    pub field_sep: Option<&'b [u8]>,
    pub prelude_vardecs: Vec<(I, &'a Expr<'a, 'b, I>)>,
    // OFS
    pub output_sep: Option<&'b [u8]>,
    // ORS
    pub output_record_sep: Option<&'b [u8]>,
    pub decs: Vec<FunDec<'a, 'b, I>>,
    pub begin: Vec<&'a Stmt<'a, 'b, I>>,
    pub prepare: Vec<&'a Stmt<'a, 'b, I>>,
    pub end: Vec<&'a Stmt<'a, 'b, I>>,
    pub pats: Vec<(Pattern<'a, 'b, I>, Option<&'a Stmt<'a, 'b, I>>)>,
    pub stage: Stage<()>,
    pub argv: Vec<&'b str>,
    pub parse_header: bool,
}

fn parse_header<'a, 'b, I: From<&'b str> + Clone>(
    arena: &'a Bump,
    begin: &mut Vec<&'a Stmt<'a, 'b, I>>,
) {
    use {self::Expr::*, Stmt::*};
    // Pick an illegal frawk identifier.
    const LOOP_VAR: &'static str = "--";
    // Append the following to begin:
    // if (getline > 0) {
    //  for (LOOP_VAR=1; LOOP_VAR <= NF; ++LOOP_VAR)
    //      FI[$LOOP_VAR] = LOOP_VAR;
    //  update_used_fields()
    // }

    let loop_var = arena.alloc(Var(LOOP_VAR.into()));
    let init = arena.alloc(Expr(arena.alloc(Assign(loop_var, arena.alloc(ILit(1))))));
    let cond = arena.alloc(Binop(
        self::Binop::LTE,
        loop_var,
        arena.alloc(Var("NF".into())),
    ));
    let update = arena.alloc(Expr(arena.alloc(Inc {
        is_inc: true,
        is_post: false,
        x: loop_var,
    })));
    let body = arena.alloc(Expr(arena.alloc(Call(
        Either::Right(Function::SetFI),
        bumpalo::vec![in arena; loop_var as _, loop_var as _],
    ))));

    let block = bumpalo::vec![in arena;
        arena.alloc(For(Some(init), Some(cond), Some(update), body)) as _,
        arena.alloc(Expr(arena.alloc(Call(
            Either::Right(Function::UpdateUsedFields),
            BumpVec::new_in(arena),
        )))) as _,
    ];
    begin.push(arena.alloc(If(
        arena.alloc(Binop(
            self::Binop::GT,
            arena.alloc(ReadStdin),
            arena.alloc(ILit(0)),
        )),
        arena.alloc(Block(block)),
        /*else*/ None,
    )));
}

impl<'a, 'b, I: From<&'b str> + Clone> Prog<'a, 'b, I> {
    pub(crate) fn from_stage(stage: Stage<()>) -> Self {
        Prog {
            field_sep: None,
            prelude_vardecs: Default::default(),
            output_sep: None,
            output_record_sep: None,
            decs: Default::default(),
            begin: vec![],
            prepare: vec![],
            end: vec![],
            pats: Default::default(),
            argv: Default::default(),
            parse_header: false,
            stage,
        }
    }
    pub(crate) fn desugar_stage<'outer>(&self, arena: &'a Bump) -> Stage<&'a Stmt<'a, 'b, I>> {
        use {self::Binop::*, self::Expr::*, Stmt::*};
        let mut conds = 0;

        let mut begin = Vec::<&_>::with_capacity(self.begin.len());
        let mut main_loop = None;
        let mut end = None;

        // Desugar -F flag
        if let Some(sep) = self.field_sep {
            begin.push(arena.alloc(Expr(arena.alloc(Assign(
                arena.alloc(Var("FS".into())),
                arena.alloc(StrLit(sep)),
            )))));
        }

        // for -H
        if self.parse_header {
            parse_header(arena, &mut begin);
        }

        // Support "output csv/tsv" mode
        if let Some(sep) = self.output_sep {
            begin.push(arena.alloc(Expr(arena.alloc(Assign(
                arena.alloc(Var("OFS".into())),
                arena.alloc(StrLit(sep)),
            )))));
        }
        if let Some(sep) = self.output_record_sep {
            begin.push(arena.alloc(Expr(arena.alloc(Assign(
                arena.alloc(Var("ORS".into())),
                arena.alloc(StrLit(sep)),
            )))));
        }

        // Assign SUBSEP, which we treat as a normal variable
        begin.push(arena.alloc(Expr(arena.alloc(Assign(
            arena.alloc(Var("SUBSEP".into())),
            arena.alloc(StrLit(&[0o034u8])),
        )))));
        // Desugar -v flags
        for (ident, exp) in self.prelude_vardecs.iter() {
            begin.push(arena.alloc(Expr(
                arena.alloc(Assign(arena.alloc(Var(ident.clone())), exp)),
            )));
        }

        // Set argc, argv
        if self.argv.len() > 0 {
            begin.push(arena.alloc(Expr(arena.alloc(Assign(
                arena.alloc(Var("ARGC".into())),
                arena.alloc(ILit(self.argv.len() as i64)),
            )))));
            let argv = arena.alloc(Var("ARGV".into()));
            for (ix, arg) in self.argv.iter().enumerate() {
                let arg = arena.alloc(StrLit(arg.as_bytes()));
                let ix = arena.alloc(ILit(ix as i64));
                let arr_exp = arena.alloc(Index(argv, ix));
                begin.push(arena.alloc(Expr(arena.alloc(Assign(arr_exp, arg)))));
            }
        }

        begin.extend(self.begin.iter().cloned());

        // Desugar patterns into if statements, with the usual desugaring for an empty action.
        let mut inner: BumpVec<&Stmt<I>> = bumpalo::vec![in arena;
            arena.alloc(Expr(arena.alloc(Inc {
                is_inc: true,
                is_post: false,
                x: arena.alloc(Var("NR".into())),
            }))) as &_,
            arena.alloc(Expr(arena.alloc(Inc {
                is_inc: true,
                is_post: false,
                x: arena.alloc(Var("FNR".into())),
            }))) as &_,
        ];
        let init_len = inner.len();
        for (pat, body) in self.pats.iter() {
            let body = if let Some(body) = body {
                *body
            } else {
                arena.alloc(Print(BumpVec::new_in(arena), None)) as &_
            };
            match pat {
                Pattern::Null => inner.push(body),
                Pattern::Bool(pat) => inner.push(arena.alloc(If(pat, body, None))),
                Pattern::Comma(l, r) => {
                    // Comma patterns run the corresponding action between pairs of lines matching
                    // patterns `l` and `r`, inclusive. One common example is the patterh
                    //  /\/*/,/*\//
                    // Matching block comments in several propular languages. We desugar these with
                    // special statements StartCond, LastCond, EndCond, as well as the Cond
                    // expression. Each of these is tagged with an identifier indicating which
                    // comma pattern the statement or expression is referencing. In the cfg module,
                    // these are compiled to simple assignments and reads on a pattern-specific
                    // local variable:
                    //
                    // StartCond sets the variable to 1
                    // EndCond sets the variable to 0
                    // LastCond stes the variable to 2
                    // Cond reads the variable.
                    //
                    // Why do you need LastCond? We can mostly make due without it. Consider the
                    // fragment:
                    //   /\/*/,/*\// { i++ }
                    // We can desugar this quite easily as:
                    //   /\/*/ { StartCond(0); } # _cond_0 = 1
                    //   Cond(0) { i++; }        # if _cond_0
                    //   /*\// { EndCond(0); }   # _cond_0 = 0
                    // We get into trouble, however, if control flow is more complex. If we wanted
                    // to strip comments we might write:
                    //   /\/*/,/*\// { next; }
                    //   { print; }
                    // But applying the above desugaring rules leads us to running `next` before we
                    // can set EndCond. No output will be printed after we encounter our first
                    // comment, regardless of what comes after.
                    //
                    // To fix this, we introduce LastCond and use it to signal that the pattern
                    // should not match in the next iteration.
                    //   /\/*/ { StartCond(0); } # _cond_0 = 1
                    //   /*\// { LastCond(0); }  # _cond_0 = 2
                    //   Cond(0) {
                    //      # We matched the end of a comment. End the condition before `next`;
                    //      if (Cond(0) == 2) EndCond(0); # _cond_0 = 0;
                    //      next;
                    //  }
                    inner.push(arena.alloc(If(l, arena.alloc(StartCond(conds)), None)));
                    inner.push(arena.alloc(If(r, arena.alloc(LastCond(conds)), None)));
                    let block: BumpVec<&Stmt<I>> = bumpalo::vec![in arena;
                        arena.alloc(If(
                            arena.alloc(Binop(EQ, arena.alloc(Cond(conds)), arena.alloc(ILit(2)))),
                            arena.alloc(EndCond(conds)),
                            None,
                        )) as &_,
                        body,
                    ];
                    inner.push(arena.alloc(If(
                        arena.alloc(Cond(conds)),
                        arena.alloc(Block(block)),
                        None,
                    )));
                    conds += 1;
                }
            }
        }

        if self.end.len() > 0 || self.prepare.len() > 0 || inner.len() > init_len {
            // Wrap the whole thing in a while((getline) > 0) { } statement.
            let main_portion: &_ = arena.alloc(While(
                /*is_toplevel=*/ true,
                arena.alloc(Binop(GT, arena.alloc(ReadStdin), arena.alloc(ILit(0)))) as &_,
                arena.alloc(Block(inner)) as &_,
            ));
            main_loop = Some(if self.prepare.len() > 0 {
                let mut block = BumpVec::with_capacity_in(self.prepare.len() + 1, arena);
                block.push(main_portion);
                block.extend(self.prepare.iter().cloned());
                arena.alloc(Stmt::Block(block))
            } else {
                main_portion
            });
        }
        if self.end.len() > 0 {
            let mut block = BumpVec::with_capacity_in(self.end.len(), arena);
            block.extend(self.end.iter().cloned());
            end = Some(arena.alloc(Stmt::Block(block)) as &_);
        }
        let mut begin_block = BumpVec::with_capacity_in(self.begin.len(), arena);
        begin_block.extend(self.begin.iter().cloned());
        match self.stage {
            Stage::Main(_) => {
                begin.extend(main_loop.into_iter().chain(end));
                Stage::Main(arena.alloc(Stmt::Block(begin_block)))
            }
            Stage::Par { .. } => Stage::Par {
                begin: if begin.len() > 0 {
                    Some(arena.alloc(Stmt::Block(begin_block)))
                } else {
                    None
                },
                main_loop,
                end,
            },
        }
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
    Pow,
    LT,
    GT,
    LTE,
    GTE,
    EQ,
}

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
    ["==", Binop::EQ],
    ["^", Binop::Pow]
);

#[derive(Debug, Clone)]
pub enum Expr<'a, 'b, I> {
    ILit(i64),
    FLit(f64),
    StrLit(&'b [u8]),
    PatLit(&'b [u8]),
    Unop(Unop, &'a Expr<'a, 'b, I>),
    Binop(Binop, &'a Expr<'a, 'b, I>, &'a Expr<'a, 'b, I>),
    Call(Either<I, Function>, BumpVec<'a, &'a Expr<'a, 'b, I>>),
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
        is_file: bool,
    },
    ReadStdin,
    // Used for comma patterns
    Cond(usize),
}

#[derive(Debug, Clone)]
pub enum Stmt<'a, 'b, I> {
    StartCond(usize),
    EndCond(usize),
    LastCond(usize),
    Expr(&'a Expr<'a, 'b, I>),
    Block(BumpVec<'a, &'a Stmt<'a, 'b, I>>),
    Print(
        BumpVec<'a, &'a Expr<'a, 'b, I>>,
        Option<(&'a Expr<'a, 'b, I>, FileSpec)>,
    ),
    // Unlike print, printf must have at least one argument.
    Printf(
        &'a Expr<'a, 'b, I>,
        BumpVec<'a, &'a Expr<'a, 'b, I>>,
        Option<(&'a Expr<'a, 'b, I>, FileSpec)>,
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
    // We mark some while loops as "special" because of the special "next" and "nextfile" commands,
    // that work as a special "labelled continue" for the toplevel loop.
    While(
        /*is_toplevel*/ bool,
        &'a Expr<'a, 'b, I>,
        &'a Stmt<'a, 'b, I>,
    ),
    ForEach(I, &'a Expr<'a, 'b, I>, &'a Stmt<'a, 'b, I>),
    Break,
    Continue,
    Next,
    NextFile,
    Return(Option<&'a Expr<'a, 'b, I>>),
}
