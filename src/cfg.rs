use crate::ast::{self, Binop, Expr, Stmt, Unop};
use crate::builtins;
use crate::common::{CompileError, Either, Graph, NodeIx, NumTy, Result};
use crate::dom;

use hashbrown::{HashMap, HashSet};
use petgraph::Direction;
use smallvec::smallvec; // macro

use std::collections::VecDeque;
use std::convert::TryFrom;
use std::hash::Hash;

// TODO: rename context to something more descriptive. Maybe split out some structs.

// consider making this just "by number" and putting branch instructions elsewhere.
// need to verify the order
// Use VecDequeue to support things like prepending definitions and phi statements to blocks during
// SSA conversion.
#[derive(Debug, Default)]
pub(crate) struct BasicBlock<'a>(pub VecDeque<PrimStmt<'a>>);
#[derive(Debug, Default)]
pub(crate) struct Transition<'a>(pub Option<PrimVal<'a>>);

impl<'a> Transition<'a> {
    fn new(pv: PrimVal<'a>) -> Transition<'a> {
        Transition(Some(pv))
    }
    fn null() -> Transition<'a> {
        Transition(None)
    }
}

// None indicates `else`
pub(crate) type CFG<'a> = Graph<BasicBlock<'a>, Transition<'a>>;
pub(crate) type Ident = (NumTy, NumTy);
pub(crate) type SmallVec<T> = smallvec::SmallVec<[T; 4]>;

#[derive(Debug, Clone)]
pub(crate) enum PrimVal<'a> {
    Var(Ident),
    ILit(i64),
    FLit(f64),
    StrLit(&'a str),
}

#[derive(Debug, Clone)]
pub(crate) enum PrimExpr<'a> {
    Val(PrimVal<'a>),
    Phi(SmallVec<(NodeIx /* pred */, Ident)>),
    CallBuiltin(builtins::Function, SmallVec<PrimVal<'a>>),
    Index(PrimVal<'a>, PrimVal<'a>),

    // For iterating over vectors.
    // TODO: make these builtins? Unfortunately, IterBegin returns an iterator...
    IterBegin(PrimVal<'a>),
    HasNext(PrimVal<'a>),
    Next(PrimVal<'a>),
    LoadBuiltin(builtins::Variable),
}

#[derive(Debug)]
pub(crate) enum PrimStmt<'a> {
    AsgnIndex(
        Ident,        /*map*/
        PrimVal<'a>,  /* index */
        PrimExpr<'a>, /* assign to */
    ),
    AsgnVar(Ident /* var */, PrimExpr<'a>),
    SetBuiltin(builtins::Variable, PrimExpr<'a>),
}

// only add constraints when doing an AsgnVar. Because these things are "shallow" it works.
// Maybe also keep an auxiliary map from Var => node in graph, also allow Key(map_ident)
// Val(map_ident).
//
// Build up network. then solve. then use that to insert conversions when producing bytecode.

impl<'a> PrimVal<'a> {
    fn replace(&mut self, mut update: impl FnMut(Ident) -> Ident) {
        if let PrimVal::Var(ident) = self {
            *ident = update(*ident)
        }
    }
}

impl<'a> PrimExpr<'a> {
    fn replace(&mut self, mut update: impl FnMut(Ident) -> Ident) {
        use PrimExpr::*;
        match self {
            Val(v) => v.replace(update),
            Phi(_) => {}
            CallBuiltin(_, args) => {
                for a in args.iter_mut() {
                    a.replace(&mut update)
                }
            }
            Index(v1, v2) => {
                v1.replace(&mut update);
                v2.replace(update);
            }
            IterBegin(v) => v.replace(update),
            HasNext(v) => v.replace(update),
            Next(v) => v.replace(update),
            LoadBuiltin(_) => {}
        }
    }
}

impl<'a> PrimStmt<'a> {
    fn replace(&mut self, mut update: impl FnMut(Ident) -> Ident) {
        use PrimStmt::*;
        match self {
            AsgnIndex(ident, v, exp) => {
                *ident = update(*ident);
                v.replace(&mut update);
                exp.replace(update);
            }
            // We handle assignments separately. Note that this is not needed for index
            // expressions, because assignments to m[k] are *uses* of m, not definitions.
            AsgnVar(_, e) => e.replace(update),
            SetBuiltin(_, e) => e.replace(update),
        }
    }
}

fn valid_lhs<'a, 'b, I>(e: &ast::Expr<'a, 'b, I>) -> bool {
    use ast::Expr::*;
    match e {
        Index(_, _) | Var(_) | Unop(ast::Unop::Column, _) => true,
        _ => false,
    }
}

pub(crate) struct Context<'b, I> {
    hm: HashMap<I, Ident>,
    defsites: HashMap<Ident, HashSet<NodeIx>>,
    orig: HashMap<NodeIx, HashSet<Ident>>,
    max: NumTy,
    cfg: CFG<'b>,
    entry: NodeIx,
    // Stack of the entry and exit nodes for the loops within which the current statement is nested.
    loop_ctx: SmallVec<(NodeIx, NodeIx)>,

    // Dominance information about `cfg`.
    dt: dom::Tree,
    df: dom::Frontier,

    num_idents: usize,
    // variable that holds the FS variable, if needed
    tmp_fs: Option<Ident>,
}

impl<'b, I> Context<'b, I> {
    pub(crate) fn cfg<'a>(&'a self) -> &'a CFG<'b> {
        &self.cfg
    }
    pub(crate) fn num_idents(&self) -> usize {
        self.num_idents
    }
    pub(crate) fn entry(&self) -> NodeIx {
        self.entry
    }
}

pub(crate) fn is_unused(i: Ident) -> bool {
    i.0 == 0
}

impl<'b, I: Hash + Eq + Clone + Default + std::fmt::Display + std::fmt::Debug> Context<'b, I>
where
    builtins::Variable: TryFrom<I>,
    builtins::Function: TryFrom<I>,
{
    fn field_sep(&mut self) -> Ident {
        if let Some(id) = self.tmp_fs {
            id
        } else {
            let n = self.fresh();
            self.tmp_fs = Some(n);
            n
        }
    }
    fn unused() -> Ident {
        (0, 0)
    }
    pub fn from_stmt<'a>(stmt: &'a Stmt<'a, 'b, I>) -> Result<Self> {
        let mut ctx = Context {
            hm: Default::default(),
            defsites: Default::default(),
            orig: Default::default(),
            max: 1, // 0 reserved for assigning to "unused" var for side-effecting operations.
            cfg: Default::default(),
            entry: Default::default(),
            loop_ctx: Default::default(),
            dt: Default::default(),
            df: Default::default(),
            num_idents: 0,
            tmp_fs: None,
        };
        // convert AST to CFG
        let (start, _) = ctx.standalone_block(stmt)?;
        ctx.entry = start;
        // SSA conversion: compute dominator tree and dominance frontier.
        let (dt, df) = {
            let di = dom::DomInfo::new(ctx.cfg(), ctx.entry());
            (di.dom_tree(), di.dom_frontier())
        };
        ctx.dt = dt;
        ctx.df = df;
        // SSA conversion: insert phi functions, and rename variables.
        ctx.insert_phis();
        ctx.rename(ctx.entry());
        Ok(ctx)
    }

    pub fn standalone_block<'a>(
        &mut self,
        stmt: &'a Stmt<'a, 'b, I>,
    ) -> Result<(NodeIx /*start*/, NodeIx /*end*/)> {
        let start = self.cfg.add_node(Default::default());
        let end = self.convert_stmt(stmt, start)?;
        Ok((start, end))
    }

    fn convert_stmt<'a>(
        &mut self,
        stmt: &'a Stmt<'a, 'b, I>,
        mut current_open: NodeIx,
    ) -> Result<NodeIx> /*next open */ {
        use Stmt::*;
        Ok(match stmt {
            Expr(e) => {
                self.convert_expr(e, current_open)?;
                current_open
            }
            Block(stmts) => {
                for s in stmts {
                    current_open = self.convert_stmt(s, current_open)?;
                }
                current_open
            }
            Print(vs, out) => {
                debug_assert!(vs.len() > 0);
                let out = if let Some((o, append)) = out {
                    let e = self.convert_val(o, current_open)?;
                    Some((e, append))
                } else {
                    None
                };
                let print = {
                    |v| {
                        if let Some((o, append)) = out {
                            PrimExpr::CallBuiltin(
                                builtins::Function::Print,
                                smallvec![v, o.clone(), PrimVal::ILit(*append as i64)],
                            )
                        } else {
                            PrimExpr::CallBuiltin(builtins::Function::PrintStdout, smallvec![v])
                        }
                    }
                };
                if vs.len() == 0 {
                    let tmp = self.fresh();
                    self.add_stmt(
                        current_open,
                        PrimStmt::AsgnVar(
                            tmp,
                            PrimExpr::CallBuiltin(
                                builtins::Function::Unop(Unop::Column),
                                smallvec![PrimVal::ILit(0)],
                            ),
                        ),
                    );
                    self.add_stmt(
                        current_open,
                        PrimStmt::AsgnVar(Self::unused(), print(PrimVal::Var(tmp))),
                    );
                    current_open
                } else if vs.len() == 1 {
                    let v = self.convert_val(vs[0], current_open)?;
                    self.add_stmt(current_open, PrimStmt::AsgnVar(Self::unused(), print(v)));
                    current_open
                } else {
                    const EMPTY: PrimVal<'static> = PrimVal::StrLit("");

                    // Assign the field separator to a local variable.
                    let fs = {
                        let fs = self.field_sep();
                        self.add_stmt(
                            current_open,
                            PrimStmt::AsgnVar(
                                fs.clone(),
                                PrimExpr::LoadBuiltin(builtins::Variable::FS),
                            ),
                        );
                        PrimVal::Var(fs)
                    };

                    // For each argument in the comma-separated list, concatenate in sequence along
                    // with the field separator. Doing this now because (1) concatenation of
                    // strings lazy, making this cheap and (2) because it simplifies how some of
                    // the downstream analysis goes. Depending on how this impacts performance we
                    // may add support for var-arg printing later on.
                    //
                    // (e.g.  how will printf work? Will we disallow dynamically computed printf
                    // strings? We probably should...)
                    let mut tmp = self.fresh();
                    self.add_stmt(current_open, PrimStmt::AsgnVar(tmp, PrimExpr::Val(EMPTY)));
                    for (i, v) in vs.iter().enumerate() {
                        let v = self.convert_val(*v, current_open)?;
                        if i != 0 {
                            let new_tmp = self.fresh();
                            self.add_stmt(
                                current_open,
                                PrimStmt::AsgnVar(
                                    new_tmp,
                                    PrimExpr::CallBuiltin(
                                        builtins::Function::Binop(Binop::Concat),
                                        smallvec![PrimVal::Var(tmp), fs.clone()],
                                    ),
                                ),
                            );
                            tmp = new_tmp;
                        }
                        let new_tmp = self.fresh();
                        self.add_stmt(
                            current_open,
                            PrimStmt::AsgnVar(
                                new_tmp,
                                PrimExpr::CallBuiltin(
                                    builtins::Function::Binop(Binop::Concat),
                                    smallvec![PrimVal::Var(tmp), v],
                                ),
                            ),
                        );
                        tmp = new_tmp;
                    }
                    self.add_stmt(
                        current_open,
                        PrimStmt::AsgnVar(Self::unused(), print(PrimVal::Var(tmp))),
                    );

                    current_open
                }
            }
            If(cond, tcase, fcase) => {
                let c_val = if let ast::Expr::PatLit(_) = cond {
                    // For conditionals, pattern literals become matches against $0.
                    use ast::{Binop::*, Expr::*, Unop::*};
                    self.convert_val(&Binop(Match, &Unop(Column, &ILit(0)), cond), current_open)?
                } else {
                    self.convert_val(cond, current_open)?
                };
                let (t_start, t_end) = self.standalone_block(tcase)?;
                let next = self.cfg.add_node(Default::default());

                // current_open => t_start if the condition holds
                self.cfg
                    .add_edge(current_open, t_start, Transition::new(c_val));
                // continue to next after the true case is evaluated
                self.cfg.add_edge(t_end, next, Transition::null());

                if let Some(fcase) = fcase {
                    // if an else case is there, compute a standalone block and set up the same
                    // connections as before, this time with a null edge rather than c_val.
                    let (f_start, f_end) = self.standalone_block(fcase)?;
                    self.cfg.add_edge(current_open, f_start, Transition::null());
                    self.cfg.add_edge(f_end, next, Transition::null());
                } else {
                    // otherwise continue directly from current_open.
                    self.cfg.add_edge(current_open, next, Transition::null());
                }
                next
            }
            For(init, cond, update, body) => {
                let init_end = if let Some(i) = init {
                    self.convert_stmt(i, current_open)?
                } else {
                    current_open
                };
                let (h, b_start, _b_end, f) =
                    self.make_loop(body, update.clone(), init_end, false)?;
                let cond_val = if let Some(c) = cond {
                    self.convert_val(c, h)?
                } else {
                    PrimVal::ILit(1)
                };
                self.cfg.add_edge(h, b_start, Transition::new(cond_val));
                self.cfg.add_edge(h, f, Transition::null());
                f
            }
            While(cond, body) => {
                let (h, b_start, _b_end, f) = self.make_loop(body, None, current_open, false)?;
                let cond_val = self.convert_val(cond, h)?;
                self.cfg.add_edge(h, b_start, Transition::new(cond_val));
                self.cfg.add_edge(h, f, Transition::null());
                f
            }
            DoWhile(cond, body) => {
                let (h, b_start, _b_end, f) = self.make_loop(body, None, current_open, true)?;
                let cond_val = self.convert_val(cond, h)?;
                self.cfg.add_edge(h, b_start, Transition::new(cond_val));
                self.cfg.add_edge(h, f, Transition::null());
                f
            }
            ForEach(v, array, body) => {
                let v_id = self.get_identifier(v);
                let array_val = self.convert_val(array, current_open)?;
                let array_iter = self.to_val(PrimExpr::IterBegin(array_val.clone()), current_open);

                // First, create the loop header, which checks if there are any more elements
                // in the array.
                let cond = PrimExpr::HasNext(array_iter.clone());
                let cond_block = self.cfg.add_node(Default::default());
                let cond_v = self.to_val(cond, cond_block);
                self.cfg
                    .add_edge(current_open, cond_block, Transition::null());

                // Then add a footer to exit the loop from cond. We will add the edge after adding
                // the edge into the loop body, as order matters.
                let footer = self.cfg.add_node(Default::default());

                self.loop_ctx.push((cond_block, footer));

                // Create the body, but start by getting the next element from the iterator and
                // assigning it to `v`
                let update = PrimStmt::AsgnVar(v_id, PrimExpr::Next(array_iter.clone()));
                let body_start = self.cfg.add_node(Default::default());
                self.add_stmt(body_start, update);
                let body_end = self.convert_stmt(body, body_start)?;
                self.cfg
                    .add_edge(cond_block, body_start, Transition::new(cond_v));
                self.cfg.add_edge(cond_block, footer, Transition::null());
                self.cfg.add_edge(body_end, cond_block, Transition::null());

                self.loop_ctx.pop().unwrap();

                footer
            }
            Break => {
                match self.loop_ctx.pop() {
                    Some((_, footer)) => {
                        // Break statements unconditionally jump to the end of the loop.
                        self.cfg.add_edge(current_open, footer, Transition::null());
                        current_open
                    }
                    None => {
                        return Err(CompileError("break statement must be inside a loop".into()))
                    }
                }
            }
            Continue => {
                match self.loop_ctx.pop() {
                    Some((header, _)) => {
                        // Continue statements unconditionally jump to the top of the loop.
                        self.cfg.add_edge(current_open, header, Transition::null());
                        current_open
                    }
                    None => {
                        return Err(CompileError(
                            "continue statement must be inside a loop".into(),
                        ))
                    }
                }
            }
        })
    }

    fn convert_expr<'a>(
        &mut self,
        expr: &'a Expr<'a, 'b, I>,
        current_open: NodeIx,
    ) -> Result<PrimExpr<'b>> /* should not create any new nodes. Expressions don't cause us to branch */
    {
        use Expr::*;
        Ok(match expr {
            ILit(n) => PrimExpr::Val(PrimVal::ILit(*n)),
            FLit(n) => PrimExpr::Val(PrimVal::FLit(*n)),
            PatLit(s) | StrLit(s) => PrimExpr::Val(PrimVal::StrLit(s)),
            Unop(op, e) => {
                let v = self.convert_val(e, current_open)?;
                PrimExpr::CallBuiltin(builtins::Function::Unop(*op), smallvec![v])
            }
            Binop(op, e1, e2) => {
                let v1 = self.convert_val(e1, current_open)?;
                let v2 = self.convert_val(e2, current_open)?;
                PrimExpr::CallBuiltin(builtins::Function::Binop(*op), smallvec![v1, v2])
            }
            Var(id) => {
                if let Ok(bi) = builtins::Variable::try_from(id.clone()) {
                    PrimExpr::LoadBuiltin(bi)
                } else {
                    let ident = self.get_identifier(id);
                    PrimExpr::Val(PrimVal::Var(ident))
                }
            }
            Index(arr, ix) => {
                let arr_v = self.convert_val(arr, current_open)?;
                let ix_v = self.convert_val(ix, current_open)?;
                PrimExpr::Index(arr_v, ix_v)
            }
            Call(fname, args) => {
                let bi = match fname {
                    Either::Left(fname) => {
                        if let Ok(bi) = builtins::Function::try_from(fname.clone()) {
                            bi
                        } else {
                            return err!("Call to unknown function \"{}\"", fname);
                        }
                    }
                    Either::Right(bi) => *bi,
                };
                let mut prim_args = SmallVec::with_capacity(args.len());
                for a in args.iter() {
                    prim_args.push(self.convert_val(a, current_open)?);
                }
                PrimExpr::CallBuiltin(bi, prim_args)
            }
            Assign(Index(arr, ix), to) => {
                return self.do_assign_index(
                    arr,
                    ix,
                    |slf, _, _| slf.convert_expr(to, current_open),
                    current_open,
                )
            }

            AssignOp(Index(arr, ix), op, to) => {
                return self.do_assign_index(
                    arr,
                    ix,
                    |slf, arr_v, ix_v| {
                        let to_v = slf.convert_val(to, current_open)?;
                        let arr_cell_v =
                            slf.to_val(PrimExpr::Index(arr_v, ix_v.clone()), current_open);
                        Ok(PrimExpr::CallBuiltin(
                            builtins::Function::Binop(*op),
                            smallvec![arr_cell_v, to_v],
                        ))
                    },
                    current_open,
                )
            }
            Assign(x, to) => {
                let to = self.convert_expr(to, current_open)?;
                return self.do_assign(x, |_| to, current_open);
            }
            AssignOp(x, op, to) => {
                let to_v = self.convert_val(to, current_open)?;
                return self.do_assign(
                    x,
                    |v| {
                        PrimExpr::CallBuiltin(
                            builtins::Function::Binop(*op),
                            smallvec![v.clone(), to_v],
                        )
                    },
                    current_open,
                );
            }
            Inc { is_inc, is_post, x } => {
                if !valid_lhs(x) {
                    return err!("invalid operand for increment operation {:?}", x);
                };
                // XXX Somewhat lazy; we emit a laod even if it is a post-increment.
                let pre = self.convert_expr(x, current_open)?;
                let post = self.convert_expr(
                    &ast::Expr::AssignOp(
                        x,
                        if *is_inc {
                            ast::Binop::Plus
                        } else {
                            ast::Binop::Minus
                        },
                        &ast::Expr::ILit(1),
                    ),
                    current_open,
                )?;
                if *is_post {
                    post
                } else {
                    pre
                }
            }
            Getline { from, into } => {
                // Another use of non-structural recursion for desugaring. Here we desugar:
                //   getline var < file
                // to
                //   var = nextline(file)
                //   readerr(file)
                // And we fill in various other pieces of sugar as well. Informally:
                //  getline < file => getline $0 < file
                //  getline var => getline var < stdin
                //  getline => getline $0
                use builtins::Function::{Nextline, NextlineStdin, ReadErr, ReadErrStdin};
                match (from, into) {
                    (from, None /* $0 */) => self.convert_expr(
                        &ast::Expr::Getline {
                            from: from.clone(),
                            into: Some(&Unop(ast::Unop::Column, &ast::Expr::ILit(0))),
                        },
                        current_open,
                    ),
                    (Some(from), Some(into)) => {
                        self.convert_expr(
                            &ast::Expr::Assign(
                                into,
                                &ast::Expr::Call(Either::Right(Nextline), vec![from]),
                            ),
                            current_open,
                        )?;
                        self.convert_expr(
                            &ast::Expr::Call(Either::Right(ReadErr), vec![from]),
                            current_open,
                        )
                    }
                    (None /*stdin*/, Some(into)) => {
                        self.convert_expr(
                            &ast::Expr::Assign(
                                into,
                                &ast::Expr::Call(Either::Right(NextlineStdin), vec![]),
                            ),
                            current_open,
                        )?;
                        self.convert_expr(
                            &ast::Expr::Call(Either::Right(ReadErrStdin), vec![]),
                            current_open,
                        )
                    }
                }?
            }
        })
    }
    fn do_assign<'a>(
        &mut self,
        v: &'a Expr<'a, 'b, I>,
        to: impl FnOnce(&PrimVal<'b>) -> PrimExpr<'b>,
        current_open: NodeIx,
    ) -> Result<PrimExpr<'b>> {
        use ast::Expr::*;
        match v {
            Var(i) => Ok(if let Ok(b) = builtins::Variable::try_from(i.clone()) {
                let res = PrimExpr::LoadBuiltin(b);
                let res_v = self.to_val(res.clone(), current_open);
                self.add_stmt(current_open, PrimStmt::SetBuiltin(b, to(&res_v)));
                res
            } else {
                let ident = self.get_identifier(i);
                let res_v = PrimVal::Var(ident);
                self.add_stmt(current_open, PrimStmt::AsgnVar(ident, to(&res_v)));
                PrimExpr::Val(res_v)
            }),
            Unop(ast::Unop::Column, n) => {
                use {ast::Unop::*, builtins::Function};
                let v = self.convert_val(n, current_open)?;
                let res = PrimExpr::CallBuiltin(Function::Unop(Column), smallvec![v.clone()]);
                let res_v = self.to_val(res.clone(), current_open);
                let to_v = self.to_val(to(&res_v), current_open);
                self.add_stmt(
                    current_open,
                    PrimStmt::AsgnVar(
                        Self::unused(),
                        PrimExpr::CallBuiltin(Function::Setcol, smallvec![v, to_v]),
                    ),
                );
                Ok(res)
            }
            _ => err!("unsupprted assignment LHS: {:?}", v),
        }
    }
    fn do_assign_index<'a>(
        &mut self,
        arr: &'a Expr<'a, 'b, I>,
        ix: &'a Expr<'a, 'b, I>,
        mut to_f: impl FnMut(&mut Self, PrimVal<'b>, PrimVal<'b>) -> Result<PrimExpr<'b>>,
        current_open: NodeIx,
    ) -> Result<PrimExpr<'b>> {
        let arr_e = self.convert_expr(arr, current_open)?;
        let arr_id = self.fresh();
        self.add_stmt(current_open, PrimStmt::AsgnVar(arr_id, arr_e));
        let arr_v = PrimVal::Var(arr_id);

        let ix_v = self.convert_val(ix, current_open)?;
        let to_e = to_f(self, arr_v.clone(), ix_v.clone())?;
        self.add_stmt(
            current_open,
            PrimStmt::AsgnIndex(arr_id, ix_v.clone(), to_e.clone()),
        );
        Ok(PrimExpr::Index(arr_v, ix_v))
    }

    fn convert_val<'a>(
        &mut self,
        expr: &'a Expr<'a, 'b, I>,
        current_open: NodeIx,
    ) -> Result<PrimVal<'b>> {
        let e = self.convert_expr(expr, current_open)?;
        Ok(self.to_val(e, current_open))
    }
    fn make_loop<'a>(
        &mut self,
        body: &'a Stmt<'a, 'b, I>,
        update: Option<&'a Stmt<'a, 'b, I>>,
        current_open: NodeIx,
        is_do: bool,
    ) -> Result<(
        NodeIx, // header
        NodeIx, // body header
        NodeIx, // body footer
        NodeIx, // footer = next open
    )> {
        // Create header and footer nodes.
        let h = self.cfg.add_node(Default::default());
        let f = self.cfg.add_node(Default::default());
        self.loop_ctx.push((h, f));

        // The body is a standalone graph.
        let (b_start, b_end) = if let Some(u) = update {
            let (start, mid) = self.standalone_block(body)?;
            let end = self.convert_stmt(u, mid)?;
            (start, end)
        } else {
            self.standalone_block(body)?
        };

        // do-while loops start by running the loop body.
        // The last few edges here are added after make_loop returns to convert_stmt.
        if is_do {
            // Current => Body => Header => Body => Footer
            //                      ^       |
            //                      ^--------
            self.cfg.add_edge(current_open, b_start, Transition::null());
        } else {
            // Current => Header => Body => Footer
            //             ^         |
            //             ^---------
            self.cfg.add_edge(current_open, h, Transition::null());
        }
        self.cfg.add_edge(b_end, h, Transition::null());
        self.loop_ctx.pop().unwrap();
        Ok((h, b_start, b_end, f))
    }

    fn to_val(&mut self, exp: PrimExpr<'b>, current_open: NodeIx) -> PrimVal<'b> {
        if let PrimExpr::Val(v) = exp {
            v
        } else {
            let f = self.fresh();
            self.add_stmt(current_open, PrimStmt::AsgnVar(f, exp));
            PrimVal::Var(f)
        }
    }

    fn fresh(&mut self) -> Ident {
        let res = self.max;
        self.max += 1;
        (res, 0)
    }

    fn record_ident(&mut self, id: Ident, blk: NodeIx) {
        self.defsites
            .entry(id)
            .or_insert(HashSet::default())
            .insert(blk);
        self.orig
            .entry(blk)
            .or_insert(HashSet::default())
            .insert(id);
    }

    fn get_identifier(&mut self, i: &I) -> Ident {
        if let Some(id) = self.hm.get(i) {
            return *id;
        }
        let next = self.fresh();
        self.hm.insert(i.clone(), next);
        next
    }

    fn add_stmt(&mut self, at: NodeIx, stmt: PrimStmt<'b>) {
        if let PrimStmt::AsgnVar(ident, _) = stmt {
            self.record_ident(ident, at);
        }
        self.cfg.node_weight_mut(at).unwrap().0.push_back(stmt);
    }

    fn insert_phis(&mut self) {
        use crate::common::WorkList;
        // TODO: do we need defsites and orig after this, or can we deallocate them here?

        // phis: the set of basic blocks that must have a phi node for a given variable.
        let mut phis = HashMap::<Ident, HashSet<NodeIx>>::new();
        let mut worklist = WorkList::default();
        // Note, to be cautious we could insert Phis for all identifiers.  But that would introduce
        // additional nodes for variables that are assigned to only once by construction. Instead
        // we only use named variables. Of course, were this to change we would need to fall back
        // on something more conservative.
        for ident in self.hm.values().map(Clone::clone) {
            // Add all defsites into the worklist.
            let defsites = if let Some(ds) = self.defsites.get(&ident) {
                ds
            } else {
                continue;
            };
            worklist.extend(defsites.iter().map(|x| *x));
            while let Some(node) = worklist.pop() {
                // For all nodes on the dominance frontier without phi nodes for this identifier,
                // create a phi node of the appropriate size and insert it at the front of the
                // block (no renaming).
                for d in self.df[node.index()].iter() {
                    let d_ix = NodeIx::new(*d as usize);
                    if phis.get(&ident).map(|s| s.contains(&d_ix)) != Some(true) {
                        let phi = PrimExpr::Phi(
                            self.cfg()
                                .neighbors_directed(d_ix, Direction::Incoming)
                                .map(|n| (n, ident))
                                .collect(),
                        );
                        let stmt = PrimStmt::AsgnVar(ident, phi);
                        self.cfg
                            .node_weight_mut(d_ix)
                            .expect("node in dominance frontier must be valid")
                            .0
                            .push_front(stmt);
                        phis.entry(ident).or_insert(HashSet::default()).insert(d_ix);
                    }
                }
            }
        }
    }

    fn rename(&mut self, cur: NodeIx) {
        #[derive(Clone)]
        struct RenameStack {
            count: NumTy,
            stack: SmallVec<NumTy>,
        }

        impl RenameStack {
            fn latest(&self) -> NumTy {
                *self
                    .stack
                    .last()
                    .expect("variable stack should never be empty")
            }
            fn get_next(&mut self) -> NumTy {
                let next = self.count + 1;
                self.count = next;
                self.stack.push(next);
                next
            }
        }

        fn rename_recursive<'b, I>(
            ctx: &mut Context<'b, I>,
            cur: NodeIx,
            state: &mut Vec<RenameStack>,
        ) {
            // We need to remember which new variables are introduced in this frame so we can
            // remove them when we are done.
            let mut defs = smallvec::SmallVec::<[NumTy; 16]>::new();

            // First, go through all the statements and update the variables to the highest
            // subscript (second component in Ident).
            for stmt in &mut ctx
                .cfg
                .node_weight_mut(cur)
                .expect("rename must be passed valid node indices")
                .0
            {
                // Note that `replace` is specialized to our use-case in this method. It does not hit
                // AsgnVar identifiers, and it skips Phi nodes.
                stmt.replace(|(x, _)| (x, state[x as usize].latest()));
                if let PrimStmt::AsgnVar((a, i), _) = stmt {
                    *i = state[*a as usize].get_next();
                    defs.push(*a);
                }
            }

            // The recursion is structured around the dominator tree. That means that normal
            // renaming may not update join points in a graph to the right value. Consider
            //
            //            A (x=1)
            //          /   \
            //         /     \
            //        B (x++) C (x+=2)
            //         \     /
            //          \   /
            //            D (x+=5)
            //
            // With flow pointing downward. The dominator tree looks like
            //
            //          A
            //        / | \
            //       B  C  D
            //
            // Without this while loop, we would wind up with something like:
            // A: x0 = 1
            // B: x1 = x0 + 1
            // C: x2 = x0 + 2
            // D: x3 = phi(x0, x0); x4 = x3 + 5
            //
            // But of course the phi node is wrong, it should be:
            // x3 = phi(x1, x2)
            //
            // To fix this, we iterate over any outgoing neighbors and find phi functions that
            // point back to the current node and update the subscript accordingly.
            let mut walker = ctx
                .cfg
                .neighbors_directed(cur, Direction::Outgoing)
                .detach();
            while let Some((edge, neigh)) = walker.next(&ctx.cfg) {
                if let Some(PrimVal::Var((x, sub))) = &mut ctx.cfg.edge_weight_mut(edge).unwrap().0
                {
                    *sub = state[*x as usize].latest();
                }
                for stmt in &mut ctx.cfg.node_weight_mut(neigh).unwrap().0 {
                    if let PrimStmt::AsgnVar(_, PrimExpr::Phi(ps)) = stmt {
                        for (pred, (x, sub)) in ps.iter_mut() {
                            if pred == &cur {
                                *sub = state[*x as usize].latest();
                                break;
                            }
                        }
                    }
                }
            }
            for child in ctx.dt[cur.index()].clone().iter() {
                rename_recursive(ctx, NodeIx::new(*child as usize), state);
            }
            for d in defs.into_iter() {
                let _res = state[d as usize].stack.pop();
                debug_assert!(_res.is_some());
            }
        }
        let mut state = vec![
            RenameStack {
                count: 0,
                stack: smallvec![0],
            };
            self.max as usize
        ];
        rename_recursive(self, cur, &mut state);
        for s in state {
            // `s.count` really counts the *extra* identifiers we introduce after (N, 0), so we
            // need to add an extra.
            self.num_idents += s.count as usize + 1;
        }
    }
}
