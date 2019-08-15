use super::hashbrown::HashMap;
use crate::types::{NodeIx, NumTy};
use petgraph::graph::Graph;
use std::collections::VecDeque;
use std::hash::Hash;

use crate::ast::{Expr, NumBinop, NumUnop, Stmt, StrBinop, StrUnop};

// Several utility functions operate on both NodeIx and NumTy values. Making them polymorphic
// on HasNum automates the conversion between these two types.

// consider making this just "by number" and putting branch instructions elsewhere.
// need to verify the order
// Use VecDequeue to support things like prepending definitions and phi statements to blocks during
// SSA conversion.
type BasicBlock<'a> = VecDeque<PrimStmt<'a>>;
// None indicates `else`
pub(crate) type CFG<'a> = Graph<BasicBlock<'a>, Option<PrimVal<'a>>, petgraph::Directed, NumTy>;
type Ident = (NumTy, NumTy); // change to u64?
type V<T> = Vec<T>; // change to smallvec?

// Inserting Phi functions:
// -1. look into making ssa generic on graphs, rename it to `dom.rs` or something, then insert Phis
// here.
//  0. Create a Declare(I) PrimStmt. (NO: instead assign it to the empty string.. hopefully
//     lifetime subtyping works out there)
//  1. populate A_orig.
//   * Add as a field in BasicBlock (HashSet<I>)
//   * Insert when a new temporary is created.
//   * When a new non-temporary is encountered, prepend a Declare to the entry node.
//  2. implement algorithm 19.6 (using vec prepends for Phi insertions)
//
// Variable renames:
//  This should be very simple with the modifications for phi functions.

#[derive(Debug, Clone)]
pub(crate) enum PrimVal<'a> {
    Var(Ident),
    NumLit(f64),
    StrLit(&'a str),
}
#[derive(Debug, Clone)]
pub(crate) enum PrimExpr<'a> {
    Val(PrimVal<'a>),
    Phi(V<PrimVal<'a>>),
    StrUnop(StrUnop, PrimVal<'a>),
    StrBinop(StrBinop, PrimVal<'a>, PrimVal<'a>),
    NumUnop(NumUnop, PrimVal<'a>),
    NumBinop(NumBinop, PrimVal<'a>, PrimVal<'a>),
    Index(PrimVal<'a>, PrimVal<'a>),

    // For iterating over vectors.
    IterBegin(PrimVal<'a>),
    HasNext(PrimVal<'a>),
    Next(PrimVal<'a>),
}
#[derive(Debug)]
pub(crate) enum PrimStmt<'a> {
    Print(V<PrimVal<'a>>, Option<PrimVal<'a>>),
    AsgnIndex(
        Ident,        /*map*/
        PrimVal<'a>,  /* index */
        PrimExpr<'a>, /* assign to */
    ),
    AsgnVar(Ident /* var */, PrimExpr<'a>),
}

#[derive(Default)]
pub(crate) struct Context<'b, I> {
    hm: HashMap<I, Ident>,
    max: NumTy,
    cfg: CFG<'b>,
    entry: NodeIx,
}

#[cfg(test)]
impl<'b, I: Default> Context<'b, I> {
    // for testing
    pub(crate) fn from_cfg(cfg: CFG<'b>) -> Self {
        let mut res = Context::default();
        res.cfg = cfg;
        res
    }
}

impl<'b, I> Context<'b, I> {
    pub fn cfg(&self) -> &CFG<'b> {
        &self.cfg
    }
    pub fn entry(&self) -> NodeIx {
        self.entry
    }
}
impl<'b, I: Hash + Eq + Clone + Default> Context<'b, I> {
    fn from_stmt<'a>(stmt: &'a Stmt<'a, 'b, I>) -> Self {
        let mut ctx = Self::default();
        let (start, _) = ctx.standalone_block(stmt);
        ctx.entry = start;
        ctx
    }
    pub fn standalone_block<'a>(
        &mut self,
        stmt: &'a Stmt<'a, 'b, I>,
    ) -> (NodeIx /*start*/, NodeIx /*end*/) {
        let start = self.cfg.add_node(Default::default());
        let end = self.convert_stmt(stmt, start);
        (start, end)
    }
    fn convert_stmt<'a>(&mut self, stmt: &'a Stmt<'a, 'b, I>, mut current_open: NodeIx) -> NodeIx /*next open */
    {
        // need "current open basic block"
        use Stmt::*;
        match stmt {
            Expr(e) => {
                self.convert_expr(e, current_open);
                current_open
            }
            Block(stmts) => {
                for s in stmts {
                    current_open = self.convert_stmt(s, current_open);
                }
                current_open
            }
            Print(vs, out) => {
                debug_assert!(vs.len() > 0);
                let mut v = V::with_capacity(vs.len());
                for i in vs.iter() {
                    v.push(self.convert_val(*i, current_open))
                }
                let out = out.as_ref().map(|x| self.convert_val(x, current_open));
                self.add_stmt(current_open, PrimStmt::Print(v, out));
                current_open
            }
            If(cond, tcase, fcase) => {
                let c_val = self.convert_val(cond, current_open);
                let (t_start, t_end) = self.standalone_block(tcase);
                let next = self.cfg.add_node(Default::default());

                // current_open => t_start if the condition holds
                self.cfg.add_edge(current_open, t_start, Some(c_val));
                // continue to next after the true case is evaluated
                self.cfg.add_edge(t_end, next, None);

                if let Some(fcase) = fcase {
                    // if an else case is there, compute a standalone block and set up the same
                    // connections as before, this time with a null edge rather than c_val.
                    let (f_start, f_end) = self.standalone_block(fcase);
                    self.cfg.add_edge(current_open, f_start, None);
                    self.cfg.add_edge(f_end, next, None);
                } else {
                    // otherwise continue directly from current_open.
                    self.cfg.add_edge(current_open, next, None);
                }
                next
            }
            For(init, cond, update, body) => {
                let init_end = if let Some(i) = init {
                    self.convert_stmt(i, current_open)
                } else {
                    current_open
                };
                let (h, b_start, _b_end, f) = self.make_loop(body, update.clone(), init_end);
                let cond_val = if let Some(c) = cond {
                    self.convert_val(c, h)
                } else {
                    PrimVal::NumLit(1.0)
                };
                self.cfg.add_edge(h, b_start, Some(cond_val));
                self.cfg.add_edge(h, f, None);
                f
            }
            While(cond, body) => {
                let (h, b_start, _b_end, f) = self.make_loop(body, None, current_open);
                let cond_val = self.convert_val(cond, h);
                self.cfg.add_edge(h, b_start, Some(cond_val));
                self.cfg.add_edge(h, f, None);
                f
            }
            ForEach(v, array, body) => {
                let v_id = self.get_identifier(v);
                let array_val = self.convert_val(array, current_open);
                let array_iter = self.to_val(PrimExpr::IterBegin(array_val.clone()), current_open);

                // First, create the loop header, which checks if there are any more elements
                // in the array.
                let cond = PrimExpr::HasNext(array_iter.clone());
                let cond_block = self.cfg.add_node(Default::default());
                let cond_v = self.to_val(cond, cond_block);
                self.cfg.add_edge(current_open, cond_block, None);

                // Create the body, but start by getting the next element from the iterator and
                // assigning it to `v`
                let update = PrimStmt::AsgnVar(v_id, PrimExpr::Next(array_iter.clone()));
                let body_start = self.cfg.add_node(Default::default());
                self.add_stmt(body_start, update);
                let body_end = self.convert_stmt(body, body_start);
                self.cfg.add_edge(cond_block, body_start, Some(cond_v));
                self.cfg.add_edge(body_end, cond_block, None);

                // Then add a footer to exit the loop from cond.
                let footer = self.cfg.add_node(Default::default());
                self.cfg.add_edge(cond_block, footer, None);

                footer
            }
        }
    }

    fn convert_expr<'a>(
        &mut self,
        expr: &'a Expr<'a, 'b, I>,
        current_open: NodeIx,
    ) -> PrimExpr<'b> /* should not create any new nodes. Expressions don't cause us to branch */
    {
        use Expr::*;
        match expr {
            NumLit(n) => PrimExpr::Val(PrimVal::NumLit(*n)),
            StrLit(s) => PrimExpr::Val(PrimVal::StrLit(s)),
            Unop(op, e) => {
                let v = self.convert_val(e, current_open);
                match op {
                    Ok(numop) => PrimExpr::NumUnop(*numop, v),
                    Err(strop) => PrimExpr::StrUnop(*strop, v),
                }
            }
            Binop(op, e1, e2) => {
                let v1 = self.convert_val(e1, current_open);
                let v2 = self.convert_val(e2, current_open);
                match op {
                    Ok(numop) => PrimExpr::NumBinop(*numop, v1, v2),
                    Err(strop) => PrimExpr::StrBinop(*strop, v1, v2),
                }
            }
            Var(id) => {
                let ident = self.get_identifier(id);
                PrimExpr::Val(PrimVal::Var(ident))
            }
            Index(arr, ix) => {
                let arr_v = self.convert_val(arr, current_open);
                let ix_v = self.convert_val(ix, current_open);
                PrimExpr::Index(arr_v, ix_v)
            }
            Assign(Var(v), to) => {
                let to_e = self.convert_expr(to, current_open);
                let ident = self.get_identifier(v);
                self.add_stmt(current_open, PrimStmt::AsgnVar(ident, to_e));
                PrimExpr::Val(PrimVal::Var(ident))
            }
            AssignOp(Var(v), op, to) => {
                let to_v = self.convert_val(to, current_open);
                let ident = self.get_identifier(v);
                let tmp = PrimExpr::NumBinop(*op, PrimVal::Var(ident), to_v);
                self.add_stmt(current_open, PrimStmt::AsgnVar(ident, tmp));
                PrimExpr::Val(PrimVal::Var(ident))
            }

            Assign(Index(arr, ix), to) => self.do_assign(
                arr,
                ix,
                |slf, _, _| slf.convert_expr(to, current_open),
                current_open,
            ),

            AssignOp(Index(arr, ix), op, to) => self.do_assign(
                arr,
                ix,
                |slf, arr_v, ix_v| {
                    let to_v = slf.convert_val(to, current_open);
                    let arr_cell_v = slf.to_val(PrimExpr::Index(arr_v, ix_v.clone()), current_open);
                    PrimExpr::NumBinop(*op, arr_cell_v, to_v)
                },
                current_open,
            ),
            // Panic here because this marks an internal error. We could move this distinction
            // up to the ast1:: level, but then we would have 4 different variants to handle
            // here.
            Assign(_, _to) => panic!("invalid assignment expression"),
            AssignOp(_, _op, _to) => panic!("invalid assign-op expression"),
        }
    }

    fn do_assign<'a>(
        &mut self,
        arr: &'a Expr<'a, 'b, I>,
        ix: &'a Expr<'a, 'b, I>,
        mut to_f: impl FnMut(&mut Self, PrimVal<'b>, PrimVal<'b>) -> PrimExpr<'b>,
        current_open: NodeIx,
    ) -> PrimExpr<'b> {
        let arr_e = self.convert_expr(arr, current_open);
        let arr_id = self.fresh();
        self.add_stmt(current_open, PrimStmt::AsgnVar(arr_id, arr_e));
        let arr_v = PrimVal::Var(arr_id);

        let ix_v = self.convert_val(ix, current_open);
        let to_e = to_f(self, arr_v.clone(), ix_v.clone());
        self.add_stmt(
            current_open,
            PrimStmt::AsgnIndex(arr_id, ix_v.clone(), to_e.clone()),
        );
        PrimExpr::Index(arr_v, ix_v)
    }

    fn convert_val<'a>(&mut self, expr: &'a Expr<'a, 'b, I>, current_open: NodeIx) -> PrimVal<'b> {
        let e = self.convert_expr(expr, current_open);
        self.to_val(e, current_open)
    }

    fn make_loop<'a>(
        &mut self,
        body: &'a Stmt<'a, 'b, I>,
        update: Option<&'a Stmt<'a, 'b, I>>,
        current_open: NodeIx,
    ) -> (
        NodeIx, /* header */
        NodeIx, /* body header */
        NodeIx, /* body footer */
        NodeIx, /* footer = next open */
    ) {
        // Create header, body, and footer nodes.
        let h = self.cfg.add_node(Default::default());
        let (b_start, b_end) = if let Some(u) = update {
            let (start, mid) = self.standalone_block(body);
            let end = self.convert_stmt(u, mid);
            (start, end)
        } else {
            self.standalone_block(body)
        };
        let f = self.cfg.add_node(Default::default());
        self.cfg.add_edge(current_open, h, None);
        self.cfg.add_edge(b_end, h, None);
        (h, b_start, b_end, f)
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

    fn get_identifier(&mut self, i: &I) -> Ident {
        if let Some(id) = self.hm.get(i) {
            return *id;
        }
        let next = self.fresh();
        self.hm.insert(i.clone(), next);
        next
    }

    fn add_stmt(&mut self, at: NodeIx, stmt: PrimStmt<'b>) {
        self.cfg.node_weight_mut(at).unwrap().push_back(stmt);
    }
}
