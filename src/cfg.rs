use crate::ast::{Expr, NumBinop, NumUnop, Stmt, StrBinop, StrUnop};
use crate::dom;
use crate::types::{CompileError, Graph, NodeIx, NumTy, Result};

use hashbrown::{HashMap, HashSet};
use petgraph::Direction;
use smallvec::{smallvec, SmallVec};

use std::collections::VecDeque;
use std::hash::Hash;

// TODO: nail down iteration order for edges
// TODO: figure out some testing strategies
// TODO: add in break and continue

// consider making this just "by number" and putting branch instructions elsewhere.
// need to verify the order
// Use VecDequeue to support things like prepending definitions and phi statements to blocks during
// SSA conversion.
type BasicBlock<'a> = VecDeque<PrimStmt<'a>>;
// None indicates `else`
pub(crate) type CFG<'a> = Graph<BasicBlock<'a>, Option<PrimVal<'a>>>;
type Ident = (NumTy, NumTy);
type V<T> = SmallVec<[T; 2]>;

#[derive(Debug, Clone)]
pub(crate) enum PrimVal<'a> {
    Var(Ident),
    NumLit(f64),
    StrLit(&'a str),
}

#[derive(Debug, Clone)]
pub(crate) enum PrimExpr<'a> {
    Val(PrimVal<'a>),
    Phi(V<(NodeIx /* pred */, Ident)>),
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

// Basic tree-walking used in `Context::rename`

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
            StrUnop(_, v) => v.replace(update),
            StrBinop(_, v1, v2) => {
                v1.replace(&mut update);
                v2.replace(update);
            }
            NumUnop(_, v) => v.replace(update),
            NumBinop(_, v1, v2) => {
                v1.replace(&mut update);
                v2.replace(update);
            }
            Index(v1, v2) => {
                v1.replace(&mut update);
                v2.replace(update);
            }
            IterBegin(v) => v.replace(update),
            HasNext(v) => v.replace(update),
            Next(v) => v.replace(update),
        }
    }
}

impl<'a> PrimStmt<'a> {
    fn replace(&mut self, mut update: impl FnMut(Ident) -> Ident) {
        use PrimStmt::*;
        match self {
            Print(vs, None) => {
                for v in vs.iter_mut() {
                    v.replace(&mut update)
                }
            }
            Print(vs, Some(v)) => {
                for v in vs.iter_mut() {
                    v.replace(&mut update)
                }
                v.replace(update);
            }
            AsgnIndex(ident, v, exp) => {
                *ident = update(*ident);
                v.replace(&mut update);
                exp.replace(update);
            }
            // We handle assignments separately. Note that this is not needed for index
            // expressions, because assignments to m[k] are *uses* of m, not definitions.
            AsgnVar(_, e) => e.replace(update),
        }
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
    loop_ctx: V<(NodeIx, NodeIx)>,

    // Dominance information about `cfg`.
    dt: dom::Tree,
    df: dom::Frontier,
}

impl<'b, I> Context<'b, I> {
    pub fn cfg<'a>(&'a self) -> &'a CFG<'b> {
        &self.cfg
    }
    pub fn entry(&self) -> NodeIx {
        self.entry
    }
}
impl<'b, I: Hash + Eq + Clone + Default> Context<'b, I> {
    pub fn from_stmt<'a>(stmt: &'a Stmt<'a, 'b, I>) -> Result<Self> {
        let mut ctx = Context {
            hm: Default::default(),
            defsites: Default::default(),
            orig: Default::default(),
            max: Default::default(),
            cfg: Default::default(),
            entry: Default::default(),
            loop_ctx: Default::default(),
            dt: Default::default(),
            df: Default::default(),
        };
        let (start, _) = ctx.standalone_block(stmt)?;
        ctx.entry = start;
        let (dt, df) = {
            let di = dom::DomInfo::new(ctx.cfg(), ctx.entry());
            (di.dom_tree(), di.dom_frontier())
        };
        ctx.dt = dt;
        ctx.df = df;
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
                let mut v = V::with_capacity(vs.len());
                for i in vs.iter() {
                    v.push(self.convert_val(*i, current_open)?)
                }
                let out = match out.as_ref() {
                    Some(x) => Some(self.convert_val(x, current_open)?),
                    None => None,
                };
                self.add_stmt(current_open, PrimStmt::Print(v, out));
                current_open
            }
            If(cond, tcase, fcase) => {
                let c_val = self.convert_val(cond, current_open)?;
                let (t_start, t_end) = self.standalone_block(tcase)?;
                let next = self.cfg.add_node(Default::default());

                // current_open => t_start if the condition holds
                self.cfg.add_edge(current_open, t_start, Some(c_val));
                // continue to next after the true case is evaluated
                self.cfg.add_edge(t_end, next, None);

                if let Some(fcase) = fcase {
                    // if an else case is there, compute a standalone block and set up the same
                    // connections as before, this time with a null edge rather than c_val.
                    let (f_start, f_end) = self.standalone_block(fcase)?;
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
                    self.convert_stmt(i, current_open)?
                } else {
                    current_open
                };
                let (h, b_start, _b_end, f) = self.make_loop(body, update.clone(), init_end)?;
                let cond_val = if let Some(c) = cond {
                    self.convert_val(c, h)?
                } else {
                    PrimVal::NumLit(1.0)
                };
                self.cfg.add_edge(h, b_start, Some(cond_val));
                self.cfg.add_edge(h, f, None);
                f
            }
            While(cond, body) => {
                let (h, b_start, _b_end, f) = self.make_loop(body, None, current_open)?;
                let cond_val = self.convert_val(cond, h)?;
                self.cfg.add_edge(h, b_start, Some(cond_val));
                self.cfg.add_edge(h, f, None);
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
                self.cfg.add_edge(current_open, cond_block, None);

                // Then add a footer to exit the loop from cond.
                let footer = self.cfg.add_node(Default::default());
                self.cfg.add_edge(cond_block, footer, None);

                self.loop_ctx.push((cond_block, footer));

                // Create the body, but start by getting the next element from the iterator and
                // assigning it to `v`
                let update = PrimStmt::AsgnVar(v_id, PrimExpr::Next(array_iter.clone()));
                let body_start = self.cfg.add_node(Default::default());
                self.add_stmt(body_start, update);
                let body_end = self.convert_stmt(body, body_start)?;
                self.cfg.add_edge(cond_block, body_start, Some(cond_v));
                self.cfg.add_edge(body_end, cond_block, None);

                self.loop_ctx.pop().unwrap();

                footer
            }
            Break => {
                match self.loop_ctx.pop() {
                    Some((_, footer)) => {
                        // Break statements unconditionally jump to the end of the loop.
                        self.cfg.add_edge(current_open, footer, None);
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
                        self.cfg.add_edge(current_open, header, None);
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
            NumLit(n) => PrimExpr::Val(PrimVal::NumLit(*n)),
            StrLit(s) => PrimExpr::Val(PrimVal::StrLit(s)),
            Unop(op, e) => {
                let v = self.convert_val(e, current_open)?;
                match op {
                    Ok(numop) => PrimExpr::NumUnop(*numop, v),
                    Err(strop) => PrimExpr::StrUnop(*strop, v),
                }
            }
            Binop(op, e1, e2) => {
                let v1 = self.convert_val(e1, current_open)?;
                let v2 = self.convert_val(e2, current_open)?;
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
                let arr_v = self.convert_val(arr, current_open)?;
                let ix_v = self.convert_val(ix, current_open)?;
                PrimExpr::Index(arr_v, ix_v)
            }
            Assign(Var(v), to) => {
                let to_e = self.convert_expr(to, current_open)?;
                let ident = self.get_identifier(v);
                self.add_stmt(current_open, PrimStmt::AsgnVar(ident, to_e));
                PrimExpr::Val(PrimVal::Var(ident))
            }
            AssignOp(Var(v), op, to) => {
                let to_v = self.convert_val(to, current_open)?;
                let ident = self.get_identifier(v);
                let tmp = PrimExpr::NumBinop(*op, PrimVal::Var(ident), to_v);
                self.add_stmt(current_open, PrimStmt::AsgnVar(ident, tmp));
                PrimExpr::Val(PrimVal::Var(ident))
            }

            Assign(Index(arr, ix), to) => {
                return self.do_assign(
                    arr,
                    ix,
                    |slf, _, _| slf.convert_expr(to, current_open),
                    current_open,
                )
            }

            AssignOp(Index(arr, ix), op, to) => {
                return self.do_assign(
                    arr,
                    ix,
                    |slf, arr_v, ix_v| {
                        let to_v = slf.convert_val(to, current_open)?;
                        let arr_cell_v =
                            slf.to_val(PrimExpr::Index(arr_v, ix_v.clone()), current_open);
                        Ok(PrimExpr::NumBinop(*op, arr_cell_v, to_v))
                    },
                    current_open,
                )
            }
            // Panic here because this marks an internal error. We could move this distinction
            // up to the ast1:: level, but then we would have 4 different variants to handle
            // here.
            Assign(_, _) | AssignOp(_, _, _) => {
                return Err(CompileError(format!("{}", "invalid assignment expression")))
            }
        })
    }

    fn do_assign<'a>(
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
    ) -> Result<(
        NodeIx, /* header */
        NodeIx, /* body header */
        NodeIx, /* body footer */
        NodeIx, /* footer = next open */
    )> {
        // Create header, body, and footer nodes.
        let h = self.cfg.add_node(Default::default());
        let f = self.cfg.add_node(Default::default());
        self.loop_ctx.push((h, f));
        let (b_start, b_end) = if let Some(u) = update {
            let (start, mid) = self.standalone_block(body)?;
            let end = self.convert_stmt(u, mid)?;
            (start, end)
        } else {
            self.standalone_block(body)?
        };
        self.cfg.add_edge(current_open, h, None);
        self.cfg.add_edge(b_end, h, None);
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
        self.cfg.node_weight_mut(at).unwrap().push_back(stmt);
    }

    fn insert_phis(&mut self) {
        // TODO: do we need defsites and orig after this, or can we deallocate them here?
        let mut phis = HashMap::<Ident, HashSet<NodeIx>>::new();
        let mut worklist = HashSet::new();
        for ident in (0..self.max).map(|x| (x, 0 as NumTy)) {
            // Add all defsites into the worklist.
            let defsites = if let Some(ds) = self.defsites.get(&ident) {
                ds
            } else {
                continue;
            };
            worklist.extend(defsites.iter().map(|x| *x));
            while worklist.len() > 0 {
                // Remove a node from the worklist.
                let node = {
                    let fst = worklist
                        .iter()
                        .next()
                        .expect("worklist cannot be empty")
                        .clone();
                    worklist
                        .take(&fst)
                        .expect("worklist must yield elements from the set")
                };
                // For all nodes on the dominance frontier without phi nodes for this identifier,
                // create a phi node of the appropriate size and insert it at the front of the
                // block (no renaming).
                for d in self.df[node.index()].iter() {
                    let d_ix = NodeIx::new(*d as usize);
                    if phis.get(&ident).map(|s| s.contains(&d_ix)) == Some(true) {
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
            stack: V<NumTy>,
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
            let mut defs = SmallVec::<[NumTy; 16]>::new();

            // First, go through all the statements and update the variables to the highest
            // subscript (second component in Ident).
            for stmt in ctx
                .cfg
                .node_weight_mut(cur)
                .expect("rename must be passed valid node indices")
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
            //            A
            //          x = 1
            //        /       \
            //       B         C
            //  x = x + 1     x = x + 2
            //        \      /
            //           D
            //       x = x + 5
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
            // D: x3 = phi(x0, x0)
            //
            // But of course D must be:
            // D: x3 = phi(x1, x2)
            //
            // To fix this, we iterate over any outgoing neighbors and find phi functions that
            // point back to the current node and update the subscript accordingly.
            let mut walker = ctx
                .cfg
                .neighbors_directed(cur, Direction::Outgoing)
                .detach();
            while let Some(neigh) = walker.next_node(&ctx.cfg) {
                for stmt in ctx.cfg.node_weight_mut(neigh).unwrap() {
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
    }
}
