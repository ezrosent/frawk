use crate::ast::{Binop, Expr, Stmt, Unop};
use crate::builtins::Builtin;
use crate::common::{CompileError, Graph, NodeIx, NumTy, Result};
use crate::dom;

use hashbrown::{HashMap, HashSet};
use petgraph::Direction;
use smallvec::smallvec; // macro

use std::collections::VecDeque;
use std::hash::Hash;

// TODO: nail down iteration order for edges
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
type SmallVec<T> = smallvec::SmallVec<[T; 4]>;

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
    CallBuiltin(Builtin, SmallVec<PrimVal<'a>>),
    Index(PrimVal<'a>, PrimVal<'a>),

    // For iterating over vectors.
    // TODO: make these builtins
    IterBegin(PrimVal<'a>),
    HasNext(PrimVal<'a>),
    Next(PrimVal<'a>),
}

#[derive(Debug)]
pub(crate) enum PrimStmt<'a> {
    AsgnIndex(
        Ident,        /*map*/
        PrimVal<'a>,  /* index */
        PrimExpr<'a>, /* assign to */
    ),
    AsgnVar(Ident /* var */, PrimExpr<'a>),
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
    loop_ctx: SmallVec<(NodeIx, NodeIx)>,

    // Dominance information about `cfg`.
    dt: dom::Tree,
    df: dom::Frontier,

    num_idents: usize,
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
impl<'b, I: Hash + Eq + Clone + Default> Context<'b, I> {
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
                let out = match out.as_ref() {
                    Some(x) => self.convert_val(x, current_open)?,
                    None => PrimVal::StrLit(""),
                };
                if vs.len() == 0 {
                    let tmp = self.fresh();
                    self.add_stmt(
                        current_open,
                        PrimStmt::AsgnVar(
                            tmp,
                            PrimExpr::CallBuiltin(
                                Builtin::Unop(Unop::Column),
                                smallvec![PrimVal::ILit(0)],
                            ),
                        ),
                    );
                    self.add_stmt(
                        current_open,
                        PrimStmt::AsgnVar(
                            Self::unused(),
                            PrimExpr::CallBuiltin(
                                Builtin::Print,
                                smallvec![PrimVal::Var(tmp), out],
                            ),
                        ),
                    );
                    current_open
                } else if vs.len() == 1 {
                    let v = self.convert_val(vs[0], current_open)?;
                    self.add_stmt(
                        current_open,
                        PrimStmt::AsgnVar(
                            Self::unused(),
                            PrimExpr::CallBuiltin(Builtin::Print, smallvec![v, out]),
                        ),
                    );
                    current_open
                } else {
                    const EMPTY: PrimVal<'static> = PrimVal::StrLit("");
                    // TODO: wire in field-separator here when we handle special variables.
                    const FS: PrimVal<'static> = PrimVal::StrLit(" ");

                    // For each argument in the comma-separated list, concatenate in sequence along
                    // with the field separator. Doing this now because (1) we intend to make
                    // concatenation of strings lazy, making this cheap and (2) because it
                    // simplifies how some of the downstream analysis goes. Depending on how this
                    // impacts performance we may add support for var-arg printing later on.
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
                                        Builtin::Binop(Binop::Concat),
                                        smallvec![PrimVal::Var(tmp), FS],
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
                                    Builtin::Binop(Binop::Concat),
                                    smallvec![PrimVal::Var(tmp), v],
                                ),
                            ),
                        );
                        tmp = new_tmp;
                    }
                    self.add_stmt(
                        current_open,
                        PrimStmt::AsgnVar(
                            Self::unused(),
                            PrimExpr::CallBuiltin(
                                Builtin::Print,
                                smallvec![PrimVal::Var(tmp), out],
                            ),
                        ),
                    );

                    current_open

                    // let tmp = self.fresh();
                    // let mut v = SmallVec::with_capacity(vs.len());
                    // for i in vs.iter() {
                    //     v.push(self.convert_val(*i, current_open)?)
                    // }
                    // let out = match out.as_ref() {
                    //     Some(x) => Some(self.convert_val(x, current_open)?),
                    //     None => None,
                    // };
                    // self.add_stmt(current_open, PrimStmt::Print(v, out));
                    // current_open
                }
            }
            If(cond, tcase, fcase) => {
                let c_val = self.convert_val(cond, current_open)?;
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
                let (h, b_start, _b_end, f) = self.make_loop(body, update.clone(), init_end)?;
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
                let (h, b_start, _b_end, f) = self.make_loop(body, None, current_open)?;
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

                // Then add a footer to exit the loop from cond.
                let footer = self.cfg.add_node(Default::default());
                self.cfg.add_edge(cond_block, footer, Transition::null());

                self.loop_ctx.push((cond_block, footer));

                // Create the body, but start by getting the next element from the iterator and
                // assigning it to `v`
                let update = PrimStmt::AsgnVar(v_id, PrimExpr::Next(array_iter.clone()));
                let body_start = self.cfg.add_node(Default::default());
                self.add_stmt(body_start, update);
                let body_end = self.convert_stmt(body, body_start)?;
                self.cfg
                    .add_edge(cond_block, body_start, Transition::new(cond_v));
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
            StrLit(s) => PrimExpr::Val(PrimVal::StrLit(s)),
            Unop(op, e) => {
                let v = self.convert_val(e, current_open)?;
                PrimExpr::CallBuiltin(Builtin::Unop(*op), smallvec![v])
            }
            Binop(op, e1, e2) => {
                let v1 = self.convert_val(e1, current_open)?;
                let v2 = self.convert_val(e2, current_open)?;
                PrimExpr::CallBuiltin(Builtin::Binop(*op), smallvec![v1, v2])
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
                // TODO: do validation here for which ops support assigns?
                let to_v = self.convert_val(to, current_open)?;
                let ident = self.get_identifier(v);
                let tmp = PrimExpr::CallBuiltin(
                    Builtin::Binop(*op),
                    smallvec![PrimVal::Var(ident), to_v],
                );
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
                        Ok(PrimExpr::CallBuiltin(
                            Builtin::Binop(*op),
                            smallvec![arr_cell_v, to_v],
                        ))
                    },
                    current_open,
                )
            }
            // Panic here because this marks an internal error. We could move this distinction
            // up to the ast1:: level, but then we would have 4 different variants to handle
            // here.
            Assign(_, _) | AssignOp(_, _, _) => return err!("{}", "invalid assignment expression"),
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
        self.cfg.add_edge(current_open, h, Transition::null());
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
        // Note, to be cautiouss we could insert Phis for all identifiers.
        // But that would introduce additional nodes for variables that are assigned to only once
        // by construction. Instead we only use named variables. Of course, we this to change we
        // would need to fall back on something more conservative.
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
