use super::hashbrown::HashMap;
use std::hash::Hash;

#[derive(Debug, Copy, Clone)]
pub(crate) enum NumUnop {
    Column, // $
    Not,    // !
    Neg,    // -
    Pos,    // +
}

// TODO(ezr) builtins?
#[derive(Debug, Copy, Clone)]
pub(crate) enum StrUnop {}

#[derive(Debug, Copy, Clone)]
pub(crate) enum NumBinop {
    Plus,
    Minus,
    Mult,
    Div,
    Mod,
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum StrBinop {
    Concat,
    Match,
}

// TODO perform some manual tests
// TODO SSA conversion:
// * Compute Dominator Tree
// * Migrate identifier representation to tuple of numbers (for subscripts)
// * Rename and insert Phi functions

pub(crate) mod ast1 {
    use super::*;

    #[derive(Debug)]
    pub(crate) enum Expr<'a, 'b, I> {
        NumLit(f64),
        StrLit(&'b str),
        Unop(Result<NumUnop, StrUnop>, &'a Expr<'a, 'b, I>),
        Binop(
            Result<NumBinop, StrBinop>,
            &'a Expr<'a, 'b, I>,
            &'a Expr<'a, 'b, I>,
        ),
        Var(I),
        Index(&'a Expr<'a, 'b, I>, &'a Expr<'a, 'b, I>),
        Assign(
            &'a Expr<'a, 'b, I>, /*var or index expression*/
            &'a Expr<'a, 'b, I>,
        ),
        AssignOp(&'a Expr<'a, 'b, I>, NumBinop, &'a Expr<'a, 'b, I>),
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
    }
}

use petgraph::graph::Graph;

pub(crate) mod ast2 {
    use super::*;
    // TODO: split out dominator calculation to its own module.
    // TODO: split all of these modules into their own files.

    // Several utility functions operate on both NodeIx and NumTy values. Making them polymorphic
    // on HasNum automates the conversion between these two types.
    trait HasNum: Copy {
        #[inline]
        fn ix(self) -> usize {
            self.num() as usize
        }

        #[inline]
        fn num(self) -> NumTy {
            const _MAX_NUMTY: NumTy = !0;
            debug_assert!((_MAX_NUMTY as usize) >= self.ix());
            self.ix() as NumTy
        }
    }

    impl HasNum for NumTy {
        #[inline]
        fn num(self) -> NumTy {
            self
        }
    }

    impl HasNum for NodeIx {
        #[inline]
        fn ix(self) -> usize {
            self.index()
        }
    }

    #[derive(Default)]
    pub(crate) struct Context<'b, I> {
        hm: HashMap<I, Ident>,
        max: NumTy,
        cfg: CFG<'b>,
        entry: NodeIx,
    }

    pub(crate) struct NodeInfo {
        // order reached in DFS.
        dfsnum: NumTy,
        // immediate dominator
        idom: NumTy,
        // semidominator,
        sdom: NumTy,
        // parent in spanning tree
        parent: NumTy,
        // ancestor in spanning forest during semidominator calculation.
        // TODO: The LLVM implementation combines parents and ancestors, and instead passes in a
        // "Last Visited" dfsnum into `eval` in order to skip nodes that haven't been linked into
        // the tree. Make that change once confident in the initial one.
        ancestor: NumTy,
    }

    impl Default for NodeInfo {
        fn default() -> NodeInfo {
            NodeInfo {
                dfsnum: NODEINFO_UNINIT,
                idom: NODEINFO_UNINIT,
                sdom: NODEINFO_UNINIT,
                parent: NODEINFO_UNINIT,
                ancestor: NODEINFO_UNINIT,
            }
        }
    }

    impl NodeInfo {
        fn seen(&self) -> bool {
            self.dfsnum != NODEINFO_UNINIT
        }
    }
    const NODEINFO_UNINIT: NumTy = !0;

    // TODO: explain dominator tree algorithm choice, (why not the one in PetGraph?)
    // TODO: consider building a safe-index API around "vector whose length is the same as this
    // graph and will not change" and generative-lifetime style safe index type, so as to avoid
    // bounds-checks.
    struct DomTreeBuilder<'a, 'b, I> {
        // Underlying program context
        ctx: &'a Context<'b, I>,
        // Semi-NCA metadata, indexed by NodeIndex
        info: Vec<NodeInfo>,
        // (pre-order) depth-first ordering of nodes.
        dfs: Vec<NodeIx>,
        // Used in semidominator calculation.
        // ancestor: Vec<NumTy>,
        best: Vec<NumTy>,
    }
    pub(crate) fn dom_tree<'a, I>(ctx: &Context<'a, I>) -> Vec<NodeInfo> {
        DomTreeBuilder::new(ctx).tree()
    }

    impl<'a, 'b, I> DomTreeBuilder<'a, 'b, I> {
        fn new(ctx: &'a Context<'b, I>) -> Self {
            DomTreeBuilder {
                ctx: ctx,
                info: (0..ctx.cfg.node_count())
                    .map(|_| Default::default())
                    .collect(),
                dfs: Default::default(),
                best: (0..ctx.cfg.node_count()).map(|_| NODEINFO_UNINIT).collect(),
            }
        }
        fn num_nodes(&self) -> NumTy {
            debug_assert_eq!(self.ctx.cfg.node_count(), self.info.len());
            self.info.len() as NumTy
        }
        fn seen(&self) -> NumTy {
            self.dfs.len() as NumTy
        }
        fn at(&self, ix: impl HasNum) -> &NodeInfo {
            &self.info[ix.ix()]
        }
        fn best_at(&self, ix: impl HasNum) -> NumTy {
            self.best[ix.ix()]
        }
        fn set_best(&mut self, n: impl HasNum, v: impl HasNum) {
            self.best[n.ix()] = v.num();
        }
        fn at_mut(&mut self, ix: impl HasNum) -> &mut NodeInfo {
            &mut self.info[ix.ix()]
        }
        fn link(&mut self, parent: impl HasNum, node: impl HasNum) {
            *(&mut self.at_mut(parent).ancestor) = parent.num();
            self.set_best(node, node);
        }

        // a.k.a AncestorWithLowestSemi in Tiger Book
        fn eval(&mut self, node: impl HasNum) -> NumTy {
            let p = self.at(node).ancestor;
            if self.at(p).ancestor != NODEINFO_UNINIT {
                let b = self.eval(p);
                *(&mut self.at_mut(node).ancestor) = self.at(p).ancestor;
                if self.at(self.at(b).sdom).dfsnum
                    < self.at(self.at(self.best_at(node)).sdom).dfsnum
                {
                    self.set_best(node, b);
                }
            }
            self.best_at(node)
        }

        fn semis(&mut self) {
            // We need to borrow self.dfs, but also other parts of the struct, so we swap it out.
            // Note that this only works because we know that `eval` and `link` do not use the
            // `dfs` vector.
            let dfs = std::mem::replace(&mut self.dfs, Default::default());
            for n in dfs[1..].iter().rev().map(|x| *x) {
                let parent = self.at(n).parent;
                let mut semi = parent;
                for pred in self
                    .ctx
                    .cfg
                    .neighbors_directed(NodeIx::new(n.ix()), petgraph::Direction::Incoming)
                {
                    let candidate = if self.at(pred).dfsnum <= self.at(n).dfsnum {
                        pred.num()
                    } else {
                        let ancestor_with_lowest = self.eval(pred);
                        self.at(ancestor_with_lowest).sdom
                    };
                    if self.at(candidate).dfsnum < self.at(semi).dfsnum {
                        semi = candidate
                    }
                }
                *(&mut self.at_mut(n).sdom) = semi;
                self.link(parent, n);
            }
            std::mem::replace(&mut self.dfs, dfs);
        }

        fn dfs(&mut self, cur_node: NodeIx, parent: NumTy) {
            // TODO: consider explicit maintenance of stack.
            //       probably not a huge win performance-wise, but it could avoid stack overflow on
            //       pathological inputs.
            debug_assert!(!self.at(cur_node).seen());
            let seen_so_far = self.seen();
            *self.at_mut(cur_node) = NodeInfo {
                dfsnum: seen_so_far,
                parent: parent,
                idom: parent, // provisional
                sdom: NODEINFO_UNINIT,
                ancestor: NODEINFO_UNINIT,
            };
            self.dfs.push(cur_node);
            // NB assumes that CFG is fully connected.
            for n in self
                .ctx
                .cfg
                .neighbors_directed(cur_node, petgraph::Direction::Outgoing)
            {
                if self.seen() == self.num_nodes() {
                    break;
                }
                if self.at(n).seen() {
                    continue;
                }
                self.dfs(n, cur_node.index() as NumTy);
            }
        }

        fn idoms(&mut self) {
            let dfs = std::mem::replace(&mut self.dfs, Default::default());
            for n in dfs[1..].iter().map(|x| *x) {
                let (mut idom, semi) = {
                    let entry = self.at(n);
                    (entry.idom, entry.sdom)
                };
                while idom > semi {
                    idom = self.at(idom).idom;
                }
                (&mut self.at_mut(n)).idom = idom;
            }
            std::mem::replace(&mut self.dfs, dfs);
        }
        fn tree(mut self) -> Vec<NodeInfo> {
            self.dfs(self.ctx.entry, NODEINFO_UNINIT);
            self.semis();
            self.idoms();
            self.info
        }
    }

    // consider making this just "by number" and putting branch instructions elsewhere.
    // need to verify the order
    type BasicBlock<'a> = V<PrimStmt<'a>>;
    // None indicates `else`
    type CFG<'a> = Graph<BasicBlock<'a>, Option<PrimVal<'a>>, petgraph::Directed, NumTy>;
    type NumTy = u32;
    type Ident = (NumTy, NumTy); // change to u64?
    type V<T> = Vec<T>; // change to smallvec?

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

    pub type NodeIx = petgraph::graph::NodeIndex<NumTy>;

    impl<'b, I: Hash + Eq + Clone + Default> Context<'b, I> {
        pub fn cfg(&self) -> &CFG<'b> {
            &self.cfg
        }
        pub fn entry(&self) -> NodeIx {
            self.entry
        }
        fn from_stmt<'a>(stmt: &'a ast1::Stmt<'a, 'b, I>) -> Self {
            let mut ctx = Self::default();
            let (start, _) = ctx.standalone_block(stmt);
            ctx.entry = start;
            ctx
        }
        pub fn standalone_block<'a>(
            &mut self,
            stmt: &'a ast1::Stmt<'a, 'b, I>,
        ) -> (NodeIx /*start*/, NodeIx /*end*/) {
            let start = self.cfg.add_node(V::default());
            let end = self.convert_stmt(stmt, start);
            (start, end)
        }
        fn convert_stmt<'a>(
            &mut self,
            stmt: &'a ast1::Stmt<'a, 'b, I>,
            mut current_open: NodeIx,
        ) -> NodeIx /*next open */ {
            // need "current open basic block"
            use ast1::Stmt::*;
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
                    let next = self.cfg.add_node(V::default());

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
                    let array_iter =
                        self.to_val(PrimExpr::IterBegin(array_val.clone()), current_open);

                    // First, create the loop header, which checks if there are any more elements
                    // in the array.
                    let cond = PrimExpr::HasNext(array_iter.clone());
                    let cond_block = self.cfg.add_node(V::default());
                    let cond_v = self.to_val(cond, cond_block);
                    self.cfg.add_edge(current_open, cond_block, None);

                    // Create the body, but start by getting the next element from the iterator and
                    // assigning it to `v`
                    let update = PrimStmt::AsgnVar(v_id, PrimExpr::Next(array_iter.clone()));
                    let body_start = self.cfg.add_node(V::default());
                    self.add_stmt(body_start, update);
                    let body_end = self.convert_stmt(body, body_start);
                    self.cfg.add_edge(cond_block, body_start, Some(cond_v));
                    self.cfg.add_edge(body_end, cond_block, None);

                    // Then add a footer to exit the loop from cond.
                    let footer = self.cfg.add_node(V::default());
                    self.cfg.add_edge(cond_block, footer, None);

                    footer
                }
            }
        }

        fn convert_expr<'a>(
            &mut self,
            expr: &'a ast1::Expr<'a, 'b, I>,
            current_open: NodeIx,
        ) -> PrimExpr<'b> /* should not create any new nodes. Expressions don't cause us to branch */
        {
            use ast1::Expr::*;
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
                        let arr_cell_v =
                            slf.to_val(PrimExpr::Index(arr_v, ix_v.clone()), current_open);
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
            arr: &'a ast1::Expr<'a, 'b, I>,
            ix: &'a ast1::Expr<'a, 'b, I>,
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

        fn convert_val<'a>(
            &mut self,
            expr: &'a ast1::Expr<'a, 'b, I>,
            current_open: NodeIx,
        ) -> PrimVal<'b> {
            let e = self.convert_expr(expr, current_open);
            self.to_val(e, current_open)
        }

        fn make_loop<'a>(
            &mut self,
            body: &'a ast1::Stmt<'a, 'b, I>,
            update: Option<&'a ast1::Stmt<'a, 'b, I>>,
            current_open: NodeIx,
        ) -> (
            NodeIx, /* header */
            NodeIx, /* body header */
            NodeIx, /* body footer */
            NodeIx, /* footer = next open */
        ) {
            // Create header, body, and footer nodes.
            let h = self.cfg.add_node(V::default());
            let (b_start, b_end) = if let Some(u) = update {
                let (start, mid) = self.standalone_block(body);
                let end = self.convert_stmt(u, mid);
                (start, end)
            } else {
                self.standalone_block(body)
            };
            let f = self.cfg.add_node(V::default());
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
            self.cfg.node_weight_mut(at).unwrap().push(stmt);
        }
    }
}
// AST1: normalize identifier => array index, along with map pointing back (maybe just do this as
// part of the AST2 conversion)
// AST2 (graph):
//  As part of conversion to AST2
//  * Desugar For/While/Foreach/(If?) into conditional jump (with new ITER_BEGIN and ITER_END expressions)
//  * Normalize expressions to only contain primitive statements
//  AST2 => AST2 passes
//  * SSA conversion
//  * type analysis
//
