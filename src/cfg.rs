use crate::arena;
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

#[derive(Debug, Default)]
pub(crate) struct BasicBlock<'a> {
    pub q: VecDeque<PrimStmt<'a>>,
    pub sealed: bool,
}

// None indicates `else`
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

pub(crate) type CFG<'a> = Graph<BasicBlock<'a>, Transition<'a>>;
#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub(crate) struct Ident {
    pub(crate) low: NumTy,
    pub(crate) sub: NumTy,
    // Whether or not something is global in other modules depends on whether or not it is a "local
    // global", i.e. a global that is only referenced from main.
    global: bool,
}

impl Ident {
    fn new_global(low: NumTy) -> Ident {
        Ident {
            low,
            sub: 0,
            global: true,
        }
    }
    fn new_local(low: NumTy) -> Ident {
        Ident {
            low,
            sub: 0,
            global: false,
        }
    }

    pub(crate) fn is_global(&self, local_globals: &HashSet<NumTy>) -> bool {
        self.global && local_globals.get(&self.low).is_none()
    }

    // used in some test programs to normalize Idents by replacing their subscript with 0
    pub(crate) fn _base(&self) -> Ident {
        Ident {
            low: self.low,
            sub: 0,
            global: self.global,
        }
    }
}

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
    CallUDF(NumTy, SmallVec<PrimVal<'a>>),
    Index(PrimVal<'a>, PrimVal<'a>),

    // For iterating over vectors.
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
    Return(PrimVal<'a>),
    IterDrop(PrimVal<'a>),
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
            CallBuiltin(_, args) | CallUDF(_, args) => {
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
    // TODO: remove pub(crate) once we consolidate the cfg modules
    pub(crate) fn replace(&mut self, mut update: impl FnMut(Ident) -> Ident) {
        use PrimStmt::*;
        match self {
            AsgnIndex(ident, v, exp) => {
                *ident = update(*ident);
                v.replace(&mut update);
                exp.replace(update);
            }
            // We handle assignments separately. Note that this is not needed for index
            // expressions, because assignments to m[k] are *uses* of m; it doesn't assign to it.
            AsgnVar(_, e) => e.replace(update),
            SetBuiltin(_, e) => e.replace(update),
            IterDrop(v) | Return(v) => v.replace(update),
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

#[derive(Debug)]
pub(crate) struct ProgramContext<'a, I> {
    shared: GlobalContext<I>,
    // Functions "know" which Option<Ident> maps to which offset in this
    // table at construction time (in the func_table passed to View).
    pub funcs: Vec<Function<'a, I>>,
    pub main_offset: usize,
}

impl<'a, I: Hash + Eq + Clone + Default + std::fmt::Display + std::fmt::Debug> ProgramContext<'a, I>
where
    builtins::Variable: TryFrom<I>,
    builtins::Function: TryFrom<I>,
{
    pub(crate) fn local_globals(&mut self) -> HashSet<NumTy> {
        std::mem::replace(&mut self.shared.local_globals, Default::default())
    }
    pub(crate) fn local_globals_ref(&self) -> &HashSet<NumTy> {
        &self.shared.local_globals
    }

    // for debugging: get a mapping from the raw identifiers to the synthetic ones.
    pub(crate) fn _invert_ident(&self) -> HashMap<Ident, I> {
        self.shared
            .hm
            .iter()
            .map(|(k, v)| (v.clone(), k.clone()))
            .collect()
    }

    pub(crate) fn from_prog<'b, 'outer>(
        arena: &'a arena::Arena<'outer>,
        p: &ast::Prog<'a, 'b, I>,
    ) -> Result<Self> {
        let mut shared: GlobalContext<I> = GlobalContext {
            hm: Default::default(),
            local_globals: Default::default(),
            may_rename: Default::default(),
            max: 1, // 0 reserved for assigning to "unused" var for side-effecting operations
        };
        let mut func_table: HashMap<Option<I>, NumTy> = Default::default();
        let mut funcs: Vec<Function<'a, I>> = Default::default();
        for fundec in p.decs.iter() {
            if let Some(_) = func_table.insert(Some(fundec.name.clone()), funcs.len() as NumTy) {
                return err!("duplicate function found for name {}", fundec.name);
            }
            if let Ok(bi) = builtins::Function::try_from(fundec.name.clone()) {
                return err!("attempted redefinition of builtin function {}", bi);
            }
            let mut ix = 0;
            let mut args_map = HashMap::new();
            let mut cfg = CFG::default();
            let entry = cfg.add_node(Default::default());

            // All exit blocks simply return the designated return node. Return statements in the
            // AST will becode assignments to this variable followed by an unconditional jump to
            // this block.
            let ret = shared.fresh_local();
            shared.may_rename.push(ret);
            let exit = cfg.add_node(Default::default());

            let f = Function {
                name: Some(fundec.name.clone()),
                ident: funcs.len() as NumTy,
                args: fundec
                    .args
                    .iter()
                    .map(|i| {
                        let name = i.clone();
                        let id = shared.fresh_local();
                        args_map.insert(i.clone(), ix);
                        ix += 1;
                        Arg { name, id }
                    })
                    .collect(),
                args_map,
                ret,
                cfg,
                defsites: Default::default(),
                orig: Default::default(),
                entry,
                exit,
                loop_ctx: Default::default(),
                dt: Default::default(),
                df: Default::default(),
            };
            funcs.push(f);
        }
        // Bind the main function
        let main_stmt = arena.alloc_v(p.desugar(arena));
        let mut cfg = CFG::default();
        let entry = cfg.add_node(Default::default());
        let exit = cfg.add_node(Default::default());
        let mut main_func = Function {
            name: None,
            ident: funcs.len() as NumTy,
            args: Default::default(),
            args_map: Default::default(),
            ret: View::<'a, 'a, I>::unused(),
            cfg,
            defsites: Default::default(),
            orig: Default::default(),
            entry,
            exit,
            loop_ctx: Default::default(),
            dt: Default::default(),
            df: Default::default(),
        };
        // Now that we have all the functions in place, it's time to fill them up and convert them
        // to SSA.
        View {
            ctx: &mut shared,
            f: &mut main_func,
            func_table: &func_table,
        }
        .fill(main_stmt)?;
        let main_offset = funcs.len();
        func_table.insert(None, main_offset as NumTy);
        funcs.push(main_func);
        for fundec in p.decs.iter() {
            let f = *func_table.get_mut(&Some(fundec.name.clone())).unwrap();
            View {
                ctx: &mut shared,
                f: funcs.get_mut(f as usize).unwrap(),
                func_table: &func_table,
            }
            .fill(fundec.body)?;
        }
        Ok(ProgramContext {
            shared,
            funcs,
            main_offset,
        })
    }
}

struct View<'a, 'b, I> {
    ctx: &'a mut GlobalContext<I>,
    f: &'a mut Function<'b, I>,
    func_table: &'a HashMap<Option<I>, NumTy>,
}

#[derive(Debug)]
struct GlobalContext<I> {
    // Map the identifiers from the AST to this IR's Idents.
    hm: HashMap<I, Ident>,
    // Global identifiers to rewrite global => local. We only store the `low` field of the
    // identifier.
    local_globals: HashSet<NumTy>,

    // Many identifiers are generated and assigned to only once by construction, so we do not add
    // them to the work list for renaming. All named identifiers are added, as well as the ones
    // created during evaluation of conditional expression (?:, and, or). Doing this saves us not
    // only time, but also extra phi nodes in the IR, which eventually lead to more assignments
    // than are required.
    // TODO: make may_rename per-function for local variables.
    may_rename: Vec<Ident>,
    max: NumTy,
}

impl<I> GlobalContext<I> {
    fn fresh(&mut self) -> Ident {
        let res = self.max;
        self.max += 1;
        Ident::new_global(res)
    }

    fn fresh_local(&mut self) -> Ident {
        let res = self.max;
        self.max += 1;
        Ident::new_local(res)
    }
}

#[derive(Debug)]
pub(crate) struct Arg<I> {
    pub name: I,
    pub id: Ident,
}

#[derive(Debug)]
pub(crate) struct Function<'a, I> {
    pub name: Option<I>,
    pub ident: NumTy,
    // args_map maps from ast-level ident to an index into args.
    args_map: HashMap<I, NumTy>,
    pub args: SmallVec<Arg<I>>,
    ret: Ident,
    pub cfg: CFG<'a>,

    defsites: HashMap<Ident, HashSet<NodeIx>>,
    orig: HashMap<NodeIx, HashSet<Ident>>,
    entry: NodeIx,
    // We enforce that a single basic block has a return statement. This is to ensure that type
    // inference infers the same type for each return site.
    pub exit: NodeIx,
    // Stack of the entry and exit nodes for the loops within which the current statement is
    // nested.
    loop_ctx: SmallVec<(NodeIx, NodeIx)>,

    // Dominance information about `cfg`.
    dt: dom::Tree,
    df: dom::Frontier,
}

pub(crate) fn is_unused(i: Ident) -> bool {
    i.low == 0
}

impl<'a, 'b, I: Hash + Eq + Clone + Default + std::fmt::Display + std::fmt::Debug> View<'a, 'b, I>
where
    builtins::Variable: TryFrom<I>,
    builtins::Function: TryFrom<I>,
{
    fn fill<'c>(&mut self, stmt: &'c Stmt<'c, 'b, I>) -> Result<()> {
        // Add a CFG corresponding to `stmt`
        let _next = self.convert_stmt(stmt, self.f.entry)?;
        // Insert edges to the exit nodes if where they do not exist
        self.finish()?;
        // SSA Conversion:
        // 1. Compute the dominator tree and dominance frontiers
        let (dt, df) = {
            let di = dom::DomInfo::new(&self.f.cfg, self.f.entry);
            (di.dom_tree(), di.dom_frontier())
        };
        self.f.dt = dt;
        self.f.df = df;
        // 2. Insert phi nodes where appropriate
        self.insert_phis();
        // 3. Rename variables
        self.rename(self.f.entry);
        Ok(())
    }

    // We want to make sure there is a single exit node. This method adds an unconditional branch
    // to the end of each basic block without one already to the exit node.
    fn finish(&mut self) -> Result<()> {
        for bb in self.f.cfg.node_indices() {
            if bb == self.f.exit {
                continue;
            }
            let mut found = false;
            let mut walker = self.f.cfg.neighbors(bb).detach();
            while let Some((e, _)) = walker.next(&self.f.cfg) {
                if let Transition(None) = self.f.cfg.edge_weight(e).unwrap() {
                    found = true;
                }
            }
            if !found {
                self.f.cfg.add_edge(bb, self.f.exit, Transition::null());
            }
        }
        // Add a return statement to the exit node.
        self.add_stmt(self.f.exit, PrimStmt::Return(PrimVal::Var(self.f.ret)))
    }

    fn unused() -> Ident {
        Ident::new_global(0)
    }

    fn standalone_expr<'c>(
        &mut self,
        expr: &'c Expr<'c, 'b, I>,
    ) -> Result<(NodeIx /*start*/, NodeIx /*end*/, PrimExpr<'b>)> {
        let start = self.f.cfg.add_node(Default::default());
        let (end, res) = self.convert_expr(expr, start)?;
        Ok((start, end, res))
    }

    pub fn standalone_block<'c>(
        &mut self,
        stmt: &'c Stmt<'c, 'b, I>,
    ) -> Result<(NodeIx /*start*/, NodeIx /*end*/)> {
        let start = self.f.cfg.add_node(Default::default());
        let end = self.convert_stmt(stmt, start)?;
        Ok((start, end))
    }

    fn convert_stmt<'c>(
        &mut self,
        stmt: &'c Stmt<'c, 'b, I>,
        mut current_open: NodeIx,
    ) -> Result<NodeIx> /*next open */ {
        use Stmt::*;
        Ok(match stmt {
            Expr(e) => {
                // We need to assign to unused here, otherwise we could generate the expression but
                // then drop it on the floor.
                let (next, e) = self.convert_expr(e, current_open)?;
                self.add_stmt(next, PrimStmt::AsgnVar(Self::unused(), e))?;
                next
            }
            Block(stmts) => {
                for s in stmts {
                    current_open = self.convert_stmt(s, current_open)?;
                }
                current_open
            }
            Print(vs, out) => {
                if vs.len() == 0 {
                    return self.convert_stmt(
                        &ast::Stmt::Print(
                            vec![&ast::Expr::Unop(Unop::Column, &ast::Expr::ILit(0))],
                            out.clone(),
                        ),
                        current_open,
                    );
                }
                let (next, out) = if let Some((o, append)) = out {
                    let (next, e) = self.convert_val(o, current_open)?;
                    (next, Some((e, append)))
                } else {
                    (current_open, None)
                };
                current_open = next;
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
                    let tmp = self.fresh_local();
                    self.add_stmt(
                        current_open,
                        PrimStmt::AsgnVar(
                            tmp,
                            PrimExpr::CallBuiltin(
                                builtins::Function::Unop(Unop::Column),
                                smallvec![PrimVal::ILit(0)],
                            ),
                        ),
                    )?;
                    self.add_stmt(
                        current_open,
                        PrimStmt::AsgnVar(Self::unused(), print(PrimVal::Var(tmp))),
                    )?;
                    current_open
                } else if vs.len() == 1 {
                    let (next, v) = self.convert_val(vs[0], current_open)?;
                    current_open = next;
                    self.add_stmt(current_open, PrimStmt::AsgnVar(Self::unused(), print(v)))?;
                    current_open
                } else {
                    const EMPTY: PrimVal<'static> = PrimVal::StrLit("");

                    // Assign the field separator to a local variable.
                    let fs = {
                        let fs = self.fresh_local();
                        self.add_stmt(
                            current_open,
                            PrimStmt::AsgnVar(
                                fs.clone(),
                                PrimExpr::LoadBuiltin(builtins::Variable::OFS),
                            ),
                        )?;
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
                    let mut tmp = self.fresh_local();
                    self.add_stmt(current_open, PrimStmt::AsgnVar(tmp, PrimExpr::Val(EMPTY)))?;
                    for (i, v) in vs.iter().enumerate() {
                        let (next, v) = self.convert_val(*v, current_open)?;
                        current_open = next;
                        if i != 0 {
                            let new_tmp = self.fresh_local();
                            self.add_stmt(
                                current_open,
                                PrimStmt::AsgnVar(
                                    new_tmp,
                                    PrimExpr::CallBuiltin(
                                        builtins::Function::Binop(Binop::Concat),
                                        smallvec![PrimVal::Var(tmp), fs.clone()],
                                    ),
                                ),
                            )?;
                            tmp = new_tmp;
                        }
                        let new_tmp = self.fresh_local();
                        self.add_stmt(
                            current_open,
                            PrimStmt::AsgnVar(
                                new_tmp,
                                PrimExpr::CallBuiltin(
                                    builtins::Function::Binop(Binop::Concat),
                                    smallvec![PrimVal::Var(tmp), v],
                                ),
                            ),
                        )?;
                        tmp = new_tmp;
                    }
                    self.add_stmt(
                        current_open,
                        PrimStmt::AsgnVar(Self::unused(), print(PrimVal::Var(tmp))),
                    )?;

                    current_open
                }
            }
            If(cond, tcase, fcase) => {
                let tcase = self.standalone_block(tcase)?;
                let fcase = if let Some(fcase) = fcase {
                    Some(self.standalone_block(fcase)?)
                } else {
                    None
                };
                self.do_condition(cond, tcase, fcase, current_open)?
            }
            For(init, cond, update, body) => {
                let init_end = if let Some(i) = init {
                    self.convert_stmt(i, current_open)?
                } else {
                    current_open
                };
                let (h, b_start, _b_end, f) =
                    self.make_loop(body, update.clone(), init_end, false)?;
                let (h_end, cond_val) = if let Some(c) = cond {
                    self.convert_val(c, h)?
                } else {
                    (h, PrimVal::ILit(1))
                };
                self.f
                    .cfg
                    .add_edge(h_end, b_start, Transition::new(cond_val));
                self.f.cfg.add_edge(h_end, f, Transition::null());
                f
            }
            While(cond, body) => {
                let (h, b_start, _b_end, f) = self.make_loop(body, None, current_open, false)?;
                let (h_end, cond_val) = self.convert_val(cond, h)?;
                self.f
                    .cfg
                    .add_edge(h_end, b_start, Transition::new(cond_val));
                self.f.cfg.add_edge(h_end, f, Transition::null());
                f
            }
            DoWhile(cond, body) => {
                let (h, b_start, _b_end, f) = self.make_loop(body, None, current_open, true)?;
                let (h_end, cond_val) = self.convert_val(cond, h)?;
                self.f
                    .cfg
                    .add_edge(h_end, b_start, Transition::new(cond_val));
                self.f.cfg.add_edge(h_end, f, Transition::null());
                f
            }
            ForEach(v, array, body) => {
                let v_id = self.get_identifier(v);
                let (next, array_val) = self.convert_val(array, current_open)?;
                current_open = next;
                let array_iter =
                    self.to_val(PrimExpr::IterBegin(array_val.clone()), current_open)?;

                // First, create the loop header, which checks if there are any more elements
                // in the array.
                let cond = PrimExpr::HasNext(array_iter.clone());
                let cond_block = self.f.cfg.add_node(Default::default());
                let cond_v = self.to_val(cond, cond_block)?;
                self.f
                    .cfg
                    .add_edge(current_open, cond_block, Transition::null());

                // Then add a footer to exit the loop from cond. We will add the edge after adding
                // the edge into the loop body, as order matters.
                let footer = self.f.cfg.add_node(Default::default());
                self.add_stmt(footer, PrimStmt::IterDrop(array_iter.clone()))?;

                self.f.loop_ctx.push((cond_block, footer));

                // Create the body, but start by getting the next element from the iterator and
                // assigning it to `v`
                let update = PrimStmt::AsgnVar(v_id, PrimExpr::Next(array_iter.clone()));
                let body_start = self.f.cfg.add_node(Default::default());
                self.add_stmt(body_start, update)?;
                let body_end = self.convert_stmt(body, body_start)?;
                self.f
                    .cfg
                    .add_edge(cond_block, body_start, Transition::new(cond_v));
                self.f.cfg.add_edge(cond_block, footer, Transition::null());
                self.f
                    .cfg
                    .add_edge(body_end, cond_block, Transition::null());

                self.f.loop_ctx.pop().unwrap();

                footer
            }
            // TODO we may want checking here to avoid folks doing "continue" and "break" inside a
            // toplevel statement that was desugared into a loop.
            //
            // This should become more doable once we add more metadata to AST nodes.
            Break => {
                match self.f.loop_ctx.pop() {
                    Some((_, footer)) => {
                        // Break statements unconditionally jump to the end of the loop.
                        self.f
                            .cfg
                            .add_edge(current_open, footer, Transition::null());
                        self.seal(current_open);
                        current_open
                    }
                    None => {
                        return Err(CompileError("break statement must be inside a loop".into()))
                    }
                }
            }
            Continue => {
                match self.f.loop_ctx.pop() {
                    Some((header, _)) => {
                        // Continue statements unconditionally jump to the top of the loop.
                        self.f
                            .cfg
                            .add_edge(current_open, header, Transition::null());
                        self.seal(current_open);
                        current_open
                    }
                    None => {
                        return Err(CompileError(
                            "continue statement must be inside a loop".into(),
                        ))
                    }
                }
            }
            Return(ret) => {
                let (current_open, e) = if let Some(ret) = ret {
                    self.convert_expr(ret, current_open)?
                } else {
                    (current_open, PrimExpr::Val(PrimVal::Var(Self::unused())))
                };
                self.add_stmt(current_open, PrimStmt::AsgnVar(self.f.ret, e))?;
                self.f
                    .cfg
                    .add_edge(current_open, self.f.exit, Transition::null());
                self.seal(current_open);
                current_open
            }
        })
    }

    fn convert_expr<'c>(
        &mut self,
        expr: &'c Expr<'c, 'b, I>,
        current_open: NodeIx,
    ) -> Result<(NodeIx, PrimExpr<'b>)> {
        use Expr::*;

        // This isn't a function because we want the &s to point to the current stack frame.
        macro_rules! to_bool {
            ($e:expr) => {
                &Unop(ast::Unop::Not, &Unop(ast::Unop::Not, $e))
            };
        }
        let res_expr = match expr {
            ILit(n) => PrimExpr::Val(PrimVal::ILit(*n)),
            FLit(n) => PrimExpr::Val(PrimVal::FLit(*n)),
            PatLit(s) | StrLit(s) => PrimExpr::Val(PrimVal::StrLit(s)),
            Unop(op, e) => {
                let (next, v) = self.convert_val(e, current_open)?;
                return Ok((
                    next,
                    PrimExpr::CallBuiltin(builtins::Function::Unop(*op), smallvec![v]),
                ));
            }
            Binop(op, e1, e2) => {
                let (next, v1) = self.convert_val(e1, current_open)?;
                let (next, v2) = self.convert_val(e2, next)?;
                return Ok((
                    next,
                    PrimExpr::CallBuiltin(builtins::Function::Binop(*op), smallvec![v1, v2]),
                ));
            }
            ITE(cond, tcase, fcase) => {
                let res_id = self.fresh_local();
                self.ctx.may_rename.push(res_id);
                let (tstart, tend, te) = self.standalone_expr(tcase)?;
                let (fstart, fend, fe) = self.standalone_expr(fcase)?;
                self.add_stmt(tend, PrimStmt::AsgnVar(res_id, te))?;
                self.add_stmt(fend, PrimStmt::AsgnVar(res_id, fe))?;
                let next =
                    self.do_condition(cond, (tstart, tend), Some((fstart, fend)), current_open)?;
                return Ok((next, PrimExpr::Val(PrimVal::Var(res_id))));
            }
            And(e1, e2) => {
                return self.convert_expr(&ITE(e1, to_bool!(e2), &ILit(0)), current_open)
            }
            Or(e1, e2) => return self.convert_expr(&ITE(e1, &ILit(1), to_bool!(e2)), current_open),
            Var(id) => {
                if let Ok(bi) = builtins::Variable::try_from(id.clone()) {
                    PrimExpr::LoadBuiltin(bi)
                } else {
                    let ident = self.get_identifier(id);
                    PrimExpr::Val(PrimVal::Var(ident))
                }
            }
            Index(arr, ix) => {
                let (next, arr_v) = self.convert_val(arr, current_open)?;
                let (next, ix_v) = self.convert_val(ix, next)?;
                return Ok((next, PrimExpr::Index(arr_v, ix_v)));
            }
            Call(fname, args) => {
                let bi = match fname {
                    Either::Left(fname) => {
                        if let Ok(bi) = builtins::Function::try_from(fname.clone()) {
                            Either::Right(bi)
                        } else {
                            Either::Left(fname.clone())
                        }
                    }
                    Either::Right(bi) => Either::Right(*bi),
                };
                let mut prim_args = SmallVec::with_capacity(args.len());
                let mut open = current_open;
                for a in args.iter() {
                    let (next, v) = self.convert_val(a, open)?;
                    open = next;
                    prim_args.push(v);
                }
                match bi {
                    Either::Left(fname) => {
                        return if let Some(i) = self.func_table.get(&Some(fname.clone())) {
                            Ok((open, PrimExpr::CallUDF(*i, prim_args)))
                        } else {
                            err!("Call to unknown function \"{}\"", fname)
                        };
                    }
                    Either::Right(bi) => {
                        if let builtins::Function::Split = bi {
                            if prim_args.len() == 2 {
                                let fs = self.fresh_local();
                                self.add_stmt(
                                    current_open,
                                    PrimStmt::AsgnVar(
                                        fs.clone(),
                                        PrimExpr::LoadBuiltin(builtins::Variable::OFS),
                                    ),
                                )?;
                                prim_args.push(PrimVal::Var(fs));
                            }
                        }
                        return Ok((open, PrimExpr::CallBuiltin(bi, prim_args)));
                    }
                }
            }
            Assign(Index(arr, ix), to) => {
                return self.do_assign_index(
                    arr,
                    ix,
                    |slf, _, _, open| slf.convert_expr(to, open),
                    current_open,
                )
            }

            AssignOp(Index(arr, ix), op, to) => {
                return self.do_assign_index(
                    arr,
                    ix,
                    |slf, arr_v, ix_v, open| {
                        let (next, to_v) = slf.convert_val(to, open)?;
                        let arr_cell_v = slf.to_val(PrimExpr::Index(arr_v, ix_v.clone()), next)?;
                        Ok((
                            next,
                            PrimExpr::CallBuiltin(
                                builtins::Function::Binop(*op),
                                smallvec![arr_cell_v, to_v],
                            ),
                        ))
                    },
                    current_open,
                )
            }
            Assign(x, to) => {
                let (next, to) = self.convert_expr(to, current_open)?;
                return self.do_assign(x, |_| to, next);
            }
            AssignOp(x, op, to) => {
                let (next, to_v) = self.convert_val(to, current_open)?;
                return self.do_assign(
                    x,
                    |v| {
                        PrimExpr::CallBuiltin(
                            builtins::Function::Binop(*op),
                            smallvec![v.clone(), to_v],
                        )
                    },
                    next,
                );
            }
            Inc { is_inc, is_post, x } => {
                if !valid_lhs(x) {
                    return err!("invalid operand for increment operation {:?}", x);
                };
                let (next, pre) = if *is_post {
                    let (next, e) = self.convert_expr(x, current_open)?;
                    let f = self.fresh_local();
                    self.add_stmt(next, PrimStmt::AsgnVar(f, e))?;
                    (next, Some(PrimExpr::Val(PrimVal::Var(f))))
                } else {
                    (current_open, None)
                };
                let (next, post) = self.convert_expr(
                    &ast::Expr::AssignOp(
                        x,
                        if *is_inc {
                            ast::Binop::Plus
                        } else {
                            ast::Binop::Minus
                        },
                        &ast::Expr::ILit(1),
                    ),
                    next,
                )?;
                return Ok((next, if *is_post { pre.unwrap() } else { post }));
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
                    (from, None /* $0 */) => {
                        return self.convert_expr(
                            &ast::Expr::Getline {
                                from: from.clone(),
                                into: Some(&Unop(ast::Unop::Column, &ast::Expr::ILit(0))),
                            },
                            current_open,
                        )
                    }
                    (Some(from), Some(into)) => {
                        let (next, _) = self.convert_expr(
                            &ast::Expr::Assign(
                                into,
                                &ast::Expr::Call(Either::Right(Nextline), vec![from]),
                            ),
                            current_open,
                        )?;
                        return self.convert_expr(
                            &ast::Expr::Call(Either::Right(ReadErr), vec![from]),
                            next,
                        );
                    }
                    (None /*stdin*/, Some(into)) => {
                        let (next, _) = self.convert_expr(
                            &ast::Expr::Assign(
                                into,
                                &ast::Expr::Call(Either::Right(NextlineStdin), vec![]),
                            ),
                            current_open,
                        )?;
                        return self.convert_expr(
                            &ast::Expr::Call(Either::Right(ReadErrStdin), vec![]),
                            next,
                        );
                    }
                };
            }
        };
        Ok((current_open, res_expr))
    }

    fn guarded_else(&mut self, from: NodeIx, to: NodeIx) {
        if self.f.cfg.node_weight(from).unwrap().sealed {
            return;
        }
        self.f.cfg.add_edge(from, to, Transition::null());
    }

    fn do_assign<'c>(
        &mut self,
        v: &'c Expr<'c, 'b, I>,
        to: impl FnOnce(&PrimVal<'b>) -> PrimExpr<'b>,
        current_open: NodeIx,
    ) -> Result<(NodeIx, PrimExpr<'b>)> {
        use ast::Expr::*;
        match v {
            Var(i) => Ok((
                current_open,
                if let Ok(b) = builtins::Variable::try_from(i.clone()) {
                    let res = PrimExpr::LoadBuiltin(b);
                    let res_v = self.to_val(res.clone(), current_open)?;
                    self.add_stmt(current_open, PrimStmt::SetBuiltin(b, to(&res_v)))?;
                    res
                } else {
                    let ident = self.get_identifier(i);
                    let res_v = PrimVal::Var(ident);
                    self.add_stmt(current_open, PrimStmt::AsgnVar(ident, to(&res_v)))?;
                    PrimExpr::Val(res_v)
                },
            )),
            Unop(ast::Unop::Column, n) => {
                use {ast::Unop::*, builtins::Function};
                let (next, v) = self.convert_val(n, current_open)?;
                let res = PrimExpr::CallBuiltin(Function::Unop(Column), smallvec![v.clone()]);
                let res_v = self.to_val(res.clone(), next)?;
                let to_v = self.to_val(to(&res_v), next)?;
                self.add_stmt(
                    next,
                    PrimStmt::AsgnVar(
                        Self::unused(),
                        PrimExpr::CallBuiltin(Function::Setcol, smallvec![v, to_v]),
                    ),
                )?;
                Ok((next, res))
            }
            _ => err!("unsupprted assignment LHS: {:?}", v),
        }
    }
    fn do_assign_index<'c>(
        &mut self,
        arr: &'c Expr<'c, 'b, I>,
        ix: &'c Expr<'c, 'b, I>,
        mut to_f: impl FnMut(
            &mut Self,
            PrimVal<'b>,
            PrimVal<'b>,
            NodeIx,
        ) -> Result<(NodeIx, PrimExpr<'b>)>,
        current_open: NodeIx,
    ) -> Result<(NodeIx, PrimExpr<'b>)> {
        let (next, arr_e) = self.convert_expr(arr, current_open)?;

        // Only assign to a new variable if we need to.
        let arr_id = if let PrimExpr::Val(PrimVal::Var(id)) = arr_e {
            id
        } else {
            let arr_id = self.fresh_local();
            self.add_stmt(next, PrimStmt::AsgnVar(arr_id, arr_e))?;
            arr_id
        };

        let arr_v = PrimVal::Var(arr_id);
        let (next, ix_v) = self.convert_val(ix, next)?;
        let (next, to_e) = to_f(self, arr_v.clone(), ix_v.clone(), next)?;
        self.add_stmt(
            next,
            PrimStmt::AsgnIndex(arr_id, ix_v.clone(), to_e.clone()),
        )?;
        Ok((next, PrimExpr::Index(arr_v, ix_v)))
    }

    fn do_condition<'c>(
        &mut self,
        cond: &'c Expr<'c, 'b, I>,
        tcase: (NodeIx, NodeIx),
        fcase: Option<(NodeIx, NodeIx)>,
        current_open: NodeIx,
    ) -> Result<NodeIx> {
        let (t_start, t_end) = tcase;
        let (current_open, c_val) = if let ast::Expr::PatLit(_) = cond {
            // For conditionals, pattern literals become matches against $0.
            use ast::{Binop::*, Expr::*, Unop::*};
            self.convert_val(&Binop(Match, &Unop(Column, &ILit(0)), cond), current_open)?
        } else {
            self.convert_val(cond, current_open)?
        };
        let next = self.f.cfg.add_node(Default::default());

        // current_open => t_start if the condition holds
        self.f
            .cfg
            .add_edge(current_open, t_start, Transition::new(c_val));
        // continue to next after the true case is evaluated, unless we are returning.
        self.guarded_else(t_end, next);

        if let Some((f_start, f_end)) = fcase {
            // Set up the same connections as before, this time with a null edge rather
            // than c_val.
            self.f
                .cfg
                .add_edge(current_open, f_start, Transition::null());
            self.f.cfg.add_edge(f_end, next, Transition::null());
        } else {
            // otherwise continue directly from current_open.
            self.guarded_else(current_open, next);
        }
        Ok(next)
    }

    fn convert_val<'c>(
        &mut self,
        expr: &'c Expr<'c, 'b, I>,
        current_open: NodeIx,
    ) -> Result<(NodeIx, PrimVal<'b>)> {
        let (next_open, e) = self.convert_expr(expr, current_open)?;
        Ok((next_open, self.to_val(e, next_open)?))
    }

    fn make_loop<'c>(
        &mut self,
        body: &'c Stmt<'c, 'b, I>,
        update: Option<&'c Stmt<'c, 'b, I>>,
        current_open: NodeIx,
        is_do: bool,
    ) -> Result<(
        NodeIx, // header
        NodeIx, // body header
        NodeIx, // body footer
        NodeIx, // footer = next open
    )> {
        // Create header and footer nodes.
        let h = self.f.cfg.add_node(Default::default());
        let f = self.f.cfg.add_node(Default::default());
        self.f.loop_ctx.push((h, f));

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
            self.f
                .cfg
                .add_edge(current_open, b_start, Transition::null());
        } else {
            // Current => Header => Body => Footer
            //             ^         |
            //             ^---------
            self.f.cfg.add_edge(current_open, h, Transition::null());
        }
        self.f.cfg.add_edge(b_end, h, Transition::null());
        self.f.loop_ctx.pop().unwrap();
        Ok((h, b_start, b_end, f))
    }

    fn to_val(&mut self, exp: PrimExpr<'b>, current_open: NodeIx) -> Result<PrimVal<'b>> {
        Ok(if let PrimExpr::Val(v) = exp {
            v
        } else {
            let f = self.fresh_local();
            self.add_stmt(current_open, PrimStmt::AsgnVar(f, exp))?;
            PrimVal::Var(f)
        })
    }

    fn fresh(&mut self) -> Ident {
        self.ctx.fresh()
    }

    fn fresh_local(&mut self) -> Ident {
        self.ctx.fresh_local()
    }

    fn record_ident(&mut self, id: Ident, blk: NodeIx) {
        // TODO: add a new set to the global context indicating which functions write to `id`, if
        // it is a global. That will then be used to rewrite it as a local.
        self.f
            .defsites
            .entry(id)
            .or_insert(HashSet::default())
            .insert(blk);
        self.f
            .orig
            .entry(blk)
            .or_insert(HashSet::default())
            .insert(id);
    }

    fn get_identifier(&mut self, i: &I) -> Ident {
        // Look for any local variables with this name first, then search the global scope, then
        // create a fresh global variable.
        if let Some(ix) = self.f.args_map.get(i) {
            self.f.args[*ix as usize].id
        } else if let Some(id) = self.ctx.hm.get(i) {
            // We have found a global identifier that is not in main. Make sure it is not marked as
            // local.
            if id.global && self.f.name.is_some() {
                self.ctx.local_globals.remove(&id.low);
            }
            *id
        } else {
            let next = self.fresh();
            self.ctx.hm.insert(i.clone(), next);
            self.ctx.may_rename.push(next);
            if self.f.name.is_none() {
                self.ctx.local_globals.insert(next.low);
            }
            next
        }
    }

    fn add_stmt(&mut self, at: NodeIx, stmt: PrimStmt<'b>) -> Result<()> {
        if let PrimStmt::AsgnVar(ident, _) = stmt {
            self.record_ident(ident, at);
        }
        let bb = self.f.cfg.node_weight_mut(at).unwrap();
        if bb.sealed {
            return err!(
                "appending to sealed basic block ({}). Last instr={:?}",
                at.index(),
                bb.q.back().unwrap()
            );
        }
        bb.q.push_back(stmt);
        Ok(())
    }

    fn seal(&mut self, at: NodeIx) {
        self.f.cfg.node_weight_mut(at).unwrap().sealed = true;
    }

    fn insert_phis(&mut self) {
        use crate::common::WorkList;
        // TODO: do we need defsites and orig after this, or can we deallocate them here?

        // phis: the set of basic blocks that must have a phi node for a given variable.
        let mut phis = HashMap::<Ident, HashSet<NodeIx>>::new();
        let mut worklist = WorkList::default();
        // TODO rework this iteration to explicitly intersect may_rename and defsites? That way we
        // do O(min(|defsites|,|may_rename|)) work.
        for ident in self.ctx.may_rename.iter().cloned() {
            if ident.global && self.ctx.local_globals.get(&ident.low).is_none() {
                continue;
            }
            // Add all defsites into the worklist.
            let defsites = if let Some(ds) = self.f.defsites.get(&ident) {
                ds
            } else {
                continue;
            };
            worklist.extend(defsites.iter().map(|x| *x));
            while let Some(node) = worklist.pop() {
                // For all nodes on the dominance frontier without phi nodes for this identifier,
                // create a phi node of the appropriate size and insert it at the front of the
                // block (no renaming).
                for d in self.f.df[node.index()].iter() {
                    let d_ix = NodeIx::new(*d as usize);
                    if phis.get(&ident).map(|s| s.contains(&d_ix)) != Some(true) {
                        let phi = PrimExpr::Phi(
                            self.f
                                .cfg
                                .neighbors_directed(d_ix, Direction::Incoming)
                                .map(|n| (n, ident))
                                .collect(),
                        );
                        let stmt = PrimStmt::AsgnVar(ident, phi);
                        self.f
                            .cfg
                            .node_weight_mut(d_ix)
                            .expect("node in dominance frontier must be valid")
                            .q
                            .push_front(stmt);
                        phis.entry(ident).or_insert(HashSet::default()).insert(d_ix);
                        if !defsites.contains(&d_ix) {
                            worklist.insert(d_ix);
                        }
                    }
                }
            }
        }
    }

    fn rename(&mut self, cur: NodeIx) {
        // TODO mark elements of renamestack as ones that do not progress, thereby changing the
        // behavior or get_next and latest
        #[derive(Clone)]
        struct RenameStack {
            // global variables do not get renamed and have no phi functions.
            global: bool,
            count: NumTy,
            stack: SmallVec<NumTy>,
        }

        impl RenameStack {
            fn latest(&self) -> NumTy {
                if self.global {
                    0
                } else {
                    *self
                        .stack
                        .last()
                        .expect("variable stack should never be empty")
                }
            }
            fn get_next(&mut self) -> NumTy {
                if self.global {
                    0
                } else {
                    let next = self.count + 1;
                    self.count = next;
                    self.stack.push(next);
                    next
                }
            }
            fn pop(&mut self) {
                if !self.global {
                    self.stack.pop().unwrap();
                }
            }
        }

        fn rename_recursive<'b, I>(
            f: &mut Function<'b, I>,
            cur: NodeIx,
            state: &mut Vec<RenameStack>,
        ) {
            // We need to remember which new variables are introduced in this frame so we can
            // remove them when we are done.
            let mut defs = smallvec::SmallVec::<[NumTy; 16]>::new();

            // First, go through all the statements and update the variables to the highest
            // subscript (second component in Ident).
            for stmt in &mut f
                .cfg
                .node_weight_mut(cur)
                .expect("rename must be passed valid node indices")
                .q
            {
                // Note that `replace` is specialized to our use-case in this method. It does not hit
                // AsgnVar identifiers, and it skips Phi nodes.
                stmt.replace(|Ident { low, global, .. }| Ident {
                    low,
                    sub: state[low as usize].latest(),
                    global,
                });
                if let PrimStmt::AsgnVar(Ident { low, sub, .. }, _) = stmt {
                    *sub = state[*low as usize].get_next();
                    defs.push(*low);
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
            let mut walker = f.cfg.neighbors_directed(cur, Direction::Outgoing).detach();
            while let Some((edge, neigh)) = walker.next(&f.cfg) {
                if let Some(PrimVal::Var(Ident { low, sub, .. })) =
                    &mut f.cfg.edge_weight_mut(edge).unwrap().0
                {
                    *sub = state[*low as usize].latest();
                }
                for stmt in &mut f.cfg.node_weight_mut(neigh).unwrap().q {
                    if let PrimStmt::AsgnVar(_, PrimExpr::Phi(ps)) = stmt {
                        for (pred, Ident { low, sub, .. }) in ps.iter_mut() {
                            if pred == &cur {
                                *sub = state[*low as usize].latest();
                                break;
                            }
                        }
                    }
                }
            }
            for child in f.dt[cur.index()].clone().iter() {
                rename_recursive(f, NodeIx::new(*child as usize), state);
            }
            for d in defs.into_iter() {
                state[d as usize].pop();
            }
        }
        let mut state = vec![
            RenameStack {
                global: false,
                count: 0,
                stack: smallvec![0],
            };
            self.ctx.max as usize
        ];
        for id in self.ctx.hm.values() {
            if id.global && self.ctx.local_globals.get(&id.low).is_none() {
                state[id.low as usize].global = true;
            }
        }
        rename_recursive(self.f, cur, &mut state);
    }
}
