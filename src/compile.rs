use crate::builtins;
use crate::bytecode;
use crate::cfg::{self, is_unused, Function, Ident, PrimExpr, PrimStmt, PrimVal, ProgramContext};
use crate::codegen;
use crate::common::{CompileError, Either, Graph, NodeIx, NumTy, Result, Stage, WorkList};
use crate::cross_stage;
use crate::input_taint::TaintedStringAnalysis;
#[cfg(feature = "llvm_backend")]
use crate::llvm;
use crate::pushdown::{FieldSet, UsedFieldAnalysis};
use crate::runtime::{self, Str};
use crate::smallvec::{self, smallvec};
use crate::string_constants::{self, StringConstantAnalysis};
use crate::types;

use hashbrown::{hash_map::Entry, HashMap, HashSet};
use regex::bytes::Regex;

use std::collections::VecDeque;
use std::mem;
use std::sync::Arc;

pub(crate) const UNUSED: u32 = u32::max_value();
pub(crate) const NULL_REG: u32 = UNUSED - 1;

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub(crate) enum Ty {
    Int = 0,
    Float = 1,
    Str = 2,
    MapIntInt = 3,
    MapIntFloat = 4,
    MapIntStr = 5,
    MapStrInt = 6,
    MapStrFloat = 7,
    MapStrStr = 8,
    IterInt = 9,
    IterStr = 10,
    Null = 11,
}

pub(crate) const NUM_TYPES: usize = Ty::Null as usize + 1;

impl std::convert::TryFrom<u32> for Ty {
    type Error = ();
    fn try_from(u: u32) -> std::result::Result<Ty, ()> {
        use Ty::*;
        Ok(match u {
            0 => Int,
            1 => Float,
            2 => Str,
            3 => MapIntInt,
            4 => MapIntFloat,
            5 => MapIntStr,
            6 => MapStrInt,
            7 => MapStrFloat,
            8 => MapStrStr,
            9 => IterInt,
            10 => IterStr,
            11 => Null,
            _ => return Err(()),
        })
    }
}

impl Default for Ty {
    fn default() -> Ty {
        Ty::Null
    }
}

impl Ty {
    fn is_iter(self) -> bool {
        if let Ty::IterInt | Ty::IterStr = self {
            true
        } else {
            false
        }
    }

    pub(crate) fn key_iter(self) -> Result<Ty> {
        use Ty::*;
        match self {
            MapIntInt | MapIntFloat | MapIntStr => Ok(IterInt),
            MapStrInt | MapStrFloat | MapStrStr => Ok(IterStr),
            Null | Int | Float | Str | IterInt | IterStr => {
                err!("attempt to get iterator from non-map type: {:?}", self)
            }
        }
    }

    pub(crate) fn iter(self) -> Result<Ty> {
        use Ty::*;
        match self {
            IterInt => Ok(Int),
            IterStr => Ok(Str),
            Null | Int | Float | Str | MapIntInt | MapIntFloat | MapIntStr | MapStrInt
            | MapStrFloat | MapStrStr => {
                err!("attempt to get element of non-iterator type: {:?}", self)
            }
        }
    }

    pub(crate) fn is_array(self) -> bool {
        use Ty::*;
        match self {
            MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat | MapStrStr => true,
            Null | Int | Float | Str | IterInt | IterStr => false,
        }
    }

    pub(crate) fn key(self) -> Result<Ty> {
        use Ty::*;
        match self {
            MapIntInt | MapIntFloat | MapIntStr => Ok(Int),
            MapStrInt | MapStrFloat | MapStrStr => Ok(Str),
            Null | Int | Float | Str | IterInt | IterStr => {
                err!("attempt to get key of non-map type: {:?}", self)
            }
        }
    }

    pub(crate) fn val(self) -> Result<Ty> {
        use Ty::*;
        match self {
            MapStrInt | MapIntInt => Ok(Int),
            MapStrFloat | MapIntFloat => Ok(Float),
            MapStrStr | MapIntStr => Ok(Str),
            Null | Int | Float | Str | IterInt | IterStr => {
                err!("attempt to get val of non-map type: {:?}", self)
            }
        }
    }
}

fn visit_used_fields<'a>(stmt: &Instr<'a>, cur_func_id: NumTy, ufa: &mut UsedFieldAnalysis) {
    match stmt {
        Either::Left(l) => ufa.visit_ll(l),
        Either::Right(r) => ufa.visit_hl(cur_func_id, r),
    }
}

fn visit_taint_analysis<'a>(stmt: &Instr<'a>, func_id: NumTy, tsa: &mut TaintedStringAnalysis) {
    match stmt {
        Either::Left(ll) => tsa.visit_ll(ll),
        Either::Right(hl) => tsa.visit_hl(func_id, hl),
    }
}

fn visit_string_constant_analysis<'a>(
    stmt: &Instr<'a>,
    func_id: NumTy,
    sca: &mut StringConstantAnalysis<'a>,
) {
    match stmt {
        Either::Left(ll) => sca.visit_ll(ll),
        Either::Right(hl) => sca.visit_hl(func_id, hl),
    }
}

pub(crate) fn bytecode<'a, LR: runtime::LineReader>(
    ctx: &mut cfg::ProgramContext<'a, &'a str>,
    reader: LR,
    ff: impl runtime::writers::FileFactory,
    num_workers: usize,
) -> Result<bytecode::Interp<'a, LR>> {
    Typer::init_from_ctx(ctx)?.to_interp(reader, ff, num_workers)
}

#[cfg(test)]
pub(crate) fn context_compiles<'a>(ctx: &mut cfg::ProgramContext<'a, &'a str>) -> Result<()> {
    Typer::init_from_ctx(ctx)?;
    Ok(())
}

#[cfg(test)]
pub(crate) fn used_fields<'a>(ctx: &mut cfg::ProgramContext<'a, &'a str>) -> Result<FieldSet> {
    Ok(Typer::init_from_ctx(ctx)?.used_fields)
}

#[cfg(feature = "llvm_backend")]
pub(crate) fn dump_llvm<'a>(
    ctx: &mut cfg::ProgramContext<'a, &'a str>,
    cfg: llvm::Config,
) -> Result<String> {
    use llvm::Generator;
    let mut typer = Typer::init_from_ctx(ctx)?;
    unsafe {
        let mut gen = Generator::init(&mut typer, cfg)?;
        gen.dump_module()
    }
}

#[cfg(all(test, feature = "llvm_backend", feature = "unstable"))]
pub(crate) fn compile_llvm<'a>(
    ctx: &mut cfg::ProgramContext<'a, &'a str>,
    cfg: llvm::Config,
) -> Result<()> {
    use llvm::Generator;
    let mut typer = Typer::init_from_ctx(ctx)?;
    unsafe {
        let mut gen = Generator::init(&mut typer, cfg)?;
        gen.compile_main()
    }
}

#[cfg(feature = "llvm_backend")]
pub(crate) fn run_llvm<'a>(
    ctx: &mut cfg::ProgramContext<'a, &'a str>,
    reader: impl codegen::intrinsics::IntoRuntime,
    ff: impl runtime::writers::FileFactory,
    cfg: llvm::Config,
) -> Result<()> {
    use crate::llvm::Generator;
    let mut typer = Typer::init_from_ctx(ctx)?;
    let used_fields = typer.used_fields.clone();
    let named_cols = typer.named_columns.take();
    unsafe {
        let gen = Generator::init(&mut typer, cfg)?;
        codegen::run_main(gen, reader, ff, &used_fields, named_cols, cfg.num_workers)
    }
}

pub(crate) fn run_cranelift<'a>(
    ctx: &mut cfg::ProgramContext<'a, &'a str>,
    reader: impl codegen::intrinsics::IntoRuntime,
    ff: impl runtime::writers::FileFactory,
    cfg: codegen::Config,
) -> Result<()> {
    use codegen::clif::Generator;
    let mut typer = Typer::init_from_ctx(ctx)?;
    let used_fields = typer.used_fields.clone();
    let named_cols = typer.named_columns.take();
    unsafe {
        let gen = Generator::init(&mut typer, cfg)?;
        codegen::run_main(gen, reader, ff, &used_fields, named_cols, cfg.num_workers)
    }
}

type SmallVec<T> = smallvec::SmallVec<[T; 2]>;

#[derive(Debug)]
pub(crate) enum HighLevel {
    // TODO we may not strictly need Call's dst_ty and Ret's Ty field. Other information may have
    // it available.
    Call {
        func_id: NumTy, /* monomorphized function id */
        dst_reg: NumTy,
        dst_ty: Ty,
        args: SmallVec<(NumTy, Ty)>,
    },
    Ret(NumTy, Ty),
    Phi(NumTy, Ty, SmallVec<(NodeIx /*pred*/, NumTy /*register*/)>),
    DropIter(NumTy, Ty),
}

#[derive(Default)]
struct Registers {
    stats: RegStatuses,
    globals: HashMap<Ident, (u32, Ty)>,
}

#[derive(Debug, Copy, Clone)]
enum RegStatus {
    Local,
    Global,
    Ret,
}

#[derive(Default, Debug)]
struct RegStatuses([Vec<RegStatus>; NUM_TYPES]);

impl RegStatuses {
    fn reg_of_ty(&mut self, ty: Ty) -> NumTy {
        self.new_reg(ty, RegStatus::Local)
    }
    fn new_reg(&mut self, ty: Ty, status: RegStatus) -> NumTy {
        if let Ty::Null = ty {
            return NULL_REG;
        }
        let v = &mut self.0[ty as usize];
        let res = v.len();
        v.push(status);

        res as NumTy
    }

    fn count(&self, ty: Ty) -> NumTy {
        self.0[ty as usize].len() as NumTy
    }

    fn get_status(&self, reg: NumTy, ty: Ty) -> RegStatus {
        if ty == Ty::Null {
            return RegStatus::Local;
        }
        self.0[ty as usize][reg as usize]
    }
}

#[derive(Default)]
pub(crate) struct Node<'a> {
    pub insts: VecDeque<Instr<'a>>,
    pub exit: bool,
}

pub(crate) type LL<'a> = bytecode::Instr<'a>;
type Instr<'a> = Either<LL<'a>, HighLevel>;
type CFG<'a> = Graph<Node<'a>, Option<NumTy /* Int register */>>;
type CallGraph = Graph<HashSet<(NumTy, Ty)>, ()>;

// Typer contains much of the state necessary for generating a typed CFG, which in turn can
// generate bytecode or LLVM.
#[derive(Default)]
pub(crate) struct Typer<'a> {
    regs: Registers,
    id_map: HashMap<
        // TODO: make newtypes for these different Ids?
        (
            NumTy,        /* cfg-level func id */
            SmallVec<Ty>, /* arg types */
        ),
        NumTy, /* bytecode-level func id */
    >,
    arity: HashMap<NumTy /* cfg-level func id */, NumTy>,
    local_globals: HashSet<NumTy>,
    // Why not just store FuncInfo's fields in a Frame?
    // We access Frames one at a time (through a View); but we need access to function arity and
    // return types across invidual views. We expose these fields in a separate type immutably to
    // facilitate that.
    //
    // Another option would be to pass a mutable reference to `frames` for all of bytecode-building
    // functions below, but then each access to frame would have the form of
    // self.frames[current_index], which is marginally less efficient and (more importantly)
    // error-prone.
    pub func_info: Vec<FuncInfo>,
    pub frames: Vec<Frame<'a>>,
    pub main_offset: Stage<usize>,

    // For projection pushdown
    used_fields: FieldSet,
    // The fields referenced by name via the FI builtin variable
    named_columns: Option<Vec<&'a [u8]>>,
    // For rejecting suspcicious programs with commands.
    taint_analysis: Option<TaintedStringAnalysis>,
    // For analysis passes that introspect into the set of constant string values that will
    // dynamically be assigned to a register
    string_constants: Option<StringConstantAnalysis<'a>>,
    // Not used for bytecode generation.
    callgraph: Graph<HashSet<(NumTy, Ty)>, ()>,

    // The global variables referenced (transitively) by each function. This is used both for
    // cross-stage state propagation for parallel execution, as well as for implementing global
    // variables in the LLVM backend. It is computed lazily because these are not needed for
    // serial, bytecode-only scripts.
    global_refs: Option<Vec<HashSet<(NumTy, Ty)>>>,
}

#[derive(Default)]
struct SlotCounter {
    slots: HashMap<(NumTy, Ty), usize>,
    counter: HashMap<Ty, usize>,
}

impl SlotCounter {
    fn get_slot(&mut self, reg: (NumTy, Ty)) -> usize {
        if let Some(c) = self.slots.get(&reg) {
            return *c;
        }
        let ctr = self.counter.entry(reg.1).or_insert(0);
        let res = *ctr;
        *ctr += 1;
        self.slots.insert(reg, res);
        res
    }
}

#[derive(Debug)]
pub(crate) struct FuncInfo {
    pub ret_ty: Ty,
    // For bytecode, we pop into each of these registers at the specified type.
    pub arg_tys: SmallVec<Ty>,
}

#[derive(Default)]
pub(crate) struct Frame<'a> {
    src_function: NumTy,
    cur_ident: NumTy,
    entry: NodeIx,
    exit: NodeIx,
    pub locals: HashMap<Ident, (u32, Ty)>,
    pub arg_regs: SmallVec<NumTy>,
    pub cfg: CFG<'a>,
    pub is_called: bool,
}

impl<'a> Frame<'a> {
    fn load_slots(
        &mut self,
        regs: impl Iterator<Item = (NumTy, Ty)>,
        ctr: &mut SlotCounter,
    ) -> Result<()> {
        let stream = self.cfg.node_weight_mut(self.entry).unwrap();
        for reg in regs {
            let slot = ctr.get_slot(reg);
            if let Some(inst) = cross_stage::load_slot_instr(reg.0, reg.1, slot)? {
                stream.insts.push_front(Either::Left(inst))
            }
        }
        Ok(())
    }
    fn store_slots(
        &mut self,
        regs: impl Iterator<Item = (NumTy, Ty)>,
        ctr: &mut SlotCounter,
    ) -> Result<()> {
        let stream = self.cfg.node_weight_mut(self.exit).unwrap();
        for reg in regs {
            let slot = ctr.get_slot(reg);
            if let Some(inst) = cross_stage::store_slot_instr(reg.0, reg.1, slot)? {
                stream.insts.push_front(Either::Left(inst))
            }
        }
        Ok(())
    }
}

struct View<'a, 'b> {
    frame: &'b mut Frame<'a>,
    regs: &'b mut Registers,
    cg: &'b mut CallGraph,
    id_map: &'b HashMap<(NumTy, SmallVec<Ty>), NumTy>,
    local_globals: &'b HashSet<NumTy>,
    arity: &'b HashMap<NumTy, NumTy>,
    func_info: &'b Vec<FuncInfo>,
    // The current basic block being filled; It'll be swaped into `frame.cfg` as we translate a
    // given function cfg.
    stream: &'b mut Node<'a>,
}

fn pop_var<'a>(instrs: &mut Vec<LL<'a>>, reg: NumTy, ty: Ty) -> Result<()> {
    use Ty::*;
    instrs.push(match ty {
        Null => return Ok(()),
        IterInt | IterStr => return err!("invalid argument type: {:?}", ty),
        Int | Float | Str | MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat
        | MapStrStr => LL::Pop(ty, reg),
    });
    Ok(())
}

fn push_var<'a>(instrs: &mut Vec<LL<'a>>, reg: NumTy, ty: Ty) -> Result<()> {
    use Ty::*;
    match ty {
        Null => Ok(()),
        Int | Float | Str | MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat
        | MapStrStr => {
            instrs.push(LL::Push(ty, reg));
            Ok(())
        }
        IterInt | IterStr => err!("invalid argument type: {:?}", ty),
    }
}

fn alloc_local<'a>(dst_reg: NumTy, dst_ty: Ty) -> Option<LL<'a>> {
    use Ty::*;
    match dst_ty {
        MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat | MapStrStr => {
            Some(LL::AllocMap(dst_ty, dst_reg))
        }
        _ => None,
    }
}

fn mov<'a>(dst_reg: u32, src_reg: u32, ty: Ty) -> Result<Option<LL<'a>>> {
    use Ty::*;
    if dst_reg == UNUSED || src_reg == UNUSED {
        return Ok(None);
    }
    let res = match ty {
        Null => return Ok(None),
        IterInt | IterStr => return err!("attempt to move values of type {:?}", ty),
        Int | Float | Str | MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat
        | MapStrStr => LL::Mov(ty, dst_reg, src_reg),
    };

    Ok(Some(res))
}

fn accum<'a>(inst: &Instr<'a>, mut f: impl FnMut(NumTy, Ty)) {
    use {Either::*, HighLevel::*};
    match inst {
        Left(ll) => ll.accum(f),
        Right(Call {
            dst_reg,
            dst_ty,
            args,
            ..
        }) => {
            f(*dst_reg, *dst_ty);
            for (reg, ty) in args.iter().cloned() {
                f(reg, ty)
            }
        }
        Right(Ret(reg, ty)) | Right(Phi(reg, ty, _)) | Right(DropIter(reg, ty)) => f(*reg, *ty),
    }
}

impl<'a> Typer<'a> {
    pub fn stage(&self) -> Stage<usize> {
        self.main_offset.clone()
    }

    fn to_interp<LR: runtime::LineReader>(
        &mut self,
        reader: LR,
        ff: impl runtime::writers::FileFactory,
        num_workers: usize,
    ) -> Result<bytecode::Interp<'a, LR>> {
        let instrs = self.to_bytecode()?;
        let cols = self.named_columns.take();
        Ok(bytecode::Interp::new(
            instrs,
            self.stage(),
            num_workers,
            |ty| self.regs.stats.count(ty) as usize,
            reader,
            ff,
            &self.used_fields,
            cols,
        ))
    }

    // At initialization time, we generate Either<LL, HL>, this function lowers the HL into LL.
    fn to_bytecode(&mut self) -> Result<Vec<Vec<LL<'a>>>> {
        let mut res = vec![vec![]; self.frames.len()];
        let ret_regs: Vec<_> = (0..self.frames.len())
            .map(|i| {
                let ret_ty = self.func_info[i].ret_ty;
                self.regs.stats.new_reg(ret_ty, RegStatus::Ret)
            })
            .collect();
        let mut bb_map: Vec<usize> = Vec::new();
        let mut jmps: Vec<usize> = Vec::new();
        // If we wanted to, we could colocate locals and args, but absent a serious performance
        // issue this seems cleaner.
        let mut args: Vec<(NumTy, Ty)> = Vec::new();
        let mut locals: Vec<(NumTy, Ty)> = Vec::new();
        for (i, frame) in self.frames.iter().enumerate() {
            if !frame.is_called {
                continue;
            }
            let instrs = &mut res[i];
            bb_map.clear();
            bb_map.reserve(frame.cfg.node_count());
            jmps.clear();

            // Start by popping any args off of the stack.
            args.extend(
                frame
                    .arg_regs
                    .iter()
                    .cloned()
                    .zip(self.func_info[i].arg_tys.iter().cloned()),
            );
            args.reverse();

            // Some local variables (maps, at time of writing) must be explicitly reallocated to
            // handle the case where no value is passed as an argument. We do this before popping
            // variables to ensure arguments are propagated if they are passed.
            //
            // This system currently is not shared with the LLVM backend, as both strings and maps
            // have to be allocated there. It is possible that the two codepaths could be merged at
            // some point.
            for instr in frame
                .locals
                .values()
                .cloned()
                .flat_map(|(reg, ty)| alloc_local(reg, ty).into_iter())
            {
                instrs.push(instr);
            }
            for (a_reg, a_ty) in args.drain(..) {
                pop_var(instrs, a_reg, a_ty)?;
            }

            for (j, n) in frame.cfg.raw_nodes().iter().enumerate() {
                bb_map.push(instrs.len());
                use HighLevel::*;
                for stmt in &n.weight.insts {
                    match stmt {
                        Either::Left(ll) => instrs.push(ll.clone()),
                        Either::Right(Call {
                            func_id,
                            dst_reg,
                            dst_ty,
                            args,
                        }) => {
                            // args have already been normalized, and return type already matches.
                            // All we need to do is push local variables (to avoid clobbers) and
                            // push args onto the stack.
                            // NB locals does not contain all of the local registers, though the
                            // ones it does not cover are "transient" in that we have no way to get
                            // a handle on them outside of the immediate context in which they are
                            // constructed (e.g. through reg_of_ty). I believe that means we can
                            // rule them out as being needed across callsites.
                            //
                            // Today, function calls are a bit slow because of all these pushes (we
                            // have a lot of local variables because we do not reuse registers). We
                            // may want to optimize this by looking only over variables referenced
                            // in reachable BBs from the current one.
                            locals.clear();
                            locals.extend(frame.locals.values().cloned());
                            for (reg, ty) in locals.iter().cloned() {
                                if !ty.is_iter() {
                                    push_var(instrs, reg, ty)?;
                                }
                            }
                            for (reg, ty) in args.iter().cloned() {
                                assert!(!ty.is_iter());
                                push_var(instrs, reg, ty)?;
                            }
                            let callee = *func_id as usize;
                            instrs.push(LL::Call(callee));

                            // Restore local variables
                            locals.reverse();
                            for (reg, ty) in locals.iter().cloned() {
                                if !ty.is_iter() {
                                    pop_var(instrs, reg, ty)?;
                                }
                            }
                            let ret_reg = ret_regs[callee];
                            debug_assert_eq!(self.func_info[callee].ret_ty, *dst_ty);
                            if let Some(inst) = mov(*dst_reg, ret_reg, *dst_ty)? {
                                instrs.push(inst);
                            }
                        }
                        Either::Right(Ret(reg, ty)) => {
                            debug_assert_eq!(self.func_info[i].ret_ty, *ty);
                            if let Some(inst) = mov(ret_regs[i], *reg, *ty)? {
                                instrs.push(inst);
                            }
                            instrs.push(LL::Ret);
                        }
                        // handles by the predecessor.
                        Either::Right(Phi(_, _, _)) => {}
                        // we do not explicitly drop iterators in the bytecode interpreter.
                        Either::Right(DropIter(_, _)) => {}
                    }
                }

                let ix = NodeIx::new(j);
                // Now handle phi nodes
                for neigh in frame.cfg.neighbors(ix) {
                    for stmt in &frame.cfg.node_weight(neigh).unwrap().insts {
                        if let Either::Right(Phi(reg, ty, preds)) = stmt {
                            for (pred, src_reg) in preds.iter() {
                                if pred == &ix {
                                    if let Some(inst) = mov(*reg, *src_reg, *ty)? {
                                        instrs.push(inst);
                                    }
                                    break;
                                }
                            }
                        } else {
                            // Phis are all at the top;
                            break;
                        }
                    }
                }
                // And then jumps
                let mut walker = frame.cfg.neighbors(ix).detach();
                let mut edges = SmallVec::new();
                while let Some(eix) = walker.next_edge(&frame.cfg) {
                    edges.push(eix)
                }
                edges.reverse();
                for eix in edges.iter().cloned() {
                    let dst = frame.cfg.edge_endpoints(eix).unwrap().1.index();
                    if let Some(reg) = frame.cfg.edge_weight(eix).unwrap().clone() {
                        jmps.push(instrs.len());
                        instrs.push(LL::JmpIf(reg.into(), dst.into()));
                    } else if dst != j + 1 {
                        jmps.push(instrs.len());
                        instrs.push(LL::Jmp(dst.into()));
                    }
                }
            }
            // Now rewrite jumps
            for j in jmps.iter() {
                match &mut instrs[*j] {
                    LL::Jmp(bb) | LL::JmpIf(_, bb) => *bb = bb_map[bb.0].into(),
                    _ => unreachable!(),
                }
            }
        }
        Ok(res)
    }

    fn init_from_ctx(pc: &mut ProgramContext<'a, &'a str>) -> Result<Typer<'a>> {
        // Type-check the code, then initialize a Typer, assigning registers to local
        // and global variables.

        let mut gen = Typer::default();
        if !pc.allow_arbitrary_commands {
            gen.taint_analysis = Some(Default::default());
        }
        if pc.fold_regex_constants || pc.parse_header {
            gen.string_constants = Some(StringConstantAnalysis::from_config(
                string_constants::Config {
                    query_regex: pc.fold_regex_constants,
                    fi_refs: pc.parse_header,
                },
            ));
        }
        let types::TypeInfo { var_tys, func_tys } = types::get_types(pc)?;
        let local_globals = pc.local_globals();
        macro_rules! init_entry {
            ($v:expr, $func_id:expr, $args:expr) => {
                // If this returns None, it seems to mean that the function is never called.
                if let Some(ret_ty) = func_tys.get(&($func_id, $args.clone())).cloned() {
                    let res = gen.frames.len() as NumTy;
                    $v.insert(res);
                    let mut f = Frame::default();
                    f.src_function = $func_id;
                    f.cur_ident = res;
                    gen.frames.push(f);
                    gen.callgraph.add_node(Default::default());
                    gen.func_info.push(FuncInfo {
                        ret_ty,
                        arg_tys: $args.clone(),
                    });
                }
            };
        }
        for (func_id, func) in pc.funcs.iter().enumerate() {
            let arity = func.args.len() as NumTy;
            gen.arity.insert(func_id as NumTy, arity);
            if arity == 0 {
                let args: SmallVec<_> = Default::default();
                if let Entry::Vacant(v) = gen.id_map.entry((func_id as u32, args.clone())) {
                    init_entry!(v, func_id as u32, args);
                }
            }
        }
        for ((id, func_id, args), ty) in var_tys.iter() {
            let map = if id.is_global(&local_globals) {
                &mut gen.regs.globals
            } else {
                if let Entry::Vacant(v) = gen.id_map.entry((*func_id, args.clone())) {
                    init_entry!(v, *func_id, args);
                }
                &mut gen.frames[gen.id_map[&(*func_id, args.clone())] as usize].locals
            };
            let reg = gen.regs.stats.new_reg(
                *ty,
                if id.is_global(&local_globals) {
                    RegStatus::Global
                } else {
                    RegStatus::Local
                },
            );
            if let Some(old) = map.insert(*id, (reg, *ty)) {
                return err!(
                    "internal error: duplicate entries for same local in types  at id={:?}; {:?} vs {:?}",
                    id,
                    old,
                    (reg, *ty)
                );
            }
        }
        gen.main_offset = pc
            .main_stage()
            .map_ref(|o| gen.id_map[&(*o as NumTy, Default::default())] as usize);
        gen.local_globals = local_globals;
        for frame in gen.frames.iter_mut() {
            let src_func = frame.src_function as usize;
            let mut stream = Default::default();
            View {
                frame,
                regs: &mut gen.regs,
                cg: &mut gen.callgraph,
                id_map: &gen.id_map,
                arity: &gen.arity,
                local_globals: &gen.local_globals,
                func_info: &gen.func_info,
                stream: &mut stream,
            }
            .process_function(&pc.funcs[src_func])?;
        }
        // TODO: mark used frames first and then exclude them from the analyses?
        gen.run_analyses()?;
        gen.mark_used_frames();
        gen.add_slots()?;
        Ok(gen)
    }

    fn run_analyses(&mut self) -> Result<()> {
        let mut ufa = UsedFieldAnalysis::default();
        let mut refs = SmallVec::new();
        for (fix, frame) in self.frames.iter().enumerate() {
            for (bbix, bb) in frame.cfg.raw_nodes().iter().enumerate() {
                for (stmtix, stmt) in bb.weight.insts.iter().enumerate() {
                    // not tracking function calls
                    visit_used_fields(stmt, frame.cur_ident, &mut ufa);
                    if let Some(tsa) = &mut self.taint_analysis {
                        visit_taint_analysis(stmt, frame.cur_ident, tsa)
                    }
                    if let Some(sca) = &mut self.string_constants {
                        if sca.cfg().query_regex {
                            if let Either::Left(LL::IsMatch(_, _, pat))
                            | Either::Left(LL::Match(_, _, pat)) = stmt
                            {
                                refs.push((fix, bbix, stmtix, *pat));
                            }
                        }
                        visit_string_constant_analysis(stmt, frame.cur_ident, sca)
                    }
                }
            }
        }
        self.used_fields = ufa.solve();
        if let Some(tsa) = &mut self.taint_analysis {
            if !tsa.ok() {
                return err!("command potentially containing interpolated user input detected.\nIf this is a false positive, you can pass the -A flag to bypass this check.");
            }
        }
        if let Some(sca) = &mut self.string_constants {
            let mut strs = Vec::new();
            if sca.cfg().query_regex {
                // Fold any regex pattern constants that we see
                for (frame, bb, stmt, reg) in refs.into_iter() {
                    strs.clear();
                    sca.possible_strings(&reg, &mut strs);
                    if strs.len() != 1 {
                        continue;
                    }
                    let text = std::str::from_utf8(&strs[0]).map_err(|e| {
                        CompileError(format!("regex patterns must be valid UTF-8: {}", e))
                    })?;
                    let re = Arc::new(Regex::new(text).map_err(|err| {
                        CompileError(format!("regex parse error during compilation: {}", err))
                    })?);
                    let inst = self.frames[frame]
                        .cfg
                        .node_weight_mut(NodeIx::new(bb))
                        .unwrap()
                        .insts
                        .get_mut(stmt)
                        .unwrap();
                    let new_inst: Instr = match inst {
                        Either::Left(LL::IsMatch(dst, s, _)) => {
                            Either::Left(LL::IsMatchConst(*dst, *s, re))
                        }
                        Either::Left(LL::Match(dst, s, _)) => {
                            Either::Left(LL::MatchConst(*dst, *s, re))
                        }
                        _ => {
                            return err!(
                                "unexpected instruction during regex constant folding: {:?}",
                                inst
                            )
                        }
                    };
                    *inst = new_inst;
                }
            }
            if sca.cfg().fi_refs {
                strs.clear();
                if sca.fi_info(&mut strs) {
                    self.named_columns = Some(strs);
                }
            }
        }
        Ok(())
    }

    fn mark_used_frames(&mut self) {
        use petgraph::visit::Dfs;
        for offset in self.main_offset.iter() {
            let mut dfs = Dfs::new(&self.callgraph, NodeIx::new(*offset));
            while let Some(ix) = dfs.next(&self.callgraph) {
                self.frames[ix.index()].is_called = true;
            }
        }
    }

    fn add_slots(&mut self) -> Result<()> {
        use cross_stage::compute_slots;
        let (begin, main_loop, end) = match self.main_offset {
            Stage::Main(_) => return Ok(()),
            Stage::Par {
                begin,
                main_loop,
                end,
            } => (begin, main_loop, end),
        };
        let global_refs = self.get_global_refs();
        let slots = compute_slots(&begin, &main_loop, &end, global_refs);
        let mut ctr = SlotCounter::default();

        // Begin stores the context of begin_stores
        if let Some(off) = begin {
            self.frames[off].store_slots(slots.begin_stores.iter().cloned(), &mut ctr)?;
        }
        if let Some(off) = main_loop {
            self.frames[off].load_slots(slots.begin_stores.iter().cloned(), &mut ctr)?;
            self.frames[off].store_slots(slots.loop_stores.iter().cloned(), &mut ctr)?;
        }
        if let Some(off) = end {
            self.frames[off].load_slots(slots.loop_stores.iter().cloned(), &mut ctr)?;
        }

        Ok(())
    }

    pub(crate) fn get_global_refs(&mut self) -> Vec<HashSet<(NumTy, Ty)>> {
        if let Some(globals) = &self.global_refs {
            return globals.clone();
        }
        let mut globals = vec![HashSet::new(); self.frames.len()];
        // First, accumulate all the local and global registers referenced in all the functions.
        // We need these for LLVM because relevant globals are passed as function parameters, and
        // locals need to be allocated explicitly at the top of each function.
        for (i, frame) in self.frames.iter().enumerate() {
            // Manually borrow fields so that  we do not mutably borrow all of `self` in the
            // closure.
            let stats = &self.regs.stats;
            let cg = &mut self.callgraph;
            for bb in frame.cfg.raw_nodes() {
                for stmt in &bb.weight.insts {
                    accum(stmt, |reg, ty| {
                        if reg == UNUSED {
                            return;
                        }
                        match stats.get_status(reg, ty) {
                            RegStatus::Global => {
                                cg.node_weight_mut(NodeIx::new(i))
                                    .unwrap()
                                    .insert((reg, ty));
                            }
                            RegStatus::Ret | RegStatus::Local => {}
                        }
                    });
                }
            }
        }

        // We use a simple iterative fixed-point algorithm for computing which globals are
        // referenced by a given function. The globals we have found so far only list the globals
        // directly referenced by a function, but we need the ones referenced transitively by all
        // functions that a given function calls.
        //
        // TODO I think the traditional technique here is to use a bit set rather than a hash set.
        // That's probably the right choice here, because there wont be that many globals and
        // global references aren't likely to be sparse (which is the case where hash sets win).
        //
        // If this ever becomes a problem, that's the obvious optimization to make. Unions for
        // bitsets should be a good deal faster than for hash sets so long as the sets are
        // sufficiently dense.
        let mut wl = WorkList::default();
        wl.extend(0..self.frames.len());
        while let Some(frame) = wl.pop() {
            // All callees of a function inherit its globals.
            use petgraph::Direction;
            let frame_ix = NodeIx::new(frame);
            let mut walker = self
                .callgraph
                .neighbors_directed(frame_ix, Direction::Incoming)
                .detach();
            while let Some(callee) = walker.next_node(&self.callgraph) {
                if callee == frame_ix {
                    continue;
                }
                let (cur_globals, callee_globals) =
                    self.callgraph.index_twice_mut(frame_ix, callee);
                let mut added = false;
                for g in cur_globals.iter().cloned() {
                    added = callee_globals.insert(g) || added;
                }
                if added {
                    wl.insert(callee.index());
                }
            }
        }
        for (i, set) in globals.iter_mut().enumerate() {
            mem::swap(set, self.callgraph.node_weight_mut(NodeIx::new(i)).unwrap());
        }
        self.global_refs = Some(globals.clone());
        globals
    }
}

impl<'a, 'b> View<'a, 'b> {
    fn process_function(&mut self, func: &Function<'a, &'a str>) -> Result<()> {
        self.frame.entry = func.entry;
        self.frame.exit = func.exit;
        // Record registers for arguments.
        for arg in func.args.iter() {
            let (reg, _) = self.reg_of_ident(&arg.id);
            self.frame.arg_regs.push(reg);
        }
        // Allocate basic blocks in CFG.
        for _ in 0..func.cfg.node_count() {
            self.frame.cfg.add_node(Default::default());
        }
        // Fill them in.
        for (i, n) in func.cfg.raw_nodes().iter().enumerate() {
            for stmt in n.weight.q.iter() {
                self.stmt(stmt)?;
            }
            let ix = NodeIx::new(i);
            let mut branches: SmallVec<petgraph::graph::EdgeIndex> = Default::default();
            let mut walker = func.cfg.neighbors(ix).detach();
            while let Some(e) = walker.next_edge(&func.cfg) {
                branches.push(e);
            }
            // We get branches back in reverse order.
            branches.reverse();
            for eix in branches.iter().cloned() {
                let transition = func.cfg.edge_weight(eix).unwrap();
                let edge = match &transition.0 {
                    Some(val) => {
                        let (mut reg, ty) = self.get_reg(val)?;
                        match ty {
                            Ty::Int => {}
                            Ty::Null | Ty::Float => {
                                let dst = self.regs.stats.reg_of_ty(Ty::Int);
                                self.convert(dst, Ty::Int, reg, ty)?;
                                reg = dst;
                            }
                            Ty::Str => {
                                let dst = self.regs.stats.reg_of_ty(Ty::Int);
                                self.pushl(LL::LenStr(dst.into(), reg.into()));
                                reg = dst;
                            }
                            _ => return err!("invalid type for branch: {:?} :: {:?}", val, ty),
                        }
                        Some(reg)
                    }
                    None => None,
                };
                let (src, dst) = func.cfg.edge_endpoints(eix).unwrap();
                self.frame.cfg.add_edge(src, dst, edge);
            }
            // In the interim, someone may have added some instructions to our basic block when
            // processing a Phi function. Merge in any of those changes.
            let cur_bb = self.frame.cfg.node_weight_mut(ix).unwrap();
            self.stream.insts.extend(cur_bb.insts.drain(..));
            self.stream.exit |= cur_bb.exit;
            mem::swap(cur_bb, self.stream);
        }
        Ok(())
    }

    // Get the register associated with a given identifier and assign a new one if it does not yet
    // have one.
    fn reg_of_ident(&mut self, id: &Ident) -> (u32, Ty) {
        let (reg, ty, _status) = self.reg_of_ident_status(id);
        (reg, ty)
    }

    fn reg_of_ident_status(&mut self, id: &Ident) -> (u32, Ty, RegStatus) {
        if is_unused(*id) {
            // We should not actually store into the "unused" identifier.
            // TODO: remove this once there's better test coverage and we use more unsafe code.
            return (UNUSED, Ty::Int, RegStatus::Local);
        }

        let ((res_reg, res_ty), status) = if id.is_global(self.local_globals) {
            (self.regs.globals[id], RegStatus::Global)
        } else {
            match self.frame.locals.get(id) {
                Some(x) => (x.clone(), RegStatus::Local),
                // In some degenerate cases, we'll run into an uninitialized local. These are
                // always null.
                None if id.sub == 0 => ((NULL_REG, Ty::Null), RegStatus::Local),
                None => panic!("uninitialized variable, malformed IR!"),
            }
        };
        (res_reg, res_ty, status)
    }

    fn pushl(&mut self, i: LL<'a>) {
        // NB: unlike pushr, this isn't the sole entrypoint for adding LLs to the stream. See also
        // the load_slots and store_slots functions.
        self.stream.insts.push_back(Either::Left(i))
    }

    fn pushr(&mut self, i: HighLevel) {
        if let HighLevel::Call { func_id, .. } = i {
            // We do not annotate the edges in the callgraph with call sites or anything, so we
            // only need one edge between each node.
            // NB we can only do this here because calls to pushr always have `stream` matching
            // `frame`. This is not true for pushl, where we swap the stream around in order to fix
            // up type conversions ahead of a Phi node.
            self.cg.update_edge(
                NodeIx::new(self.frame.cur_ident as usize),
                NodeIx::new(func_id as usize),
                (),
            );
        }
        self.stream.insts.push_back(Either::Right(i))
    }

    // Get the register associated with a value. For identifiers this has the same semantics as
    // reg_of_ident. For literals, a new register of the appropriate type is allocated.
    fn get_reg(&mut self, v: &PrimVal<'a>) -> Result<(u32, Ty)> {
        let (reg, ty, _status) = self.get_reg_status(v)?;
        Ok((reg, ty))
    }

    fn get_reg_status(&mut self, v: &PrimVal<'a>) -> Result<(u32, Ty, RegStatus)> {
        use RegStatus::*;
        match v {
            PrimVal::ILit(i) => {
                let nreg = self.regs.stats.new_reg(Ty::Int, Local);
                self.pushl(LL::StoreConstInt(nreg.into(), *i));
                Ok((nreg, Ty::Int, Local))
            }
            PrimVal::FLit(f) => {
                let nreg = self.regs.stats.new_reg(Ty::Float, Local);
                self.pushl(LL::StoreConstFloat(nreg.into(), *f));
                Ok((nreg, Ty::Float, Local))
            }
            PrimVal::StrLit(s) => {
                let nreg = self.regs.stats.new_reg(Ty::Str, Local);
                self.pushl(LL::StoreConstStr(nreg.into(), Str::from(*s).into()));
                Ok((nreg, Ty::Str, Local))
            }
            PrimVal::Var(v) => Ok(self.reg_of_ident_status(v)),
        }
    }

    fn ensure_ty(&mut self, reg: u32, from_ty: Ty, to_ty: Ty) -> Result<u32> {
        if from_ty == to_ty {
            return Ok(reg);
        }
        let new_reg = self.regs.stats.reg_of_ty(to_ty);
        self.convert(new_reg, to_ty, reg, from_ty)?;
        Ok(new_reg)
    }

    // Move src into dst at type Ty.
    fn mov(&mut self, dst_reg: u32, src_reg: u32, ty: Ty) -> Result<()> {
        if let Some(inst) = mov(dst_reg, src_reg, ty)? {
            self.pushl(inst);
        }
        Ok(())
    }

    fn convert(&mut self, dst_reg: u32, dst_ty: Ty, src_reg: u32, src_ty: Ty) -> Result<()> {
        use Ty::*;
        if dst_reg == UNUSED || src_reg == UNUSED {
            return Ok(());
        }
        if dst_reg == src_reg && dst_ty == src_ty {
            return Ok(());
        }

        let res = match (dst_ty, src_ty) {
            (Null, _) => return Ok(()),
            (Float, Null) => LL::StoreConstFloat(dst_reg.into(), Default::default()),
            (Int, Null) => LL::StoreConstInt(dst_reg.into(), Default::default()),
            (Str, Null) => LL::StoreConstStr(dst_reg.into(), Default::default()),
            (Float, Int) => LL::IntToFloat(dst_reg.into(), src_reg.into()),
            (Str, Int) => LL::IntToStr(dst_reg.into(), src_reg.into()),

            (Int, Float) => LL::FloatToInt(dst_reg.into(), src_reg.into()),
            (Str, Float) => LL::FloatToStr(dst_reg.into(), src_reg.into()),

            (Int, Str) => LL::StrToInt(dst_reg.into(), src_reg.into()),
            (Float, Str) => LL::StrToFloat(dst_reg.into(), src_reg.into()),
            (dst, src) => {
                if dst == src {
                    return self.mov(dst_reg, src_reg, dst);
                }
                return err!(
                    "invalid attempt to convert or move dst={:?} src={:?}",
                    dst,
                    src
                );
            }
        };

        Ok(self.pushl(res))
    }

    // Store values into a register at a given type, converting if necessary.
    fn store(&mut self, dst_reg: u32, dst_ty: Ty, src: &PrimVal<'a>) -> Result<()> {
        match src {
            PrimVal::Var(id) => {
                let (src_reg, src_ty) = self.reg_of_ident(id);
                self.convert(dst_reg, dst_ty, src_reg, src_ty)?;
            }
            PrimVal::ILit(i) => {
                if dst_ty == Ty::Int {
                    self.pushl(LL::StoreConstInt(dst_reg.into(), *i));
                } else {
                    let ir = self.regs.stats.reg_of_ty(Ty::Int);
                    self.pushl(LL::StoreConstInt(ir.into(), *i));
                    self.convert(dst_reg, dst_ty, ir, Ty::Int)?;
                }
            }
            PrimVal::FLit(f) => {
                if dst_ty == Ty::Float {
                    self.pushl(LL::StoreConstFloat(dst_reg.into(), *f));
                } else {
                    let ir = self.regs.stats.reg_of_ty(Ty::Float);
                    self.pushl(LL::StoreConstFloat(ir.into(), *f));
                    self.convert(dst_reg, dst_ty, ir, Ty::Float)?;
                }
            }
            PrimVal::StrLit(s) => {
                if dst_ty == Ty::Str {
                    self.pushl(LL::StoreConstStr(dst_reg.into(), Str::from(*s).into()));
                } else {
                    let ir = self.regs.stats.reg_of_ty(Ty::Str);
                    self.pushl(LL::StoreConstStr(ir.into(), Str::from(*s).into()));
                    self.convert(dst_reg, dst_ty, ir, Ty::Str)?;
                }
            }
        };
        Ok(())
    }

    // Generate bytecode for map lookups.
    fn load_map(
        &mut self,
        dst_reg: u32,
        dst_ty: Ty,
        arr_reg: u32,
        arr_ty: Ty,
        key: &PrimVal<'a>,
    ) -> Result<()> {
        if dst_reg == UNUSED {
            return Ok(());
        }
        // Convert `key` if necessary.
        let target_ty = arr_ty.key()?;
        let (mut key_reg, key_ty) = self.get_reg(key)?;
        if target_ty != key_ty {
            let inter = self.regs.stats.reg_of_ty(target_ty);
            self.convert(inter, target_ty, key_reg, key_ty)?;
            key_reg = inter;
        }

        // Determine if we will need to convert the result when storing the variable.
        let arr_val_ty = arr_ty.val()?;
        let load_reg = if dst_ty == arr_val_ty {
            dst_reg
        } else {
            self.regs.stats.reg_of_ty(arr_val_ty)
        };

        // Emit the corresponding instruction.
        use Ty::*;
        match arr_ty {
            MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat | MapStrStr => self
                .pushl(LL::Lookup {
                    map_ty: arr_ty,
                    dst: load_reg,
                    map: arr_reg,
                    key: key_reg,
                }),
            Null | Int | Float | Str | IterInt | IterStr => {
                return err!("[load_map] expected map type, found {:?}", arr_ty)
            }
        };
        // Convert the result: note that if we had load_reg == dst_reg, then this is a noop.
        self.convert(dst_reg, dst_ty, load_reg, arr_val_ty)
    }

    fn builtin(
        &mut self,
        dst_reg: u32,
        dst_ty: Ty,
        bf: &builtins::Function,
        args: &cfg::SmallVec<PrimVal<'a>>,
    ) -> Result<()> {
        use crate::ast::{Binop::*, Unop::*};
        use builtins::Function::*;

        // Compile the argument values
        let mut args_regs = cfg::SmallVec::with_capacity(args.len());
        let mut args_tys = cfg::SmallVec::with_capacity(args.len());
        for arg in args.iter() {
            let (reg, ty) = self.get_reg(arg)?;
            args_regs.push(reg);
            args_tys.push(ty);
        }

        // Now, perform any necessary conversions if input types do not match the argument types.
        let mut conv_regs: cfg::SmallVec<_> = smallvec![UNUSED; args.len()];
        let (conv_tys, res_ty) = bf.type_sig(&args_tys[..])?;

        for (areg, (aty, (creg, cty))) in args_regs.iter().cloned().zip(
            args_tys
                .iter()
                .cloned()
                .zip(conv_regs.iter_mut().zip(conv_tys.iter().cloned())),
        ) {
            if aty == cty {
                *creg = areg;
            } else {
                let reg = self.regs.stats.reg_of_ty(cty);

                self.convert(reg, cty, areg, aty)?;
                *creg = reg;
            }
        }

        let mut res_reg = if dst_ty == res_ty {
            dst_reg
        } else {
            self.regs.stats.reg_of_ty(res_ty)
        };

        // Helper macro for generating code for binary operators
        macro_rules! gen_op {
            ($op:tt, $([$ty:tt, $inst:tt]),* ) => {
                match conv_tys[0] {
                    // TODO implement better handling of "NULL" and get rid of hacks like this.
                    $( Ty::$ty  => if res_reg != UNUSED {
                        self.pushl(LL::$inst(res_reg.into(),
                                        conv_regs[0].into(),
                                        conv_regs[1].into()));
                    }, )*
                    _ => return err!("unexpected operands for {}", stringify!($op)),
                }
            }
        };

        // This match has grown into a bit of a cludge. It maps a (now typed) invocation of a
        // builtin function at the cfg-level to a bytecode instruction. Most of it is pretty
        // mechanical. One subtlety to watch out for is the role of res_reg. reg_reg may be unused,
        // and we are careful not to leak any unused registers into the final bytecode. How to we
        // handle an unsed result register? It depends:
        //
        // 1. If the instruction has no side-effects, we just discard the instruction.
        // 2. If the instruction has side-effects, then we allocate a fresh register and use it.
        //    There are more sophisticated ways to handle this (i.e. reusing "placeholder
        //    registers" within a function), but for now we are keeping things simple.

        match bf {
            Unop(Column) => self.pushl(LL::GetColumn(res_reg.into(), conv_regs[0].into())),
            Unop(Not) => self.pushl(if conv_tys[0] == Ty::Str {
                LL::NotStr(res_reg.into(), conv_regs[0].into())
            } else {
                debug_assert_eq!(conv_tys[0], Ty::Int);
                LL::Not(res_reg.into(), conv_regs[0].into())
            }),
            Unop(Neg) => self.pushl(if conv_tys[0] == Ty::Float {
                LL::NegFloat(res_reg.into(), conv_regs[0].into())
            } else {
                LL::NegInt(res_reg.into(), conv_regs[0].into())
            }),
            Unop(Pos) => self.mov(res_reg, conv_regs[0], conv_tys[0])?,
            Binop(Plus) => gen_op!(Plus, [Float, AddFloat], [Int, AddInt]),
            Binop(Minus) => gen_op!(Minus, [Float, MinusFloat], [Int, MinusInt]),
            Binop(Mult) => gen_op!(Minus, [Float, MulFloat], [Int, MulInt]),
            Binop(Div) => gen_op!(Div, [Float, Div]),
            Binop(Pow) => gen_op!(Pow, [Float, Pow]),
            Binop(Mod) => gen_op!(Mod, [Float, ModFloat], [Int, ModInt]),
            Binop(Concat) => gen_op!(Concat, [Str, Concat]),
            Binop(IsMatch) => gen_op!(IsMatch, [Str, IsMatch]),
            Binop(LT) => gen_op!(LT, [Float, LTFloat], [Int, LTInt], [Str, LTStr]),
            Binop(GT) => gen_op!(GT, [Float, GTFloat], [Int, GTInt], [Str, GTStr]),
            Binop(LTE) => gen_op!(LTE, [Float, LTEFloat], [Int, LTEInt], [Str, LTEStr]),
            Binop(GTE) => gen_op!(GTE, [Float, GTEFloat], [Int, GTEInt], [Str, GTEStr]),
            Binop(EQ) => gen_op!(EQ, [Float, EQFloat], [Int, EQInt], [Str, EQStr]),
            FloatFunc(ff) => {
                if res_reg != UNUSED {
                    match ff.arity() {
                        1 => self.pushl(LL::Float1(*ff, res_reg.into(), conv_regs[0].into())),
                        2 => self.pushl(LL::Float2(*ff, res_reg.into(), conv_regs[0].into(), conv_regs[1].into())),
                        a => return err!("only known float functions have arity 1 and 2, but this one has arity {}", a),
                    }
                }
            }
            IntFunc(bw) => {
                if res_reg != UNUSED {
                    match bw.arity() {
                        1 => self.pushl(LL::Int1(*bw, res_reg.into(), conv_regs[0].into())),
                        2 => self.pushl(LL::Int2(*bw, res_reg.into(), conv_regs[0].into(), conv_regs[1].into())),
                        a => return err!("only known float functions have arity 1 and 2, but this one has arity {}", a),
                    }
                }
            }
            Match => gen_op!(Match, [Str, Match]),
            SubstrIndex => gen_op!(SubstrIndex, [Str, SubstrIndex]),
            Contains => {
                if res_reg != UNUSED {
                    match conv_tys[0] {
                        Ty::MapIntInt
                        | Ty::MapIntStr
                        | Ty::MapIntFloat
                        | Ty::MapStrInt
                        | Ty::MapStrStr
                        | Ty::MapStrFloat => self.pushl(LL::Contains {
                            map_ty: conv_tys[0],
                            dst: res_reg,
                            map: conv_regs[0],
                            key: conv_regs[1],
                        }),
                        Ty::Null | Ty::Int | Ty::Float | Ty::Str | Ty::IterInt | Ty::IterStr => {
                            return err!("unexpected non-map type for Contains: {:?}", conv_tys[0]);
                        }
                    }
                }
            }
            UpdateUsedFields => self.pushl(LL::UpdateUsedFields()),
            SetFI => self.pushl(LL::SetFI(conv_regs[0].into(), conv_regs[1].into())),
            System => {
                if res_reg == UNUSED {
                    res_reg = self.regs.stats.reg_of_ty(res_ty);
                }
                self.pushl(LL::RunCmd(res_reg.into(), conv_regs[0].into()))
            }
            ReadErr => {
                if res_reg != UNUSED {
                    self.pushl(LL::ReadErr(
                        res_reg.into(),
                        conv_regs[0].into(),
                        /*is_file=*/ true,
                    ))
                }
            }
            ReadErrCmd => {
                if res_reg != UNUSED {
                    self.pushl(LL::ReadErr(
                        res_reg.into(),
                        conv_regs[0].into(),
                        /*is_file=*/ false,
                    ))
                }
            }
            Nextline => self.pushl(LL::NextLine(
                res_reg.into(),
                conv_regs[0].into(),
                /*is_file=*/ true,
            )),
            NextlineCmd => self.pushl(LL::NextLine(
                res_reg.into(),
                conv_regs[0].into(),
                /*is_file=*/ false,
            )),
            ReadErrStdin => {
                if res_reg != UNUSED {
                    self.pushl(LL::ReadErrStdin(res_reg.into()))
                }
            }
            NextlineStdin => self.pushl(LL::NextLineStdin(res_reg.into())),
            ReadLineStdinFused => self.pushl(LL::NextLineStdinFused()),
            NextFile => self.pushl(LL::NextFile()),
            Setcol => self.pushl(LL::SetColumn(conv_regs[0].into(), conv_regs[1].into())),
            Sub => {
                if res_reg == UNUSED {
                    res_reg = self.regs.stats.reg_of_ty(res_ty);
                }
                self.pushl(LL::Sub(
                    res_reg.into(),
                    conv_regs[0].into(),
                    conv_regs[1].into(),
                    conv_regs[2].into(),
                ))
            }
            GSub => {
                if res_reg == UNUSED {
                    res_reg = self.regs.stats.reg_of_ty(res_ty);
                }
                self.pushl(LL::GSub(
                    res_reg.into(),
                    conv_regs[0].into(),
                    conv_regs[1].into(),
                    conv_regs[2].into(),
                ))
            }
            EscapeCSV => {
                if res_reg != UNUSED {
                    self.pushl(LL::EscapeCSV(res_reg.into(), conv_regs[0].into()))
                }
            }
            EscapeTSV => {
                if res_reg != UNUSED {
                    self.pushl(LL::EscapeTSV(res_reg.into(), conv_regs[0].into()))
                }
            }
            Substr => {
                if res_reg != UNUSED {
                    self.pushl(LL::Substr(
                        res_reg.into(),
                        conv_regs[0].into(),
                        conv_regs[1].into(),
                        conv_regs[2].into(),
                    ))
                }
            }
            ToInt => self.convert(res_reg, Ty::Int, conv_regs[0], conv_tys[0])?,
            HexToInt => {
                if res_reg != UNUSED {
                    self.pushl(LL::HexStrToInt(res_reg.into(), conv_regs[0].into()))
                }
            }
            Rand => {
                if res_reg == UNUSED {
                    res_reg = self.regs.stats.reg_of_ty(res_ty);
                }
                self.pushl(LL::Rand(res_reg.into()))
            }
            Srand => {
                if res_reg == UNUSED {
                    res_reg = self.regs.stats.reg_of_ty(res_ty);
                }
                self.pushl(LL::Srand(res_reg.into(), conv_regs[0].into()))
            }
            ReseedRng => {
                if res_reg == UNUSED {
                    res_reg = self.regs.stats.reg_of_ty(res_ty);
                }
                self.pushl(LL::ReseedRng(res_reg.into()))
            }
            Split => {
                if res_reg == UNUSED {
                    res_reg = self.regs.stats.reg_of_ty(res_ty);
                }
                self.pushl(if conv_tys[1] == Ty::MapIntStr {
                    LL::SplitInt(
                        res_reg.into(),
                        conv_regs[0].into(),
                        conv_regs[1].into(),
                        conv_regs[2].into(),
                    )
                } else if conv_tys[1] == Ty::MapStrStr {
                    LL::SplitStr(
                        res_reg.into(),
                        conv_regs[0].into(),
                        conv_regs[1].into(),
                        conv_regs[2].into(),
                    )
                } else {
                    return err!("invalid input types to split: {:?}", &conv_tys[..]);
                })
            }
            Length => {
                if res_reg != UNUSED {
                    self.pushl(match conv_tys[0] {
                        Ty::Null => LL::StoreConstInt(res_reg.into(), 0),
                        Ty::MapIntInt
                        | Ty::MapIntStr
                        | Ty::MapIntFloat
                        | Ty::MapStrInt
                        | Ty::MapStrStr
                        | Ty::MapStrFloat => LL::Len {
                            map_ty: conv_tys[0],
                            map: conv_regs[0],
                            dst: res_reg.into(),
                        },
                        Ty::Str => LL::LenStr(res_reg.into(), conv_regs[0].into()),
                        _ => return err!("invalid input type for length: {:?}", &conv_tys[..]),
                    })
                }
            }
            Delete => match &conv_tys[0] {
                Ty::MapIntInt
                | Ty::MapIntStr
                | Ty::MapIntFloat
                | Ty::MapStrInt
                | Ty::MapStrStr
                | Ty::MapStrFloat => self.pushl(LL::Delete {
                    map_ty: conv_tys[0],
                    map: conv_regs[0],
                    key: conv_regs[1],
                }),
                _ => return err!("incorrect parameter types for Delete: {:?}", &conv_tys[..]),
            },
            Close => {
                self.pushl(LL::Close(conv_regs[0].into()));
                assert_eq!(res_ty, Ty::Str);
                if res_reg != UNUSED {
                    self.pushl(LL::StoreConstStr(res_reg.into(), Default::default()));
                }
            }
            JoinCSV => {
                if res_reg != UNUSED {
                    self.pushl(LL::JoinCSV(
                        res_reg.into(),
                        conv_regs[0].into(),
                        conv_regs[1].into(),
                    ))
                }
            }
            JoinTSV => {
                if res_reg != UNUSED {
                    self.pushl(LL::JoinTSV(
                        res_reg.into(),
                        conv_regs[0].into(),
                        conv_regs[1].into(),
                    ))
                }
            }
            JoinCols => {
                if res_reg != UNUSED {
                    self.pushl(LL::JoinColumns(
                        res_reg.into(),
                        conv_regs[0].into(),
                        conv_regs[1].into(),
                        conv_regs[2].into(),
                    ))
                }
            }
        };
        self.convert(dst_reg, dst_ty, res_reg, res_ty)
    }

    fn expr(&mut self, dst_reg: u32, dst_ty: Ty, exp: &cfg::PrimExpr<'a>) -> Result<()> {
        match exp {
            PrimExpr::Val(v) => self.store(dst_reg, dst_ty, v)?,
            PrimExpr::Phi(preds) => {
                let mut pred_regs = SmallVec::with_capacity(preds.len());
                for (prev, id) in preds.iter().cloned() {
                    let (id_reg, id_ty) = self.reg_of_ident(&id);
                    if id_ty == dst_ty {
                        pred_regs.push((prev, id_reg));
                        continue;
                    }
                    // `id` doesn't have the right type. To handle this, we need to reach into the
                    // cfg, grab the stream, and append conversion code to it.
                    //
                    // We do two loads into cfg for the sake of our friend the borrow checker.
                    // NB this is mostly required because Phis have to be at the top of
                    // their basic block. We could probably do the conversions here otherwise.
                    {
                        let other_bb = self.frame.cfg.node_weight_mut(prev).unwrap();
                        mem::swap(self.stream, other_bb);
                    }
                    let conv_reg = self.regs.stats.reg_of_ty(dst_ty);
                    self.convert(conv_reg, dst_ty, id_reg, id_ty)?;
                    {
                        let other_bb = self.frame.cfg.node_weight_mut(prev).unwrap();
                        mem::swap(self.stream, other_bb);
                    }
                    pred_regs.push((prev, conv_reg));
                }
                self.pushr(HighLevel::Phi(dst_reg, dst_ty, pred_regs));
            }
            PrimExpr::CallBuiltin(bf, vs) => self.builtin(dst_reg, dst_ty, bf, vs)?,
            PrimExpr::Sprintf(fmt, args) => {
                // avoid spurious "variant never constructed" warning we get by using LL::Sprintf
                use bytecode::Instr::Sprintf;
                if dst_reg == UNUSED {
                    return Ok(());
                }
                let (mut fmt_reg, fmt_ty) = self.get_reg(fmt)?;
                fmt_reg = self.ensure_ty(fmt_reg, fmt_ty, Ty::Str)?;
                let mut arg_regs = Vec::with_capacity(args.len());
                for a in args {
                    arg_regs.push(self.get_reg(a)?);
                }
                if let Ty::Str = dst_ty {
                    self.pushl(Sprintf {
                        dst: dst_reg.into(),
                        fmt: fmt_reg.into(),
                        args: arg_regs,
                    });
                } else {
                    let reg = self.regs.stats.reg_of_ty(Ty::Str);
                    self.pushl(Sprintf {
                        dst: reg.into(),
                        fmt: fmt_reg.into(),
                        args: arg_regs,
                    });
                    self.convert(dst_reg, dst_ty, reg, Ty::Str)?;
                }
            }
            PrimExpr::CallUDF(func_id, vs) => {
                let mut args = SmallVec::with_capacity(vs.len());
                for v in vs.iter() {
                    args.push(self.get_reg(v)?);
                }
                // Normalize the call
                let true_arity = self.arity[func_id] as usize;
                if args.len() < true_arity {
                    // Not enough arguments; fill in the rest with nulls.
                    // TODO reuse registers here. This is wasteful for functions with a lot of
                    // arguments.
                    let null_ty = types::null_ty();
                    let null_reg = self.regs.stats.reg_of_ty(null_ty);
                    for _ in 0..(true_arity - args.len()) {
                        args.push((null_reg, null_ty));
                    }
                }
                if args.len() > true_arity {
                    // Too many arguments; don't pass the extra.
                    args.truncate(true_arity);
                }
                let monomorphized =
                    self.id_map[&(*func_id, args.iter().map(|(_, y)| y).cloned().collect())];
                let info = &self.func_info[monomorphized as usize];

                if dst_ty != info.ret_ty {
                    let inter_reg = self.regs.stats.reg_of_ty(info.ret_ty);
                    self.pushr(HighLevel::Call {
                        dst_reg: inter_reg,
                        dst_ty: info.ret_ty,
                        func_id: monomorphized,
                        args,
                    });
                    self.convert(dst_reg, dst_ty, inter_reg, info.ret_ty)?;
                } else {
                    self.pushr(HighLevel::Call {
                        dst_reg,
                        dst_ty,
                        func_id: monomorphized,
                        args,
                    });
                }
            }
            PrimExpr::Index(arr, k) => {
                let (arr_reg, arr_ty) = if let PrimVal::Var(arr_id) = arr {
                    self.reg_of_ident(arr_id)
                } else {
                    return err!("attempt to index into scalar literal: {}", arr);
                };
                self.load_map(dst_reg, dst_ty, arr_reg, arr_ty, k)?;
            }
            PrimExpr::IterBegin(pv) => {
                let (arr_reg, arr_ty) = self.get_reg(pv)?;
                if dst_ty.iter()? != arr_ty.key()? {
                    return err!(
                        "illegal iterator assignment {:?} = begin({:?})",
                        dst_ty,
                        arr_ty
                    );
                }
                use Ty::*;
                match arr_ty {
                    MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat | MapStrStr => {
                        self.pushl(LL::IterBegin {
                            map_ty: arr_ty,
                            dst: dst_reg,
                            map: arr_reg,
                        })
                    }
                    // Covered by the error check above
                    Null | Int | Float | Str | IterInt | IterStr => unreachable!(),
                };
            }
            PrimExpr::HasNext(pv) => {
                let target_reg = if dst_ty == Ty::Int {
                    dst_reg
                } else {
                    self.regs.stats.reg_of_ty(Ty::Int)
                };
                let (iter_reg, iter_ty) = self.get_reg(pv)?;
                assert!(matches!(iter_ty.iter(), Ok(Ty::Int) | Ok(Ty::Str)));
                self.pushl(LL::IterHasNext {
                    iter_ty,
                    dst: target_reg,
                    iter: iter_reg,
                });
                self.convert(dst_reg, dst_ty, target_reg, Ty::Int)?
            }
            PrimExpr::Next(pv) => {
                let (iter_reg, iter_ty) = self.get_reg(pv)?;
                let elt_ty = iter_ty.iter()?;
                let target_reg = if dst_ty == elt_ty {
                    dst_reg
                } else {
                    self.regs.stats.reg_of_ty(elt_ty)
                };
                assert!(matches!(iter_ty.iter(), Ok(Ty::Int) | Ok(Ty::Str)));
                self.pushl(LL::IterGetNext {
                    iter_ty,
                    dst: target_reg,
                    iter: iter_reg,
                });
                self.convert(dst_reg, dst_ty, target_reg, elt_ty)?
            }
            PrimExpr::LoadBuiltin(bv) => {
                if dst_reg == UNUSED {
                    return Ok(());
                }
                let target_ty = Ty::from(*bv);
                let target_reg = if target_ty == dst_ty {
                    dst_reg
                } else {
                    self.regs.stats.reg_of_ty(target_ty)
                };
                self.pushl(match target_ty {
                    Ty::Str => LL::LoadVarStr(target_reg.into(), *bv),
                    Ty::Int => LL::LoadVarInt(target_reg.into(), *bv),
                    Ty::MapIntStr => LL::LoadVarIntMap(target_reg.into(), *bv),
                    Ty::MapStrInt => LL::LoadVarStrMap(target_reg.into(), *bv),
                    _ => unreachable!(),
                });
                self.convert(dst_reg, dst_ty, target_reg, target_ty)?
            }
        };
        Ok(())
    }

    fn stmt(&mut self, stmt: &cfg::PrimStmt<'a>) -> Result<()> {
        match stmt {
            PrimStmt::AsgnIndex(arr, pv, pe) => {
                let (a_reg, a_ty) = self.reg_of_ident(arr);
                let (mut k_reg, k_ty) = self.get_reg(pv)?;
                let a_key_ty = a_ty.key()?;
                k_reg = self.ensure_ty(k_reg, k_ty, a_key_ty)?;
                let v_ty = a_ty.val()?;
                let v_reg = self.regs.stats.reg_of_ty(v_ty);
                self.expr(v_reg, v_ty, pe)?;
                use Ty::*;
                match a_ty {
                    MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat | MapStrStr => {
                        self.pushl(LL::Store {
                            map_ty: a_ty,
                            map: a_reg,
                            key: k_reg,
                            val: v_reg,
                        })
                    }
                    Null | Int | Float | Str | IterInt | IterStr => {
                        return err!(
                            "in stmt {:?} computed type is non-map type {:?}",
                            stmt,
                            a_ty
                        )
                    }
                };
            }
            PrimStmt::AsgnVar(id, pe) => {
                let (dst_reg, dst_ty) = self.reg_of_ident(id);
                self.expr(dst_reg, dst_ty, pe)?;
            }
            PrimStmt::SetBuiltin(v, pe) => {
                let ty = Ty::from(*v);
                let reg = self.regs.stats.reg_of_ty(ty);
                self.expr(reg, ty, pe)?;
                use Ty::*;
                self.pushl(match ty {
                    Str => LL::StoreVarStr(*v, reg.into()),
                    MapIntStr => LL::StoreVarIntMap(*v, reg.into()),
                    MapStrInt => LL::StoreVarStrMap(*v, reg.into()),
                    Int => LL::StoreVarInt(*v, reg.into()),
                    _ => return err!("unexpected type for variable {} : {:?}", v, ty),
                });
            }
            PrimStmt::Return(v) => {
                let (mut v_reg, v_ty) = self.get_reg(v)?;
                let ret_ty = self.func_info[self.frame.cur_ident as usize].ret_ty;
                v_reg = self.ensure_ty(v_reg, v_ty, ret_ty)?;
                self.pushr(HighLevel::Ret(v_reg, ret_ty));
                self.stream.exit = true;
            }
            PrimStmt::PrintAll(args, out) => {
                use bytecode::Instr::PrintAll;
                let mut arg_regs = Vec::with_capacity(args.len());
                for a in args {
                    let (a_reg, a_ty) = self.get_reg(a)?;
                    arg_regs.push(self.ensure_ty(a_reg, a_ty, Ty::Str)?.into());
                }
                let out_reg = if let Some((out, append)) = out {
                    // Would use map, but I supposed we have no equivalent to sequenceA_ and/or
                    // monad transformers.
                    let (mut out_reg, out_ty) = self.get_reg(out)?;
                    out_reg = self.ensure_ty(out_reg, out_ty, Ty::Str)?;
                    Some((out_reg.into(), *append))
                } else {
                    None
                };
                self.pushl(PrintAll {
                    output: out_reg,
                    args: arg_regs,
                });
            }
            PrimStmt::Printf(fmt, args, out) => {
                // avoid spurious "variant never constructed" warning we get by using LL::Printf
                use bytecode::Instr::Printf;
                let (mut fmt_reg, fmt_ty) = self.get_reg(fmt)?;
                fmt_reg = self.ensure_ty(fmt_reg, fmt_ty, Ty::Str)?;
                let mut arg_regs = Vec::with_capacity(args.len());
                for a in args {
                    arg_regs.push(self.get_reg(a)?);
                }
                let out_reg = if let Some((out, append)) = out {
                    let (mut out_reg, out_ty) = self.get_reg(out)?;
                    out_reg = self.ensure_ty(out_reg, out_ty, Ty::Str)?;
                    Some((out_reg.into(), *append))
                } else {
                    None
                };
                self.pushl(Printf {
                    output: out_reg,
                    fmt: fmt_reg.into(),
                    args: arg_regs,
                });
            }
            PrimStmt::IterDrop(v) => {
                let (reg, ty) = self.get_reg(v)?;
                self.pushr(HighLevel::DropIter(reg, ty))
            }
        };
        Ok(())
    }
}
