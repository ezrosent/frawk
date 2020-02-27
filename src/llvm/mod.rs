use crate::builtins::Variable;
use crate::bytecode::{self, Accum};
use crate::common::{Either, NodeIx, NumTy, Result};
use crate::compile::{self, Ty, Typer};
use crate::libc::c_char;
use crate::runtime;

use crate::smallvec::{self, smallvec};
use hashbrown::{HashMap, HashSet};
use llvm_sys::{
    analysis::{LLVMVerifierFailureAction, LLVMVerifyModule},
    core::*,
    execution_engine::*,
    prelude::*,
    target::*,
};
use petgraph::visit::Dfs;

pub(crate) mod intrinsics;
use intrinsics::IntrinsicMap;

use std::ffi::{CStr, CString};
use std::mem::{self, MaybeUninit};
use std::ptr;

type Pred = llvm_sys::LLVMIntPredicate;
type FPred = llvm_sys::LLVMRealPredicate;

type SmallVec<T> = smallvec::SmallVec<[T; 2]>;

#[derive(Clone)]
struct IterState {
    iter_ptr: LLVMValueRef,  /* ptr to elt type */
    cur_index: LLVMValueRef, /* ptr to integer */
    len: LLVMValueRef,       /* integer */
}

struct Function {
    // NB this is always the same as the matching FuncInfo val; it's in here for convenience.
    val: LLVMValueRef,
    builder: LLVMBuilderRef,
    locals: HashMap<(NumTy, Ty), LLVMValueRef>,
    iters: HashMap<(NumTy, Ty), IterState>,
    skip_drop: HashSet<(NumTy, Ty)>,
    args: SmallVec<(NumTy, Ty)>,
    id: usize,
}

struct FuncInfo {
    val: LLVMValueRef,
    globals: HashMap<(NumTy, Ty), usize>,
    num_args: usize,
}

struct View<'a> {
    f: &'a mut Function,
    decls: &'a Vec<FuncInfo>,
    tmap: &'a TypeMap,
    intrinsics: &'a IntrinsicMap,
    ctx: LLVMContextRef,
    module: LLVMModuleRef,
    printfs: &'a mut HashMap<(SmallVec<Ty>, PrintfKind), LLVMValueRef>,
    // We keep an extra builder always pointed at the start of the function. This is because
    // binding new string values requires an `alloca`; and we do not want to call `alloca` where a
    // string variable is referenced: for example, we do not want to call alloca in a loop.
    entry_builder: LLVMBuilderRef,
}

impl Drop for Function {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeBuilder(self.builder);
        }
    }
}

#[derive(Copy, Clone)]
struct TypeRef {
    base: LLVMTypeRef,
    ptr: LLVMTypeRef,
}

impl TypeRef {
    fn null() -> TypeRef {
        TypeRef {
            base: ptr::null_mut(),
            ptr: ptr::null_mut(),
        }
    }
}

// Common LLVM types used in code generation.
struct TypeMap {
    // Map from compile::Ty => TypeRef
    table: [TypeRef; compile::NUM_TYPES],
    runtime_ty: LLVMTypeRef,
    var_ty: LLVMTypeRef,
}

impl TypeMap {
    fn new(ctx: LLVMContextRef) -> TypeMap {
        unsafe {
            TypeMap {
                table: [TypeRef::null(); compile::NUM_TYPES],
                runtime_ty: LLVMPointerType(LLVMVoidTypeInContext(ctx), 0),
                var_ty: LLVMIntTypeInContext(ctx, (mem::size_of::<usize>() * 8) as libc::c_uint),
            }
        }
    }

    #[inline(always)]
    fn init(&mut self, ty: Ty, r: TypeRef) {
        self.table[ty as usize] = r;
    }

    #[inline(always)]
    fn get_ty(&self, ty: Ty) -> LLVMTypeRef {
        self.table[ty as usize].base
    }

    #[inline(always)]
    fn get_ptr_ty(&self, ty: Ty) -> LLVMTypeRef {
        self.table[ty as usize].ptr
    }
}

#[derive(Copy, Clone, Hash, Eq, PartialEq)]
enum PrintfKind {
    Stdout,
    File,
    Sprintf,
}

pub(crate) struct Generator<'a, 'b> {
    types: &'b mut Typer<'a>,
    ctx: LLVMContextRef,
    module: LLVMModuleRef,
    engine: LLVMExecutionEngineRef,
    decls: Vec<FuncInfo>,
    funcs: Vec<Function>,
    type_map: TypeMap,
    intrinsics: IntrinsicMap,
    printfs: HashMap<(SmallVec<Ty>, PrintfKind), LLVMValueRef>,
}

impl<'a, 'b> Drop for Generator<'a, 'b> {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeModule(self.module);
        }
    }
}

unsafe fn alloc_local(
    builder: LLVMBuilderRef,
    ty: Ty,
    tmap: &TypeMap,
    intrinsics: &IntrinsicMap,
) -> Result<LLVMValueRef> {
    use Ty::*;
    let val = match ty {
        Int => LLVMConstInt(tmap.get_ty(Int), 0, /*sign_extend=*/ 1),
        Float => LLVMConstReal(tmap.get_ty(Float), 0.0),
        Str => {
            let str_ty = tmap.get_ty(Str);
            let v = LLVMConstInt(str_ty, 0, /*sign_extend=*/ 0);
            let v_loc = LLVMBuildAlloca(builder, str_ty, c_str!(""));
            LLVMBuildStore(builder, v, v_loc);
            v_loc
        }
        MapIntInt | MapIntStr | MapIntFloat | MapStrInt | MapStrStr | MapStrFloat => {
            let fname = match ty {
                MapIntInt => "alloc_intint",
                MapIntFloat => "alloc_intfloat",
                MapIntStr => "alloc_intstr",
                MapStrInt => "alloc_strint",
                MapStrFloat => "alloc_strfloat",
                MapStrStr => "alloc_strstr",
                _ => unreachable!(),
            };
            LLVMBuildCall(
                builder,
                intrinsics.get(fname),
                ptr::null_mut(),
                0,
                c_str!(""),
            )
        }
        IterInt | IterStr => return err!("we should not be default-allocating any iterators"),
    };
    Ok(val)
}

impl<'a, 'b> Generator<'a, 'b> {
    pub unsafe fn optimize(&mut self, main: LLVMValueRef) {
        // TODO: allow us to customize the opt level once we have more command-line flags, etc.
        //
        // Based on optimize_module in weld, in turn based on similar code in the LLVM opt tool.
        use llvm_sys::transforms::pass_manager_builder::*;
        static OPT: bool = true;
        let mpm = LLVMCreatePassManager();
        let fpm = LLVMCreateFunctionPassManagerForModule(self.module);

        let builder = LLVMPassManagerBuilderCreate();
        LLVMPassManagerBuilderSetOptLevel(builder, if OPT { 3 } else { 0 });
        LLVMPassManagerBuilderSetSizeLevel(builder, 0);
        if OPT {
            LLVMPassManagerBuilderUseInlinerWithThreshold(builder, 250);
        }

        LLVMPassManagerBuilderPopulateFunctionPassManager(builder, fpm);
        LLVMPassManagerBuilderPopulateModulePassManager(builder, mpm);
        LLVMPassManagerBuilderDispose(builder);

        for f in self.decls.iter() {
            LLVMRunFunctionPassManager(fpm, f.val);
        }
        for fv in self.printfs.values() {
            LLVMRunFunctionPassManager(fpm, *fv);
        }
        LLVMRunFunctionPassManager(fpm, main);

        LLVMFinalizeFunctionPassManager(fpm);
        LLVMRunPassManager(mpm, self.module);
        LLVMDisposePassManager(fpm);
        LLVMDisposePassManager(mpm);
    }

    pub unsafe fn init(types: &'b mut Typer<'a>) -> Result<Generator<'a, 'b>> {
        if llvm_sys::support::LLVMLoadLibraryPermanently(ptr::null()) != 0 {
            return err!("failed to load in-process library");
        }
        let ctx = LLVMContextCreate();
        let module = LLVMModuleCreateWithNameInContext(c_str!("frawk_main"), ctx);
        // JIT-specific initialization.
        LLVM_InitializeNativeTarget();
        LLVM_InitializeNativeAsmPrinter();
        LLVMLinkInMCJIT();
        let mut maybe_engine = MaybeUninit::<LLVMExecutionEngineRef>::uninit();
        let mut err: *mut c_char = ptr::null_mut();
        if LLVMCreateExecutionEngineForModule(maybe_engine.as_mut_ptr(), module, &mut err) != 0 {
            let res = err!(
                "failed to create program: {}",
                CStr::from_ptr(err).to_str().unwrap()
            );
            LLVMDisposeMessage(err);
            return res;
        }
        let engine = maybe_engine.assume_init();
        let nframes = types.frames.len();
        let mut res = Generator {
            types,
            ctx,
            module,
            engine,
            decls: Vec::with_capacity(nframes),
            funcs: Vec::with_capacity(nframes),
            type_map: TypeMap::new(ctx),
            intrinsics: intrinsics::register(module, ctx),
            printfs: Default::default(),
        };
        res.build_map();
        res.build_decls();
        for i in 0..nframes {
            res.gen_function(i)?;
        }
        Ok(res)
    }

    unsafe fn dump_module_inner(&mut self) -> String {
        let c_str = LLVMPrintModuleToString(self.module);
        let res = CStr::from_ptr(c_str).to_string_lossy().into_owned();
        libc::free(c_str as *mut _);
        res
    }

    pub unsafe fn dump_module(&mut self) -> Result<String> {
        self.gen_main()?;
        self.verify()?;
        Ok(self.dump_module_inner())
    }

    // For benchmarking.
    pub unsafe fn _compile_main(&mut self) -> Result<()> {
        self.gen_main()?;
        self.verify()?;
        let addr = LLVMGetFunctionAddress(self.engine, c_str!("__frawk_main"));
        ptr::read_volatile(&addr);
        Ok(())
    }

    pub unsafe fn run_main(
        &mut self,
        stdin: impl std::io::Read + 'static,
        stdout: impl std::io::Write + 'static,
    ) -> Result<()> {
        let mut rt = intrinsics::Runtime::new(stdin, stdout);
        self.gen_main()?;
        self.verify()?;
        let addr = LLVMGetFunctionAddress(self.engine, c_str!("__frawk_main"));
        let main_fn = mem::transmute::<u64, extern "C" fn(*mut libc::c_void)>(addr);
        main_fn((&mut rt) as *mut _ as *mut libc::c_void);
        Ok(())
    }

    unsafe fn build_map(&mut self) {
        use mem::size_of;
        let make = |ty| TypeRef {
            base: ty,
            ptr: LLVMPointerType(ty, 0),
        };
        let voidptr = LLVMPointerType(LLVMVoidTypeInContext(self.ctx), 0);
        self.type_map.init(
            Ty::Int,
            make(LLVMIntTypeInContext(
                self.ctx,
                (size_of::<runtime::Int>() * 8) as libc::c_uint,
            )),
        );
        self.type_map
            .init(Ty::Float, make(LLVMDoubleTypeInContext(self.ctx)));
        self.type_map.init(
            Ty::Str,
            make(LLVMIntTypeInContext(self.ctx, 128 as libc::c_uint)),
        );
        self.type_map.init(Ty::MapIntInt, make(voidptr));
        self.type_map.init(Ty::MapIntFloat, make(voidptr));
        self.type_map.init(Ty::MapIntStr, make(voidptr));
        self.type_map.init(Ty::MapStrInt, make(voidptr));
        self.type_map.init(Ty::MapStrFloat, make(voidptr));
        self.type_map.init(Ty::MapStrStr, make(voidptr));
        // NB: iterators do not have types of their own, and we should never ask for their types.
        // See the IterState type and its uses for more info.
        self.type_map.init(Ty::IterInt, TypeRef::null());
        self.type_map.init(Ty::IterStr, TypeRef::null());
    }

    fn llvm_ty(&self, ty: Ty) -> LLVMTypeRef {
        if let Ty::Str = ty {
            self.type_map.get_ptr_ty(ty)
        } else {
            self.type_map.get_ty(ty)
        }
    }

    fn llvm_ptr_ty(&self, ty: Ty) -> LLVMTypeRef {
        self.type_map.get_ptr_ty(ty)
    }

    unsafe fn build_decls(&mut self) {
        let global_refs = self.types.get_global_refs();
        debug_assert_eq!(global_refs.len(), self.types.func_info.len());
        let mut arg_tys = SmallVec::new();
        for (i, (info, refs)) in self
            .types
            .func_info
            .iter()
            .zip(global_refs.iter())
            .enumerate()
        {
            let mut globals = HashMap::new();
            let name = CString::new(format!("_frawk_udf_{}", i)).unwrap();
            // First, we add the listed function parameters.
            arg_tys.extend(info.arg_tys.iter().map(|ty| self.llvm_ty(*ty)));
            // Then, we add on the referenced globals.
            for (reg, ty) in refs.iter().cloned() {
                let ix = arg_tys.len();
                arg_tys.push(self.llvm_ptr_ty(ty));
                // Vals are ignored if we are main.
                globals.insert((reg, ty), ix);
            }
            // Finally, we add a pointer to the runtime; always the last parameter.
            arg_tys.push(self.type_map.runtime_ty);
            let ty = LLVMFunctionType(
                self.type_map.get_ty(info.ret_ty),
                arg_tys.as_mut_ptr(),
                arg_tys.len() as u32,
                /*IsVarArg=*/ 0,
            );
            let val = LLVMAddFunction(self.module, name.as_ptr(), ty);
            let builder = LLVMCreateBuilderInContext(self.ctx);
            // We make these private, as we generate a separate main that calls into them. This
            // way, function bodies that get inlined into main do not have to show up in generated
            // code.
            LLVMSetLinkage(val, llvm_sys::LLVMLinkage::LLVMLinkerPrivateLinkage);
            let id = self.funcs.len();
            self.decls.push(FuncInfo {
                val,
                globals,
                num_args: arg_tys.len(),
            });
            let args: SmallVec<_> = self.types.frames[i]
                .arg_regs
                .iter()
                .cloned()
                .zip(self.types.func_info[i].arg_tys.iter().cloned())
                .collect();
            self.funcs.push(Function {
                val,
                builder,
                iters: Default::default(),
                locals: Default::default(),
                skip_drop: Default::default(),
                args,
                id,
            });
            arg_tys.clear();
        }
    }

    unsafe fn alloc_local(&self, builder: LLVMBuilderRef, ty: Ty) -> Result<LLVMValueRef> {
        alloc_local(builder, ty, &self.type_map, &self.intrinsics)
    }

    unsafe fn gen_main(&mut self) -> Result<()> {
        let ty = LLVMFunctionType(
            LLVMVoidTypeInContext(self.ctx),
            &mut self.type_map.runtime_ty,
            1,
            /*IsVarArg=*/ 0,
        );
        let decl = LLVMAddFunction(self.module, c_str!("__frawk_main"), ty);
        let builder = LLVMCreateBuilderInContext(self.ctx);
        let bb = LLVMAppendBasicBlockInContext(self.ctx, decl, c_str!(""));
        LLVMPositionBuilderAtEnd(builder, bb);

        // We need to allocate all of the global variables that our main function uses, and then
        // pass them as arguments, along with the runtime.
        let main_info = &self.decls[self.types.main_offset];
        let mut args: SmallVec<_> = smallvec![ptr::null_mut(); main_info.num_args];
        for ((_reg, ty), arg_ix) in main_info.globals.iter() {
            let local = self.alloc_local(builder, *ty)?;
            let param = if let Ty::Str = ty {
                // Already a pointer; we're good to go!
                local
            } else {
                let loc = LLVMBuildAlloca(builder, self.llvm_ty(*ty), c_str!(""));
                LLVMBuildStore(builder, local, loc);
                loc
            };
            args[*arg_ix] = param;
        }

        // Pass the runtime last.
        args[main_info.num_args - 1] = LLVMGetParam(decl, 0);
        LLVMBuildCall(
            builder,
            main_info.val,
            args.as_mut_ptr(),
            args.len() as libc::c_uint,
            c_str!(""),
        );
        LLVMBuildRetVoid(builder);
        LLVMDisposeBuilder(builder);
        self.optimize(decl);
        Ok(())
    }

    unsafe fn verify(&mut self) -> Result<()> {
        let mut error = ptr::null_mut();
        let code = LLVMVerifyModule(
            self.module,
            LLVMVerifierFailureAction::LLVMReturnStatusAction,
            &mut error,
        );
        let res = if code != 0 {
            let err_str = CStr::from_ptr(error).to_string_lossy().into_owned();
            err!("Module verification failed: {}", err_str)
        } else {
            Ok(())
        };
        LLVMDisposeMessage(error);
        res
    }

    unsafe fn gen_function(&mut self, func_id: usize) -> Result<()> {
        use compile::HighLevel::*;
        let frame = &self.types.frames[func_id];
        let builder = self.funcs[func_id].builder;
        let entry_bb =
            LLVMAppendBasicBlockInContext(self.ctx, self.funcs[func_id].val, c_str!("entry"));
        let entry_builder = LLVMCreateBuilderInContext(self.ctx);
        LLVMPositionBuilderAtEnd(entry_builder, entry_bb);
        let mut bbs = Vec::with_capacity(frame.cfg.node_count());
        for _ in 0..frame.cfg.node_count() {
            let bb = LLVMAppendBasicBlockInContext(self.ctx, self.funcs[func_id].val, c_str!(""));
            bbs.push(bb);
        }
        LLVMPositionBuilderAtEnd(builder, bbs[0]);
        for (local, (reg, ty)) in frame.locals.iter() {
            // implicitly-declared locals are just the ones with a subscript of 0.
            if local.sub == 0 {
                let val = self.alloc_local(self.funcs[func_id].builder, *ty)?;
                self.funcs[func_id].locals.insert((*reg, *ty), val);
            }
        }

        // As of writing; we'll only ever have a single return statement for a given function, but
        // we do not lose very much by having this function support multiple returns if we decide
        // to refactor some of the higher-level code in the future.
        let mut exits = Vec::with_capacity(1);
        let mut phis = Vec::new();
        let f = &mut self.funcs[func_id];
        let mut view = View {
            f,
            tmap: &self.type_map,
            intrinsics: &self.intrinsics,
            decls: &self.decls,
            printfs: &mut self.printfs,
            ctx: self.ctx,
            module: self.module,
            entry_builder,
        };
        // handle arguments
        for (i, arg) in view.f.args.iter().cloned().enumerate() {
            let argv = LLVMGetParam(view.f.val, i as libc::c_uint);
            // We insert into `locals` directly because we know these aren't globals, and we want
            // to avoid the extra ref/drop for string params.
            view.f.locals.insert(arg, argv);
            view.f.skip_drop.insert(arg);
        }
        // Why use DFS? The main issue we want to avoid is encountering registers that we haven't
        // defined yet. There are two cases to consider:
        // * Globals: these are all pre-declared, so if we encounter one we should be fine.
        // * Locals: these are in SSA form, so "definition dominates use." In other words, any path
        //   through the CFG starting at the entry node will pass through a definition for a node
        //   before it is referenced.
        let mut dfs_walker = Dfs::new(&frame.cfg, NodeIx::new(0));
        while let Some(n) = dfs_walker.next(&frame.cfg) {
            let i = n.index();
            let bb = frame.cfg.node_weight(n).unwrap();
            LLVMPositionBuilderAtEnd(view.f.builder, bbs[i]);
            // Generate instructions for this basic block.
            for (j, inst) in bb.iter().enumerate() {
                match inst {
                    Either::Left(ll) => view.gen_ll_inst(ll)?,
                    Either::Right(hl) => {
                        // We record `ret` and `phi` for extra processing once the rest of the
                        // instructions have been generated.
                        view.gen_hl_inst(hl)?;
                        match hl {
                            Ret(_, _) => exits.push((i, j)),
                            Phi(_, _, _) => phis.push((i, j)),
                            DropIter(_, _) | Call { .. } => {}
                        }
                    }
                }
            }
            let mut walker = frame.cfg.neighbors(NodeIx::new(i)).detach();
            let mut tcase = None;
            let mut ecase = None;
            while let Some(e) = walker.next_edge(&frame.cfg) {
                let (_, t) = frame.cfg.edge_endpoints(e).unwrap();
                let bb = bbs[t.index()];
                if let Some(e) = frame.cfg.edge_weight(e).unwrap().clone() {
                    assert!(tcase.is_none());
                    tcase = Some((e, bb));
                } else {
                    assert!(ecase.is_none());
                    ecase = Some(bb);
                }
            }
            // Not all nodes (e.g. rets) have outgoing edges.
            if let Some(ecase) = ecase {
                view.branch(tcase, ecase)?;
            }
        }

        // We don't do return statements when we first find them, because returns are responsible
        // for dropping all local variables, and we aren't guaranteed that our traversal will visit
        // the exit block last.
        let node_weight = |bb, inst| &frame.cfg.node_weight(NodeIx::new(bb)).unwrap()[inst];
        for (exit_bb, return_inst) in exits.into_iter() {
            LLVMPositionBuilderAtEnd(view.f.builder, bbs[exit_bb]);
            let var = if let Either::Right(Ret(reg, ty)) = node_weight(exit_bb, return_inst) {
                (*reg, *ty)
            } else {
                unreachable!()
            };
            if view.has_var(var) {
                view.ret(var)?
            } else {
                // var isn't bound.
                // This can happen when a return-specific variable is never assigned in a given
                // function. It probably means this function is "void"; but because you can assign
                // to the result of a void function we need to allocate something here.
                let ty = var.1;
                let val = alloc_local(view.f.builder, ty, &self.type_map, &self.intrinsics)?;
                view.ret_val(val, ty)?
            }
        }

        // Now that we have initialized all local variables, we can wire in predecessors to phis.
        let mut preds = SmallVec::new();
        let mut blocks = SmallVec::new();
        for (phi_bb, phi_inst) in phis.into_iter() {
            if let Either::Right(Phi(reg, ty, ps)) = node_weight(phi_bb, phi_inst) {
                let phi_node = view.get_local((*reg, *ty))?;
                for (pred_bb, pred_reg) in ps.iter() {
                    preds.push(view.get_local((*pred_reg, *ty))?);
                    blocks.push(bbs[pred_bb.index()]);
                }
                LLVMAddIncoming(
                    phi_node,
                    preds.as_mut_ptr(),
                    blocks.as_mut_ptr(),
                    preds.len() as libc::c_uint,
                );
            } else {
                unreachable!()
            }
            preds.clear();
            blocks.clear();
        }
        LLVMBuildBr(entry_builder, bbs[0]);
        LLVMDisposeBuilder(entry_builder);
        Ok(())
    }
}

impl<'a> View<'a> {
    unsafe fn has_var(&self, var: (NumTy, Ty)) -> bool {
        self.f.locals.get(&var).is_some() || self.decls[self.f.id].globals.get(&var).is_some()
    }
    unsafe fn get_local(&self, local: (NumTy, Ty)) -> Result<LLVMValueRef> {
        if let Some(v) = self.f.locals.get(&local) {
            Ok(*v)
        } else if let Some(ix) = self.decls[self.f.id].globals.get(&local) {
            let gv = LLVMGetParam(self.f.val, *ix as libc::c_uint);
            Ok(if let Ty::Str = local.1 {
                // no point in loading the string directly. We manipulate them as pointers.
                gv
            } else {
                // XXX: do we need to ref maps here?
                LLVMBuildLoad(self.f.builder, gv, c_str!(""))
            })
        } else {
            // We'll see if we need to be careful about iteration order here. We may want to do a
            // DFS starting at entry.
            err!(
                "unbound variable {:?} (must call bind_val on it before)",
                local
            )
        }
    }

    fn is_global(&self, reg: (NumTy, Ty)) -> bool {
        self.decls[self.f.id].globals.get(&reg).is_some()
    }

    unsafe fn var_val(&self, v: &Variable) -> LLVMValueRef {
        LLVMConstInt(self.tmap.var_ty, *v as u64, /*sign_extend=*/ 0)
    }

    unsafe fn drop_reg(&mut self, reg: (NumTy, Ty)) -> Result<()> {
        let val = self.get_local(reg)?;
        self.drop_val(val, reg.1)
    }

    unsafe fn drop_val(&mut self, mut val: LLVMValueRef, ty: Ty) -> Result<()> {
        use Ty::*;
        match ty {
            MapIntInt | MapIntStr | MapIntFloat | MapStrInt | MapStrStr | MapStrFloat => {
                let func = self.intrinsics.get("drop_map");
                LLVMBuildCall(self.f.builder, func, &mut val, 1, c_str!(""));
            }
            Str => {
                let func = self.intrinsics.get("drop_str");
                LLVMBuildCall(self.f.builder, func, &mut val, 1, c_str!(""));
            }
            _ => {}
        };
        Ok(())
    }

    unsafe fn call(&mut self, func: &'static str, args: &mut [LLVMValueRef]) -> LLVMValueRef {
        let f = self.intrinsics.get(func);
        LLVMBuildCall(
            self.f.builder,
            f,
            args.as_mut_ptr(),
            args.len() as libc::c_uint,
            c_str!(""),
        )
    }

    unsafe fn bind_reg<T>(&mut self, r: &bytecode::Reg<T>, to: LLVMValueRef)
    where
        bytecode::Reg<T>: Accum,
    {
        self.bind_val(r.reflect(), to);
    }

    unsafe fn alloca(&mut self, ty: Ty) -> LLVMValueRef {
        let ty = self.tmap.get_ty(ty);
        let res = LLVMBuildAlloca(self.entry_builder, ty, c_str!(""));
        let v = LLVMConstInt(ty, 0, /*sign_extend=*/ 0);
        LLVMBuildStore(self.entry_builder, v, res);
        res
    }

    unsafe fn iter_begin(&mut self, dst: (NumTy, Ty), arr: (NumTy, Ty)) -> Result<()> {
        use Ty::*;
        let arrv = self.get_local(arr)?;
        let (len_fn, begin_fn) = match arr.1 {
            MapIntInt => ("len_intint", "iter_intint"),
            MapIntStr => ("len_intstr", "iter_intstr"),
            MapIntFloat => ("len_intfloat", "iter_intfloat"),
            MapStrInt => ("len_strint", "iter_strint"),
            MapStrStr => ("len_strstr", "iter_strstr"),
            MapStrFloat => ("len_strfloat", "iter_strfloat"),
            _ => return err!("iterating over non-map type: {:?}", arr.1),
        };

        let iter_ptr = self.call(begin_fn, &mut [arrv]);
        let cur_index = self.alloca(Ty::Int);
        let len = self.call(len_fn, &mut [arrv]);
        let _old = self.f.iters.insert(
            dst,
            IterState {
                iter_ptr,
                cur_index,
                len,
            },
        );
        debug_assert!(_old.is_none());
        Ok(())
    }

    fn get_iter(&self, iter: (NumTy, Ty)) -> Result<&IterState> {
        if let Some(istate) = self.f.iters.get(&iter) {
            Ok(istate)
        } else {
            err!("unbound iterator: {:?}", iter)
        }
    }

    unsafe fn iter_hasnext(&mut self, iter: (NumTy, Ty), dst: (NumTy, Ty)) -> Result<()> {
        let istate = self.get_iter(iter)?;
        let cur = LLVMBuildLoad(self.f.builder, istate.cur_index, c_str!(""));
        let len = istate.len;
        let hasnext = self.cmp(Either::Left(Pred::LLVMIntULT), cur, len);
        self.bind_val(dst, hasnext);
        Ok(())
    }

    unsafe fn iter_getnext(&mut self, iter: (NumTy, Ty), dst: (NumTy, Ty)) -> Result<()> {
        let (res, res_loc) = {
            let istate = self.get_iter(iter)?;
            let cur = LLVMBuildLoad(self.f.builder, istate.cur_index, c_str!(""));
            let indices = &mut [cur];
            let res_loc = LLVMBuildGEP(
                self.f.builder,
                istate.iter_ptr,
                indices.as_mut_ptr(),
                indices.len() as libc::c_uint,
                c_str!(""),
            );
            let res = LLVMBuildLoad(self.f.builder, res_loc, c_str!(""));

            let next_ix = LLVMBuildAdd(
                self.f.builder,
                cur,
                LLVMConstInt(self.tmap.get_ty(Ty::Int), 1, /*sign_extend=*/ 1),
                c_str!(""),
            );
            LLVMBuildStore(self.f.builder, next_ix, istate.cur_index);
            (res, res_loc)
        };
        if let Ty::Str = dst.1 {
            self.call("ref_str", &mut [res_loc]);
        }
        self.bind_val(dst, res);

        Ok(())
    }

    unsafe fn bind_val(&mut self, val: (NumTy, Ty), to: LLVMValueRef) {
        // if val is global, then find the relevant parameter and store it directly.
        // if val is an existing local, fail
        // if val.ty is a string, alloca a new string, store it, then bind the result.
        // otherwise, just bind the result directly.
        #[cfg(debug_assertions)]
        {
            if let Ty::Str = val.1 {
                use llvm_sys::LLVMTypeKind::*;
                // make sure we are passing string values, not pointers here.
                assert_eq!(LLVMGetTypeKind(LLVMTypeOf(to)), LLVMIntegerTypeKind);
            }
        }
        // Note: we ref strings ahead of time, either before call8ing bind_val in a MovStr, or as
        // the result of a function call.
        use Ty::*;
        if let Some(ix) = self.decls[self.f.id].globals.get(&val) {
            // We're storing into a global variable. If it's a string or map, that means we have to
            // alter the reference counts appropriately.
            //  - if Str, call drop, store, call ref.
            //  - if Map, load the value, drop it, ref `to` then store it
            //  - otherwise, just store it directly
            let param = LLVMGetParam(self.f.val, *ix as libc::c_uint);
            let new_global = to;
            match val.1 {
                MapIntInt | MapIntStr | MapIntFloat | MapStrInt | MapStrStr | MapStrFloat => {
                    let prev_global = LLVMBuildLoad(self.f.builder, param, c_str!(""));
                    self.call("drop_map", &mut [prev_global]);
                    self.call("ref_map", &mut [new_global]);
                    LLVMBuildStore(self.f.builder, new_global, param);
                }
                Str => {
                    self.call("drop_str", &mut [param]);
                    LLVMBuildStore(self.f.builder, new_global, param);
                    self.call("ref_str", &mut [param]);
                }
                _ => {
                    LLVMBuildStore(self.f.builder, new_global, param);
                }
            };
            return;
        }
        debug_assert!(self.f.locals.get(&val).is_none());
        match val.1 {
            MapIntInt | MapIntStr | MapIntFloat | MapStrInt | MapStrStr | MapStrFloat => {
                self.call("ref_map", &mut [to]);
            }
            Str => {
                let loc = self.alloca(Ty::Str);
                self.call("drop_str", &mut [loc]);
                LLVMBuildStore(self.f.builder, to, loc);
                self.f.locals.insert(val, loc);
                return;
            }
            _ => {}
        }
        self.f.locals.insert(val, to);
    }

    unsafe fn lookup_map(
        &mut self,
        map: (NumTy, Ty),
        key: (NumTy, Ty),
        dst: (NumTy, Ty),
    ) -> Result<()> {
        assert_eq!(map.1.key()?, key.1);
        assert_eq!(map.1.val()?, dst.1);
        use Ty::*;
        let func = match map.1 {
            MapIntInt => "lookup_intint",
            MapIntFloat => "lookup_intfloat",
            MapIntStr => "lookup_intstr",
            MapStrInt => "lookup_strint",
            MapStrFloat => "lookup_strfloat",
            MapStrStr => "lookup_strstr",
            _ => unreachable!(),
        };
        let mapv = self.get_local(map)?;
        let keyv = self.get_local(key)?;
        let resv = self.call(func, &mut [mapv, keyv]);
        self.bind_val(dst, resv);
        Ok(())
    }

    unsafe fn delete_map(&mut self, map: (NumTy, Ty), key: (NumTy, Ty)) -> Result<()> {
        assert_eq!(map.1.key()?, key.1);
        use Ty::*;
        let func = match map.1 {
            MapIntInt => "delete_intint",
            MapIntFloat => "delete_intfloat",
            MapIntStr => "delete_intstr",
            MapStrInt => "delete_strint",
            MapStrFloat => "delete_strfloat",
            MapStrStr => "delete_strstr",
            _ => unreachable!(),
        };
        let mapv = self.get_local(map)?;
        let keyv = self.get_local(key)?;
        self.call(func, &mut [mapv, keyv]);
        Ok(())
    }

    unsafe fn contains_map(
        &mut self,
        map: (NumTy, Ty),
        key: (NumTy, Ty),
        dst: (NumTy, Ty),
    ) -> Result<()> {
        assert_eq!(map.1.key()?, key.1);
        use Ty::*;
        let func = match map.1 {
            MapIntInt => "contains_intint",
            MapIntFloat => "contains_intfloat",
            MapIntStr => "contains_intstr",
            MapStrInt => "contains_strint",
            MapStrFloat => "contains_strfloat",
            MapStrStr => "contains_strstr",
            _ => unreachable!(),
        };
        let mapv = self.get_local(map)?;
        let keyv = self.get_local(key)?;
        let resv = self.call(func, &mut [mapv, keyv]);
        self.bind_val(dst, resv);
        Ok(())
    }

    unsafe fn len_map(&mut self, map: (NumTy, Ty), dst: (NumTy, Ty)) -> Result<()> {
        use Ty::*;
        let func = match map.1 {
            MapIntInt => "len_intint",
            MapIntFloat => "len_intfloat",
            MapIntStr => "len_intstr",
            MapStrInt => "len_strint",
            MapStrFloat => "len_strfloat",
            MapStrStr => "len_strstr",
            _ => unreachable!(),
        };
        let mapv = self.get_local(map)?;
        let resv = self.call(func, &mut [mapv]);
        self.bind_val(dst, resv);
        Ok(())
    }

    unsafe fn store_map(
        &mut self,
        map: (NumTy, Ty),
        key: (NumTy, Ty),
        val: (NumTy, Ty),
    ) -> Result<()> {
        assert_eq!(map.1.key()?, key.1);
        assert_eq!(map.1.val()?, val.1);
        use Ty::*;
        let func = match map.1 {
            MapIntInt => "insert_intint",
            MapIntFloat => "insert_intfloat",
            MapIntStr => "insert_intstr",
            MapStrInt => "insert_strint",
            MapStrFloat => "insert_strfloat",
            MapStrStr => "insert_strstr",
            _ => unreachable!(),
        };
        let mapv = self.get_local(map)?;
        let keyv = self.get_local(key)?;
        let valv = self.get_local(val)?;
        self.call(func, &mut [mapv, keyv, valv]);
        Ok(())
    }

    unsafe fn runtime_val(&self) -> LLVMValueRef {
        LLVMGetParam(
            self.f.val,
            self.decls[self.f.id].num_args as libc::c_uint - 1,
        )
    }

    unsafe fn cmp(
        &mut self,
        pred: Either<Pred, FPred>,
        l: LLVMValueRef,
        r: LLVMValueRef,
    ) -> LLVMValueRef {
        let res = match pred {
            Either::Left(ipred) => LLVMBuildICmp(self.f.builder, ipred, l, r, c_str!("")),
            Either::Right(fpred) => LLVMBuildFCmp(self.f.builder, fpred, l, r, c_str!("")),
        };
        // Comparisons return an `i1`; we need to zero-extend it back to an integer.
        // This means we'll have a good amount of 'zext's followed by 'trunc's, but those should
        // be both (a) cheap and (b) easy to optimize.
        let int_ty = self.tmap.get_ty(Ty::Int);
        LLVMBuildZExt(self.f.builder, res, int_ty, c_str!(""))
    }

    unsafe fn branch(
        &mut self,
        tcase: Option<(u32, LLVMBasicBlockRef)>,
        fcase: LLVMBasicBlockRef,
    ) -> Result<()> {
        if let Some((reg, t_bb)) = tcase {
            let val = self.get_local((reg, Ty::Int))?;
            let int_ty = self.tmap.get_ty(Ty::Int);
            let val_bool = LLVMBuildICmp(
                self.f.builder,
                Pred::LLVMIntNE,
                val,
                LLVMConstInt(int_ty, 0, /*sign_extend=*/ 1),
                c_str!(""),
            );
            LLVMBuildCondBr(self.f.builder, val_bool, t_bb, fcase);
        } else {
            LLVMBuildBr(self.f.builder, fcase);
        }
        Ok(())
    }

    unsafe fn gen_ll_inst<'b>(&mut self, inst: &compile::LL<'b>) -> Result<()> {
        use crate::bytecode::Instr::*;
        match inst {
            StoreConstStr(sr, s) => {
                // We don't know where we're storing this string literal. If it's in the middle of
                // a loop, we could be calling drop on it repeatedly. If the string is boxed, that
                // will lead to double-frees. In our current setup, these literals will all be
                // either empty, or references to word-aligned arena-allocated strings, so that's
                // actually fine.
                assert!(s.drop_is_trivial());
                let sc = s.clone().into_bits();
                // There is no way to pass a 128-bit integer to LLVM directly. We have to convert
                // it to a string first.
                let as_hex = CString::new(format!("{:x}", sc)).unwrap();
                let ty = self.tmap.get_ty(Ty::Str);
                let v = LLVMConstIntOfString(ty, as_hex.as_ptr(), /*radix=*/ 16);
                self.bind_reg(sr, v);
            }
            StoreConstInt(ir, i) => {
                let (reg, cty) = ir.reflect();
                let ty = self.tmap.get_ty(cty);
                let v = LLVMConstInt(ty, *i as u64, /*sign_extend=*/ 1);
                self.bind_val((reg, cty), v);
            }
            StoreConstFloat(fr, f) => {
                let (reg, cty) = fr.reflect();
                let ty = self.tmap.get_ty(cty);
                let v = LLVMConstReal(ty, *f);
                self.bind_val((reg, cty), v);
            }
            IntToStr(sr, ir) => {
                let arg = self.get_local(ir.reflect())?;
                let res = self.call("int_to_str", &mut [arg]);
                self.bind_reg(sr, res);
            }
            FloatToStr(sr, fr) => {
                let arg = self.get_local(fr.reflect())?;
                let res = self.call("float_to_str", &mut [arg]);
                self.bind_reg(sr, res);
            }
            StrToInt(ir, sr) => {
                let str_ref = self.get_local(sr.reflect())?;
                let res = self.call("str_to_int", &mut [str_ref]);
                self.bind_reg(ir, res);
            }
            StrToFloat(fr, sr) => {
                let str_ref = self.get_local(sr.reflect())?;
                let res = self.call("str_to_float", &mut [str_ref]);
                self.bind_reg(fr, res);
            }
            FloatToInt(ir, fr) => {
                let fv = self.get_local(fr.reflect())?;
                let dst_ty = self.tmap.get_ty(Ty::Int);
                let res = LLVMBuildFPToSI(self.f.builder, fv, dst_ty, c_str!(""));
                self.bind_reg(ir, res);
            }
            IntToFloat(fr, ir) => {
                let iv = self.get_local(ir.reflect())?;
                let dst_ty = self.tmap.get_ty(Ty::Float);
                let res = LLVMBuildSIToFP(self.f.builder, iv, dst_ty, c_str!(""));
                self.bind_reg(fr, res);
            }
            AddInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let addv = LLVMBuildAdd(self.f.builder, lv, rv, c_str!(""));
                self.bind_reg(res, addv);
            }
            AddFloat(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let addv = LLVMBuildFAdd(self.f.builder, lv, rv, c_str!(""));
                self.bind_reg(res, addv);
            }
            MulInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let addv = LLVMBuildMul(self.f.builder, lv, rv, c_str!(""));
                self.bind_reg(res, addv);
            }
            MulFloat(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let addv = LLVMBuildFMul(self.f.builder, lv, rv, c_str!(""));
                self.bind_reg(res, addv);
            }
            MinusInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let addv = LLVMBuildSub(self.f.builder, lv, rv, c_str!(""));
                self.bind_reg(res, addv);
            }
            MinusFloat(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let addv = LLVMBuildFSub(self.f.builder, lv, rv, c_str!(""));
                self.bind_reg(res, addv);
            }
            ModInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let addv = LLVMBuildSRem(self.f.builder, lv, rv, c_str!(""));
                self.bind_reg(res, addv);
            }
            ModFloat(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let addv = LLVMBuildFRem(self.f.builder, lv, rv, c_str!(""));
                self.bind_reg(res, addv);
            }
            Div(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let addv = LLVMBuildFDiv(self.f.builder, lv, rv, c_str!(""));
                self.bind_reg(res, addv);
            }
            Not(res, ir) => {
                let operand = self.get_local(ir.reflect())?;
                let ty = self.tmap.get_ty(Ty::Int);
                let zero = LLVMConstInt(ty, 0, /*sign_extend=*/ 1);
                let cmp = self.cmp(Either::Left(Pred::LLVMIntEQ), operand, zero);
                self.bind_reg(res, cmp);
            }
            NotStr(res, sr) => {
                let mut sv = self.get_local(sr.reflect())?;
                let strlen = self.intrinsics.get("str_len");
                let lenv = LLVMBuildCall(self.f.builder, strlen, &mut sv, 1, c_str!(""));
                let ty = self.tmap.get_ty(Ty::Int);
                let zero = LLVMConstInt(ty, 0, /*sign_extend=*/ 1);
                let cmp = self.cmp(Either::Left(Pred::LLVMIntEQ), lenv, zero);
                self.bind_reg(res, cmp);
            }
            NegInt(res, ir) => {
                let operand = self.get_local(ir.reflect())?;
                let ty = self.tmap.get_ty(Ty::Int);
                let zero = LLVMConstInt(ty, 0, /*sign_extend=*/ 1);
                let neg = LLVMBuildSub(self.f.builder, zero, operand, c_str!(""));
                self.bind_reg(res, neg);
            }
            NegFloat(res, fr) => {
                let operand = self.get_local(fr.reflect())?;
                let neg = LLVMBuildFNeg(self.f.builder, operand, c_str!(""));
                self.bind_reg(res, neg);
            }
            Concat(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let resv = self.call("concat", &mut [lv, rv]);
                self.bind_reg(res, resv);
            }
            Match(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let rt = self.runtime_val();
                let resv = self.call("match_pat_loc", &mut [rt, lv, rv]);
                self.bind_reg(res, resv);
            }
            IsMatch(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let rt = self.runtime_val();
                let resv = self.call("match_pat", &mut [rt, lv, rv]);
                self.bind_reg(res, resv);
            }
            LenStr(res, s) => {
                let sv = self.get_local(s.reflect())?;
                let lenv = self.call("str_len", &mut [sv]);
                self.bind_reg(res, lenv);
            }
            Sub(res, pat, s, in_s) => {
                let patv = self.get_local(pat.reflect())?;
                let sv = self.get_local(s.reflect())?;
                let in_sv = self.get_local(in_s.reflect())?;
                let rt = self.runtime_val();
                let resv = self.call("subst_first", &mut [rt, patv, sv, in_sv]);
                self.bind_reg(res, resv);
            }
            GSub(res, pat, s, in_s) => {
                let patv = self.get_local(pat.reflect())?;
                let sv = self.get_local(s.reflect())?;
                let in_sv = self.get_local(in_s.reflect())?;
                let rt = self.runtime_val();
                let resv = self.call("subst_all", &mut [rt, patv, sv, in_sv]);
                self.bind_reg(res, resv);
            }
            Substr(res, base, l, r) => {
                let basev = self.get_local(base.reflect())?;
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let resv = self.call("substr", &mut [basev, lv, rv]);
                self.bind_reg(res, resv);
            }
            LTFloat(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = self.cmp(Either::Right(FPred::LLVMRealOLT), lv, rv);
                self.bind_reg(res, ltv);
            }
            LTInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = self.cmp(Either::Left(Pred::LLVMIntSLT), lv, rv);
                self.bind_reg(res, ltv);
            }
            LTStr(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let resv = self.call("str_lt", &mut [lv, rv]);
                self.bind_reg(res, resv);
            }
            GTFloat(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = self.cmp(Either::Right(FPred::LLVMRealOGT), lv, rv);
                self.bind_reg(res, ltv);
            }
            GTInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = self.cmp(Either::Left(Pred::LLVMIntSGT), lv, rv);
                self.bind_reg(res, ltv);
            }
            GTStr(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let resv = self.call("str_gt", &mut [lv, rv]);
                self.bind_reg(res, resv);
            }
            LTEFloat(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = self.cmp(Either::Right(FPred::LLVMRealOLE), lv, rv);
                self.bind_reg(res, ltv);
            }
            LTEInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = self.cmp(Either::Left(Pred::LLVMIntSLE), lv, rv);
                self.bind_reg(res, ltv);
            }
            LTEStr(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let resv = self.call("str_lte", &mut [lv, rv]);
                self.bind_reg(res, resv);
            }
            GTEFloat(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = self.cmp(Either::Right(FPred::LLVMRealOGE), lv, rv);
                self.bind_reg(res, ltv);
            }
            GTEInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = self.cmp(Either::Left(Pred::LLVMIntSGE), lv, rv);
                self.bind_reg(res, ltv);
            }
            GTEStr(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let resv = self.call("str_gte", &mut [lv, rv]);
                self.bind_reg(res, resv);
            }
            EQFloat(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = self.cmp(Either::Right(FPred::LLVMRealOEQ), lv, rv);
                self.bind_reg(res, ltv);
            }
            EQInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = self.cmp(Either::Left(Pred::LLVMIntEQ), lv, rv);
                self.bind_reg(res, ltv);
            }
            EQStr(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let resv = self.call("str_eq", &mut [lv, rv]);
                self.bind_reg(res, resv);
            }
            SetColumn(dst, src) => {
                let dv = self.get_local(dst.reflect())?;
                let sv = self.get_local(src.reflect())?;
                self.call("set_col", &mut [self.runtime_val(), dv, sv]);
            }
            GetColumn(dst, src) => {
                let sv = self.get_local(src.reflect())?;
                let resv = self.call("get_col", &mut [self.runtime_val(), sv]);
                self.bind_reg(dst, resv);
            }
            SplitInt(flds, to_split, arr, pat) => {
                let rt = self.runtime_val();
                let tsv = self.get_local(to_split.reflect())?;
                let arrv = self.get_local(arr.reflect())?;
                let patv = self.get_local(pat.reflect())?;
                let resv = self.call("split_int", &mut [rt, tsv, arrv, patv]);
                self.bind_reg(flds, resv);
            }
            SplitStr(flds, to_split, arr, pat) => {
                let rt = self.runtime_val();
                let tsv = self.get_local(to_split.reflect())?;
                let arrv = self.get_local(arr.reflect())?;
                let patv = self.get_local(pat.reflect())?;
                let resv = self.call("split_str", &mut [rt, tsv, arrv, patv]);
                self.bind_reg(flds, resv);
            }
            Sprintf { dst, fmt, args } => {
                let arg_tys: SmallVec<_> = args.iter().map(|x| x.1).collect();
                let sprintf_fn = self.wrapped_printf((arg_tys, PrintfKind::Sprintf));
                let mut arg_vs = SmallVec::with_capacity(args.len() + 1);
                arg_vs.push(self.get_local(fmt.reflect())?);
                for a in args.iter().cloned() {
                    arg_vs.push(self.get_local(a)?);
                }
                let resv = LLVMBuildCall(
                    self.f.builder,
                    sprintf_fn,
                    arg_vs.as_mut_ptr(),
                    arg_vs.len() as libc::c_uint,
                    c_str!(""),
                );
                self.bind_reg(dst, resv);
            }
            Printf { output, fmt, args } => {
                // First, extract the types and use that to get a handle on a wrapped printf
                // function.
                let arg_tys: SmallVec<_> = args.iter().map(|x| x.1).collect();
                let printf_fn = self.wrapped_printf((
                    arg_tys,
                    if output.is_some() {
                        PrintfKind::File
                    } else {
                        PrintfKind::Stdout
                    },
                ));
                let mut arg_vs = SmallVec::with_capacity(if output.is_some() {
                    args.len() + 4
                } else {
                    args.len() + 2
                });
                arg_vs.push(self.runtime_val());
                arg_vs.push(self.get_local(fmt.reflect())?);
                for a in args.iter().cloned() {
                    arg_vs.push(self.get_local(a)?);
                }
                if let Some((path, append)) = output {
                    arg_vs.push(self.get_local(path.reflect())?);
                    let int_ty = self.tmap.get_ty(Ty::Int);
                    arg_vs.push(LLVMConstInt(int_ty, if *append { 1 } else { 0 }, 0));
                }
                LLVMBuildCall(
                    self.f.builder,
                    printf_fn,
                    arg_vs.as_mut_ptr(),
                    arg_vs.len() as libc::c_uint,
                    c_str!(""),
                );
            }
            PrintStdout(txt) => {
                let txtv = self.get_local(txt.reflect())?;
                self.call("print_stdout", &mut [self.runtime_val(), txtv]);
            }
            Close(file) => {
                let filev = self.get_local(file.reflect())?;
                self.call("close_file", &mut [self.runtime_val(), filev]);
            }
            Print(txt, out, append) => {
                let int_ty = self.tmap.get_ty(Ty::Int);
                let appv = LLVMConstInt(int_ty, *append as u64, /*sign_extend=*/ 1);
                let txtv = self.get_local(txt.reflect())?;
                let outv = self.get_local(out.reflect())?;
                self.call("print", &mut [self.runtime_val(), txtv, outv, appv]);
            }

            ReadErr(dst, file) => {
                let filev = self.get_local(file.reflect())?;
                let resv = self.call("read_err", &mut [self.runtime_val(), filev]);
                self.bind_reg(dst, resv);
            }
            NextLine(dst, file) => {
                let filev = self.get_local(file.reflect())?;
                let resv = self.call("next_line", &mut [self.runtime_val(), filev]);
                self.bind_reg(dst, resv);
            }
            ReadErrStdin(dst) => {
                let resv = self.call("read_err_stdin", &mut [self.runtime_val()]);
                self.bind_reg(dst, resv);
            }
            NextLineStdin(dst) => {
                let resv = self.call("next_line_stdin", &mut [self.runtime_val()]);
                self.bind_reg(dst, resv);
            }
            LookupIntInt(res, arr, k) => {
                self.lookup_map(arr.reflect(), k.reflect(), res.reflect())?
            }
            LookupIntStr(res, arr, k) => {
                self.lookup_map(arr.reflect(), k.reflect(), res.reflect())?
            }
            LookupIntFloat(res, arr, k) => {
                self.lookup_map(arr.reflect(), k.reflect(), res.reflect())?
            }
            LookupStrInt(res, arr, k) => {
                self.lookup_map(arr.reflect(), k.reflect(), res.reflect())?
            }
            LookupStrStr(res, arr, k) => {
                self.lookup_map(arr.reflect(), k.reflect(), res.reflect())?
            }
            LookupStrFloat(res, arr, k) => {
                self.lookup_map(arr.reflect(), k.reflect(), res.reflect())?
            }
            ContainsIntInt(res, arr, k) => {
                self.contains_map(arr.reflect(), k.reflect(), res.reflect())?
            }
            ContainsIntStr(res, arr, k) => {
                self.contains_map(arr.reflect(), k.reflect(), res.reflect())?
            }
            ContainsIntFloat(res, arr, k) => {
                self.contains_map(arr.reflect(), k.reflect(), res.reflect())?
            }
            ContainsStrInt(res, arr, k) => {
                self.contains_map(arr.reflect(), k.reflect(), res.reflect())?
            }
            ContainsStrStr(res, arr, k) => {
                self.contains_map(arr.reflect(), k.reflect(), res.reflect())?
            }
            ContainsStrFloat(res, arr, k) => {
                self.contains_map(arr.reflect(), k.reflect(), res.reflect())?
            }
            DeleteIntInt(arr, k) => self.delete_map(arr.reflect(), k.reflect())?,
            DeleteIntFloat(arr, k) => self.delete_map(arr.reflect(), k.reflect())?,
            DeleteIntStr(arr, k) => self.delete_map(arr.reflect(), k.reflect())?,
            DeleteStrInt(arr, k) => self.delete_map(arr.reflect(), k.reflect())?,
            DeleteStrFloat(arr, k) => self.delete_map(arr.reflect(), k.reflect())?,
            DeleteStrStr(arr, k) => self.delete_map(arr.reflect(), k.reflect())?,
            LenIntInt(res, arr) => self.len_map(arr.reflect(), res.reflect())?,
            LenIntFloat(res, arr) => self.len_map(arr.reflect(), res.reflect())?,
            LenIntStr(res, arr) => self.len_map(arr.reflect(), res.reflect())?,
            LenStrInt(res, arr) => self.len_map(arr.reflect(), res.reflect())?,
            LenStrFloat(res, arr) => self.len_map(arr.reflect(), res.reflect())?,
            LenStrStr(res, arr) => self.len_map(arr.reflect(), res.reflect())?,
            StoreIntInt(arr, k, v) => self.store_map(arr.reflect(), k.reflect(), v.reflect())?,
            StoreIntFloat(arr, k, v) => self.store_map(arr.reflect(), k.reflect(), v.reflect())?,
            StoreIntStr(arr, k, v) => self.store_map(arr.reflect(), k.reflect(), v.reflect())?,
            StoreStrInt(arr, k, v) => self.store_map(arr.reflect(), k.reflect(), v.reflect())?,
            StoreStrFloat(arr, k, v) => self.store_map(arr.reflect(), k.reflect(), v.reflect())?,
            StoreStrStr(arr, k, v) => self.store_map(arr.reflect(), k.reflect(), v.reflect())?,
            LoadVarStr(dst, var) => {
                let v = self.var_val(var);
                let res = self.call("load_var_str", &mut [self.runtime_val(), v]);
                let dreg = dst.reflect();
                self.bind_val(dreg, res);
                // The "load_var_" function refs the result for the common case that we are binding
                // the result to a local variable. If we are storing it directly into a global,
                // then bind_val would have already reffed it, so we decrement the count again.
                //
                // NB: We could do this as an extra parameter to the intrinsics. This makes the
                // code a bit cleaner, but it's worth revisiting in the future.
                if self.is_global(dreg) {
                    self.drop_reg(dreg)?;
                }
            }
            StoreVarStr(var, src) => {
                let v = self.var_val(var);
                let sv = self.get_local(src.reflect())?;
                self.call("store_var_str", &mut [self.runtime_val(), v, sv]);
            }
            LoadVarInt(dst, var) => {
                let v = self.var_val(var);
                let res = self.call("load_var_int", &mut [self.runtime_val(), v]);
                self.bind_reg(dst, res);
            }
            StoreVarInt(var, src) => {
                let v = self.var_val(var);
                let sv = self.get_local(src.reflect())?;
                self.call("store_var_int", &mut [self.runtime_val(), v, sv]);
            }
            LoadVarIntMap(dst, var) => {
                let v = self.var_val(var);
                let res = self.call("load_var_intmap", &mut [self.runtime_val(), v]);
                // See the comment in the LoadVarStr case.
                let dreg = dst.reflect();
                self.bind_val(dreg, res);
                if self.is_global(dreg) {
                    self.drop_reg(dreg)?;
                }
            }
            StoreVarIntMap(var, src) => {
                let v = self.var_val(var);
                let sv = self.get_local(src.reflect())?;
                self.call("store_var_intmap", &mut [self.runtime_val(), v, sv]);
            }
            MovInt(dst, src) => self.bind_reg(dst, self.get_local(src.reflect())?),
            MovFloat(dst, src) => self.bind_reg(dst, self.get_local(src.reflect())?),
            MovStr(dst, src) => {
                let sv = self.get_local(src.reflect())?;
                self.call("ref_str", &mut [sv]);
                let loaded = LLVMBuildLoad(self.f.builder, sv, c_str!(""));
                self.bind_reg(dst, loaded);
            }
            MovMapIntInt(dst, src) => self.bind_reg(dst, self.get_local(src.reflect())?),
            MovMapIntFloat(dst, src) => self.bind_reg(dst, self.get_local(src.reflect())?),
            MovMapIntStr(dst, src) => self.bind_reg(dst, self.get_local(src.reflect())?),
            MovMapStrInt(dst, src) => self.bind_reg(dst, self.get_local(src.reflect())?),
            MovMapStrFloat(dst, src) => self.bind_reg(dst, self.get_local(src.reflect())?),
            MovMapStrStr(dst, src) => self.bind_reg(dst, self.get_local(src.reflect())?),
            IterBeginIntInt(dst, arr) => self.iter_begin(dst.reflect(), arr.reflect())?,
            IterBeginIntFloat(dst, arr) => self.iter_begin(dst.reflect(), arr.reflect())?,
            IterBeginIntStr(dst, arr) => self.iter_begin(dst.reflect(), arr.reflect())?,
            IterBeginStrInt(dst, arr) => self.iter_begin(dst.reflect(), arr.reflect())?,
            IterBeginStrFloat(dst, arr) => self.iter_begin(dst.reflect(), arr.reflect())?,
            IterBeginStrStr(dst, arr) => self.iter_begin(dst.reflect(), arr.reflect())?,
            IterHasNextInt(dst, iter) => self.iter_hasnext(iter.reflect(), dst.reflect())?,
            IterHasNextStr(dst, iter) => self.iter_hasnext(iter.reflect(), dst.reflect())?,
            IterGetNextInt(dst, iter) => self.iter_getnext(iter.reflect(), dst.reflect())?,
            IterGetNextStr(dst, iter) => self.iter_getnext(iter.reflect(), dst.reflect())?,

            PushInt(_) | PushFloat(_) | PushStr(_) | PushIntInt(_) | PushIntFloat(_)
            | PushIntStr(_) | PushStrInt(_) | PushStrFloat(_) | PushStrStr(_) | PopInt(_)
            | PopFloat(_) | PopStr(_) | PopIntInt(_) | PopIntFloat(_) | PopIntStr(_)
            | PopStrInt(_) | PopStrFloat(_) | PopStrStr(_) => {
                return err!("unexpected explicit push/pop in llvm")
            }
            Ret | Halt | Jmp(_) | JmpIf(_, _) | Call(_) => {
                return err!("unexpected bytecode-level control flow")
            }
        };
        Ok(())
    }

    unsafe fn ret(&mut self, val: (NumTy, Ty)) -> Result<()> {
        self.ret_val(self.get_local(val)?, val.1)
    }

    unsafe fn ret_val(&mut self, to_return: LLVMValueRef, ty: Ty) -> Result<()> {
        let locals = mem::replace(&mut self.f.locals, Default::default());
        for ((reg, ty), llval) in locals.iter() {
            let (reg, ty) = (*reg, *ty);
            if self.f.skip_drop.contains(&(reg, ty)) || llval == &to_return {
                continue;
            }
            self.drop_val(*llval, ty)?;
        }
        if let Ty::Str = ty {
            let loaded = LLVMBuildLoad(self.f.builder, to_return, c_str!(""));
            LLVMBuildRet(self.f.builder, loaded);
        } else {
            LLVMBuildRet(self.f.builder, to_return);
        }
        let _old_locals = mem::replace(&mut self.f.locals, locals);
        debug_assert_eq!(_old_locals.len(), 0);
        Ok(())
    }
    unsafe fn gen_hl_inst(&mut self, inst: &compile::HighLevel) -> Result<()> {
        use compile::HighLevel::*;
        match inst {
            Call {
                func_id,
                dst_reg,
                dst_ty,
                args,
            } => {
                let source = &self.decls[self.f.id];
                let target = &self.decls[*func_id as usize];
                // Allocate room for and insert regular params, globals, and the runtime.
                let mut argvs: SmallVec<LLVMValueRef> =
                    smallvec![ptr::null_mut(); args.len() + target.globals.len() + 1];
                for (i, arg) in args.iter().cloned().enumerate() {
                    argvs[i] = self.get_local(arg)?;
                }
                for (global, ix) in target.globals.iter() {
                    let cur_ix = source
                        .globals
                        .get(global)
                        .cloned()
                        .expect("callee must have all globals");
                    argvs[*ix] = LLVMGetParam(self.f.val, cur_ix as libc::c_uint);
                }
                let rt_ix = argvs.len() - 1;
                debug_assert_eq!(rt_ix + 1, target.num_args);
                argvs[rt_ix] = self.runtime_val();
                let resv = LLVMBuildCall(
                    self.f.builder,
                    target.val,
                    argvs.as_mut_ptr(),
                    argvs.len() as libc::c_uint,
                    c_str!(""),
                );
                self.bind_val((*dst_reg, *dst_ty), resv);
            }
            Phi(reg, ty, _preds) => {
                self.f.skip_drop.insert((*reg, *ty));
                let res = LLVMBuildPhi(
                    self.f.builder,
                    if ty == &Ty::Str {
                        self.tmap.get_ptr_ty(*ty)
                    } else {
                        self.tmap.get_ty(*ty)
                    },
                    c_str!(""),
                );
                // NB why not `self.bind_val((*reg, *ty), res);` ?
                // Phis are always local, so most of the special handling can be skipped. This also
                // allows us to avoid extra refs and drops for phi nodes containing strings.
                self.f.locals.insert((*reg, *ty), res);
            }
            // Returns are handled elsewhere
            Ret(_reg, _ty) => {}
            DropIter(reg, ty) => {
                let drop_fn = match ty {
                    Ty::IterInt => "drop_iter_int",
                    Ty::IterStr => "drop_iter_str",
                    _ => return err!("can only drop iterators, got {:?}", ty),
                };
                let IterState { iter_ptr, len, .. } = self.get_iter((*reg, *ty))?.clone();
                self.call(drop_fn, &mut [iter_ptr, len]);
            }
        };
        Ok(())
    }

    // The printf function has variable arity. In addition to being a bit tricky to get right,
    // C-style var args are unstable in Rust. To get around this problem, we generate a stub method
    // for each invocation type which packages an array of types and argument pointers on the stack
    // and passes it into the runtime.
    //
    // We could implement this all inline, but making it a separate function allows us to cache the
    // codegen across compatible invocations, and also makes the generated code a lot cleaner.
    unsafe fn wrapped_printf(&mut self, key: (SmallVec<Ty>, PrintfKind)) -> LLVMValueRef {
        use std::io::{Cursor, Write};
        use PrintfKind::*;
        let kind = key.1;
        if let Some(v) = self.printfs.get(&key) {
            return *v;
        }
        let args = &key.0[..];

        let ix = self.printfs.len();
        let name = "_pf";
        // 64 bit integers should only ever need 20 digits or so.
        let mut name_c: smallvec::SmallVec<[u8; 32]> = smallvec![0; 32];
        for (i, b) in name.as_bytes().iter().enumerate() {
            name_c[i] = *b;
        }
        let mut w = Cursor::new(&mut name_c[name.as_bytes().len()..]);
        write!(w, "{:x}", ix).unwrap();
        assert_eq!(name_c[name_c.len() - 1], 0);

        // The var-arg portion + runtime + format spec
        //  (+ output + append, if named_output)
        let mut arg_lltys = smallvec::SmallVec::<[_; 8]>::with_capacity(args.len() + 4);
        match kind {
            File | Stdout => {
                arg_lltys.push(self.tmap.runtime_ty);
            }
            Sprintf => {}
        };
        arg_lltys.push(self.tmap.get_ptr_ty(Ty::Str)); // spec
        arg_lltys.extend(args.iter().cloned().map(|ty| {
            if ty == Ty::Str {
                self.tmap.get_ptr_ty(ty)
            } else {
                self.tmap.get_ty(ty)
            }
        }));
        if let File = kind {
            arg_lltys.push(self.tmap.get_ptr_ty(Ty::Str)); // output
            arg_lltys.push(self.tmap.get_ty(Ty::Int)); // append
        }

        let ret = match kind {
            File | Stdout => LLVMVoidTypeInContext(self.ctx),
            Sprintf => self.tmap.get_ty(Ty::Str),
        };
        let func_ty = LLVMFunctionType(ret, arg_lltys.as_mut_ptr(), arg_lltys.len() as u32, 0);
        let builder = LLVMCreateBuilderInContext(self.ctx);
        let f = LLVMAddFunction(self.module, name_c.as_ptr() as *const libc::c_char, func_ty);
        let bb = LLVMAppendBasicBlockInContext(self.ctx, f, c_str!(""));
        LLVMPositionBuilderAtEnd(builder, bb);

        // Allocate an array of u32s and an array of void*s to pass the arguments and their types.
        let u32_ty = LLVMIntTypeInContext(self.ctx, 32);
        let usize_ty =
            LLVMIntTypeInContext(self.ctx, mem::size_of::<*mut ()>() as libc::c_uint * 8);
        let len = args.len() as libc::c_uint;
        let types_ty = LLVMArrayType(u32_ty, len);
        let args_ty = LLVMArrayType(usize_ty, len);
        let types_array = LLVMBuildAlloca(builder, types_ty, c_str!(""));
        let args_array = LLVMBuildAlloca(builder, args_ty, c_str!(""));
        let zero = LLVMConstInt(u32_ty, 0, /*sign_extend=*/ 0);

        for (i, t) in args.iter().cloned().enumerate() {
            let mut index = [zero, LLVMConstInt(u32_ty, i as u64, /*sign_extend=*/ 0)];

            // Store a u32 code representing the type into the current index.
            let ty_ptr = LLVMBuildGEP(builder, types_array, index.as_mut_ptr(), 2, c_str!(""));
            let tval = LLVMConstInt(u32_ty, t as u32 as u64, /*sign_extend=*/ 0);
            LLVMBuildStore(builder, tval, ty_ptr);

            let arg_ptr = LLVMBuildGEP(builder, args_array, index.as_mut_ptr(), 2, c_str!(""));
            // Translate `i` to the param of the generated function.
            let offset = match kind {
                // Format spec, runtime
                File | Stdout => 2,
                // Just the format spec
                Sprintf => 1,
            };
            let argval = LLVMGetParam(f, i as libc::c_uint + offset);
            // Cast the value to void*, then store it into the array.
            let cast_val = if let Ty::Str = t {
                LLVMBuildPtrToInt(builder, argval, usize_ty, c_str!(""))
            } else {
                LLVMBuildBitCast(builder, argval, usize_ty, c_str!(""))
            };
            LLVMBuildStore(builder, cast_val, arg_ptr);
        }
        let mut start_index = [zero, zero];
        let args_ptr = LLVMBuildGEP(builder, args_array, start_index.as_mut_ptr(), 2, c_str!(""));
        let tys_ptr = LLVMBuildGEP(
            builder,
            types_array,
            start_index.as_mut_ptr(),
            2,
            c_str!(""),
        );
        let len_v = LLVMConstInt(
            self.tmap.get_ty(Ty::Int),
            len as u64,
            /*sign_extend=*/ 0,
        );
        match kind {
            File => {
                let intrinsic = self.intrinsics.get("printf_impl_file");
                // runtime, spec, args, tys, num_args, output, append
                let mut args = [
                    LLVMGetParam(f, 0),
                    LLVMGetParam(f, 1),
                    args_ptr,
                    tys_ptr,
                    len_v,
                    LLVMGetParam(f, len - 2),
                    LLVMGetParam(f, len - 1),
                ];
                LLVMBuildCall(
                    builder,
                    intrinsic,
                    args.as_mut_ptr(),
                    args.len() as libc::c_uint,
                    c_str!(""),
                );
                LLVMBuildRetVoid(builder);
            }
            Stdout => {
                let intrinsic = self.intrinsics.get("printf_impl_stdout");
                // runtime, spec, args, tys, num_args
                let mut args = [
                    LLVMGetParam(f, 0),
                    LLVMGetParam(f, 1),
                    args_ptr,
                    tys_ptr,
                    len_v,
                ];
                LLVMBuildCall(
                    builder,
                    intrinsic,
                    args.as_mut_ptr(),
                    args.len() as libc::c_uint,
                    c_str!(""),
                );
                LLVMBuildRetVoid(builder);
            }
            Sprintf => {
                let intrinsic = self.intrinsics.get("sprintf_impl");
                let mut args = [LLVMGetParam(f, 0), args_ptr, tys_ptr, len_v];
                let resv = LLVMBuildCall(
                    builder,
                    intrinsic,
                    args.as_mut_ptr(),
                    args.len() as libc::c_uint,
                    c_str!(""),
                );
                LLVMBuildRet(builder, resv);
            }
        }
        LLVMSetLinkage(f, llvm_sys::LLVMLinkage::LLVMLinkerPrivateLinkage);
        LLVMDisposeBuilder(builder);
        self.printfs.insert(key, f);
        f
    }
}
