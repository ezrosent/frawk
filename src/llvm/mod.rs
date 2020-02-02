use crate::bytecode::Accum;
use crate::common::{raw_guard, Either, NumTy, Result};
use crate::compile::{self, Ty, Typer};
use crate::libc::c_char;
use crate::llvm_sys as llvm;
use crate::runtime;
use llvm::{
    analysis::{LLVMVerifierFailureAction, LLVMVerifyModule},
    core::*,
    execution_engine::*,
    prelude::*,
    target::*,
    LLVMLinkage,
};

use hashbrown::HashMap;

pub mod intrinsics;

use std::ffi::{CStr, CString};
use std::mem::{self, MaybeUninit};
use std::ptr;

type Pred = llvm::LLVMIntPredicate;
type FPred = llvm::LLVMRealPredicate;

type SmallVec<T> = smallvec::SmallVec<[T; 2]>;

// TODO make sure we use this function in main, otherwise linker may decide to get rid of it.
#[no_mangle]
pub extern "C" fn __test_print() {
    println!("hello! this is rust code called from llvm");
}

struct Function {
    // TODO consider dropping `name`. Unclear if we need it. LLVM seems to take ownership, so we
    // might be able to give the memory back at construction time (or share a single string and
    // avoid the allocations).
    name: CString,
    val: LLVMValueRef,
    builder: LLVMBuilderRef,
    globals: HashMap<(NumTy, Ty), usize>,
    locals: HashMap<(NumTy, Ty), LLVMValueRef>,
    num_args: usize,
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

struct TypeMap {
    table: [TypeRef; compile::NUM_TYPES],
    runtime_ty: LLVMTypeRef,
}

impl TypeMap {
    fn new(ctx: LLVMContextRef) -> TypeMap {
        unsafe {
            TypeMap {
                table: [TypeRef::null(); compile::NUM_TYPES],
                runtime_ty: LLVMPointerType(LLVMVoidTypeInContext(ctx), 0),
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

struct Generator<'a, 'b> {
    types: &'b mut Typer<'a>,
    ctx: LLVMContextRef,
    module: LLVMModuleRef,
    engine: LLVMExecutionEngineRef,
    pass_manager: LLVMPassManagerRef,
    decls: Vec<Function>,
    type_map: TypeMap,
    intrinsics: HashMap<&'static str, LLVMValueRef>,
}

impl<'a, 'b> Drop for Generator<'a, 'b> {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeModule(self.module);
            LLVMDisposePassManager(self.pass_manager);
        }
    }
}

impl<'a, 'b> Generator<'a, 'b> {
    pub unsafe fn init(types: &'b mut Typer<'a>) -> Result<Generator<'a, 'b>> {
        if llvm::support::LLVMLoadLibraryPermanently(ptr::null()) != 0 {
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
        let pass_manager = LLVMCreateFunctionPassManagerForModule(module);
        {
            use llvm::transforms::scalar::*;
            llvm::transforms::util::LLVMAddPromoteMemoryToRegisterPass(pass_manager);
            LLVMAddConstantPropagationPass(pass_manager);
            LLVMAddInstructionCombiningPass(pass_manager);
            LLVMAddReassociatePass(pass_manager);
            LLVMAddGVNPass(pass_manager);
            LLVMAddCFGSimplificationPass(pass_manager);
            LLVMInitializeFunctionPassManager(pass_manager);
        }
        let nframes = types.frames.len();
        let mut res = Generator {
            types,
            ctx,
            module,
            engine,
            pass_manager,
            decls: Vec::with_capacity(nframes),
            type_map: TypeMap::new(ctx),
            intrinsics: intrinsics::register(module, ctx),
        };
        res.build_map();
        res.build_decls();
        Ok(res)
    }

    unsafe fn build_map(&mut self) {
        use mem::size_of;
        // TODO: make this a void* instead?
        let make = |ty| TypeRef {
            base: ty,
            ptr: LLVMPointerType(ty, 0),
        };
        let uintptr = LLVMIntTypeInContext(self.ctx, (size_of::<usize>() * 8) as libc::c_uint);
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
        self.type_map.init(Ty::MapIntInt, make(uintptr));
        self.type_map.init(Ty::MapIntFloat, make(uintptr));
        self.type_map.init(Ty::MapIntStr, make(uintptr));
        self.type_map.init(Ty::MapStrInt, make(uintptr));
        self.type_map.init(Ty::MapStrFloat, make(uintptr));
        self.type_map.init(Ty::MapStrStr, make(uintptr));
        // TODO: handle iterators.
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

    // TODO: get a big loop together for translating basic blocks
    //       - skip phis until next stage
    //       - allocate the relevant locals
    //       - for main, allocate globals.
    //       - Get scalar instructions done, but stub out runtime for now.
    // TODO: control flow
    //       - read up on this more, get the instructions down. structure should follow bytecode
    //       translation closely.
    // TODO: runtime
    //       - fill in runtime, figure out where to allocate it in main. It probably needs to be
    //       passed as a a param to every function (if so, just make it the last? or first?).

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
            let name = CString::new(if i == self.types.main_offset {
                format!("_frawk_main")
            } else {
                format!("_frawk_udf_{}", i)
            })
            .unwrap();
            // First, we add the listed function parameters.
            arg_tys.extend(info.arg_tys.iter().map(|ty| self.llvm_ty(*ty)));
            // Then, we add on the referenced globals.
            for (reg, ty) in refs.iter().cloned() {
                let ix = arg_tys.len();
                arg_tys.push(self.llvm_ptr_ty(ty));
                globals.insert((reg, ty), ix);
            }
            // Finally, we add a pointer to the runtime; always the last parameter.
            arg_tys.push(self.type_map.runtime_ty);
            let ty = LLVMFunctionType(
                self.llvm_ty(info.ret_ty),
                arg_tys.as_mut_ptr(),
                arg_tys.len() as u32,
                /*IsVarArg=*/ 0,
            );
            let val = LLVMAddFunction(self.module, name.as_ptr(), ty);
            let builder = LLVMCreateBuilderInContext(self.ctx);
            let block = LLVMAppendBasicBlockInContext(self.ctx, val, c_str!(""));
            LLVMPositionBuilderAtEnd(builder, block);
            self.decls.push(Function {
                name,
                val,
                builder,
                globals,
                locals: Default::default(),
                num_args: arg_tys.len(),
            });
            arg_tys.clear();
        }
    }

    unsafe fn alloc_local(
        &self,
        builder: LLVMBuilderRef,
        reg: NumTy,
        ty: Ty,
    ) -> Result<LLVMValueRef> {
        use Ty::*;
        let val = match ty {
            Int => LLVMConstInt(self.llvm_ty(Int), 0, /*sign_extend=*/ 1),
            Float => LLVMConstReal(self.llvm_ty(Float), 0.0),
            Str => {
                let str_ty = self.type_map.get_ty(Str);
                let v = LLVMConstInt(str_ty, 0, /*sign_extend=*/ 0);
                let v_loc = LLVMBuildAlloca(builder, str_ty, c_str!(""));
                LLVMBuildStore(builder, v, v_loc);
                v_loc
            }
            MapIntInt | MapIntStr | MapIntFloat | MapStrInt | MapStrStr | MapStrFloat => {
                LLVMConstInt(self.llvm_ty(ty), 0, /*sign_extend=*/ 0)
            }
            IterInt | IterStr => return err!("we should not be allocating any iterators"),
        };
        Ok(val)
    }

    unsafe fn gen_function(&mut self, func_id: usize) -> Result<()> {
        let frame = &self.types.frames[func_id];
        for (local, (reg, ty)) in frame.locals.iter() {
            debug_assert!(!local.global);
            // implicitly-declared locals are just the ones with a subscript of 0.
            if local.sub == 0 {
                let val = self.alloc_local(self.decls[func_id].builder, *reg, *ty)?;
                self.decls[func_id].locals.insert((*reg, *ty), val);
            }
        }
        let info = &self.types.func_info[func_id];
        for (i, bb) in frame.cfg.raw_nodes().iter().enumerate() {
            let decl = &mut self.decls[func_id];
            // TODO branches, etc.
            for inst in &bb.weight {
                match inst {
                    Either::Left(ll) => decl.gen_ll_inst(ll, &self.type_map, &self.intrinsics)?,
                    Either::Right(hl) => decl.gen_hl_inst(hl)?,
                }
            }
        }
        Ok(())
    }
}

impl Function {
    unsafe fn get_local(&self, local: (NumTy, Ty)) -> Result<LLVMValueRef> {
        if let Some(v) = self.locals.get(&local) {
            Ok(*v)
        } else if let Some(ix) = self.globals.get(&local) {
            let gv = LLVMGetParam(self.val, *ix as libc::c_uint);
            // TODO lifetimes, (similar to bind_val)
            unimplemented!()
        // Ok(if let Ty::Str = local.1 {
        //     // no point in loading the string directly. We manipulate them as pointers.
        //     // Or do we want to ref them?
        //     gv
        // } else {
        //     unimplemented!()
        // })
        } else {
            err!("unbound local variable {:?}", local)
        }
    }
    unsafe fn bind_val(&mut self, val: (NumTy, Ty), to: LLVMValueRef, tmap: &TypeMap) {
        // if val is global, then find the relevant parameter and store it directly.
        // if val is an existing local, fail
        // if val.ty is a string, alloca a new string, store it, then bind the result.
        // otherwise, just bind the result directly.
        if let Some(ix) = self.globals.get(&val) {
            // TODO Lifetimes.
            // We want to do the below, but issue the correct refs/drops where necessary.
            // (probably ref to, drop from,then store, for the relevant types).
            unimplemented!()
            // let param = LLVMGetParam(self.val, *ix as libc::c_uint);
            // LLVMBuildStore(self.builder, to, param);
            // return;
        }
        debug_assert!(self.locals.get(&val).is_none());
        if let Ty::Str = val.1 {
            let str_ty = tmap.get_ty(Ty::Str);
            let loc = LLVMBuildAlloca(self.builder, str_ty, c_str!(""));
            LLVMBuildStore(self.builder, to, loc);
            self.locals.insert(val, loc);
        } else {
            self.locals.insert(val, to);
        }
    }
    // TODO, pass in fields from Generator as needed.
    unsafe fn gen_ll_inst<'a>(
        &mut self,
        inst: &compile::LL<'a>,
        tmap: &TypeMap,
        intrinsics: &HashMap<&'static str, LLVMValueRef>,
    ) -> Result<()> {
        use crate::bytecode::Instr::*;
        match inst {
            StoreConstStr(sr, s) => {
                let sc = s.clone().into_bits();
                // There is no way to pass a 128-bit integer to LLVM directly. We have to convert
                // it to a string first.
                let as_hex = CString::new(format!("{:x}", sc)).unwrap();
                let ty = tmap.get_ty(Ty::Str);
                let v = LLVMConstIntOfString(ty, as_hex.as_ptr(), /*radix=*/ 16);
                self.bind_val(sr.reflect(), v, tmap);
            }
            StoreConstInt(ir, i) => {
                let (reg, cty) = ir.reflect();
                let ty = tmap.get_ty(cty);
                let v = LLVMConstInt(ty, *i as u64, /*sign_extend=*/ 1);
                self.bind_val((reg, cty), v, tmap);
            }
            StoreConstFloat(fr, f) => {
                let (reg, cty) = fr.reflect();
                let ty = tmap.get_ty(cty);
                let v = LLVMConstReal(ty, *f);
                self.bind_val((reg, cty), v, tmap);
            }
            IntToStr(sr, ir) => {
                let mut arg = self.get_local(ir.reflect())?;
                let conv = intrinsics["int_to_str"];
                let res = LLVMBuildCall(
                    self.builder,
                    conv,
                    &mut arg,
                    /*num_args=*/ 1,
                    c_str!(""),
                );
                self.bind_val(sr.reflect(), res, tmap);
            }
            FloatToStr(sr, fr) => {
                let mut arg = self.get_local(fr.reflect())?;
                let conv = intrinsics["float_to_str"];
                let res = LLVMBuildCall(
                    self.builder,
                    conv,
                    &mut arg,
                    /*num_args=*/ 1,
                    c_str!(""),
                );
                self.bind_val(sr.reflect(), res, tmap);
            }
            StrToInt(ir, sr) => {
                let mut str_ref = self.get_local(sr.reflect())?;
                let conv = intrinsics["str_to_int"];
                let res = LLVMBuildCall(
                    self.builder,
                    conv,
                    &mut str_ref,
                    /*num_args=*/ 1,
                    c_str!(""),
                );
                self.bind_val(ir.reflect(), res, tmap);
            }
            StrToFloat(fr, sr) => {
                let mut str_ref = self.get_local(sr.reflect())?;
                let conv = intrinsics["str_to_float"];
                let res = LLVMBuildCall(
                    self.builder,
                    conv,
                    &mut str_ref,
                    /*num_args=*/ 1,
                    c_str!(""),
                );
                self.bind_val(fr.reflect(), res, tmap);
            }
            FloatToInt(ir, fr) => {
                let fv = self.get_local(fr.reflect())?;
                let dst_ty = tmap.get_ty(Ty::Int);
                let res = LLVMBuildFPToSI(self.builder, fv, dst_ty, c_str!(""));
                self.bind_val(ir.reflect(), res, tmap);
            }
            IntToFloat(fr, ir) => {
                let iv = self.get_local(ir.reflect())?;
                let dst_ty = tmap.get_ty(Ty::Float);
                let res = LLVMBuildSIToFP(self.builder, iv, dst_ty, c_str!(""));
                self.bind_val(fr.reflect(), res, tmap);
            }
            AddInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let addv = LLVMBuildAdd(self.builder, lv, rv, c_str!(""));
                self.bind_val(res.reflect(), addv, tmap);
            }
            AddFloat(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let addv = LLVMBuildFAdd(self.builder, lv, rv, c_str!(""));
                self.bind_val(res.reflect(), addv, tmap);
            }
            MulInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let addv = LLVMBuildMul(self.builder, lv, rv, c_str!(""));
                self.bind_val(res.reflect(), addv, tmap);
            }
            MulFloat(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let addv = LLVMBuildFMul(self.builder, lv, rv, c_str!(""));
                self.bind_val(res.reflect(), addv, tmap);
            }
            MinusInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let addv = LLVMBuildSub(self.builder, lv, rv, c_str!(""));
                self.bind_val(res.reflect(), addv, tmap);
            }
            MinusFloat(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let addv = LLVMBuildFSub(self.builder, lv, rv, c_str!(""));
                self.bind_val(res.reflect(), addv, tmap);
            }
            ModInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let addv = LLVMBuildSRem(self.builder, lv, rv, c_str!(""));
                self.bind_val(res.reflect(), addv, tmap);
            }
            ModFloat(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let addv = LLVMBuildFRem(self.builder, lv, rv, c_str!(""));
                self.bind_val(res.reflect(), addv, tmap);
            }
            Div(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let addv = LLVMBuildFDiv(self.builder, lv, rv, c_str!(""));
                self.bind_val(res.reflect(), addv, tmap);
            }
            Not(res, ir) => {
                let operand = self.get_local(ir.reflect())?;
                let ty = tmap.get_ty(Ty::Int);
                let zero = LLVMConstInt(ty, 0, /*sign_extend=*/ 1);
                let cmp = LLVMBuildICmp(self.builder, Pred::LLVMIntEQ, operand, zero, c_str!(""));
                self.bind_val(res.reflect(), cmp, tmap);
            }
            NotStr(res, sr) => unimplemented!(),
            NegInt(res, ir) => {
                let operand = self.get_local(ir.reflect())?;
                let ty = tmap.get_ty(Ty::Int);
                let zero = LLVMConstInt(ty, 0, /*sign_extend=*/ 1);
                let neg = LLVMBuildSub(self.builder, zero, operand, c_str!(""));
                self.bind_val(res.reflect(), neg, tmap);
            }
            NegFloat(res, fr) => {
                let operand = self.get_local(fr.reflect())?;
                let neg = LLVMBuildFNeg(self.builder, operand, c_str!(""));
                self.bind_val(res.reflect(), neg, tmap);
            }
            Concat(res, l, r) => {
                let str_ty = tmap.get_ty(Ty::Str);
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let concat = intrinsics["concat"];
                let mut args = [lv, rv];
                let resv = LLVMBuildCall(
                    self.builder,
                    concat,
                    args.as_mut_ptr(),
                    /*num_args=*/ args.len() as libc::c_uint,
                    c_str!(""),
                );
                self.bind_val(res.reflect(), resv, tmap);
            }
            Match(res, l, r) => unimplemented!(),
            LenStr(res, s) => unimplemented!(),
            LTFloat(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = LLVMBuildFCmp(self.builder, FPred::LLVMRealOLT, lv, rv, c_str!(""));
                self.bind_val(res.reflect(), ltv, tmap);
            }
            LTInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = LLVMBuildICmp(self.builder, Pred::LLVMIntSLT, lv, rv, c_str!(""));
                self.bind_val(res.reflect(), ltv, tmap);
            }
            LTStr(res, l, r) => unimplemented!(),
            GTFloat(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = LLVMBuildFCmp(self.builder, FPred::LLVMRealOGT, lv, rv, c_str!(""));
                self.bind_val(res.reflect(), ltv, tmap);
            }
            GTInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = LLVMBuildICmp(self.builder, Pred::LLVMIntSGT, lv, rv, c_str!(""));
                self.bind_val(res.reflect(), ltv, tmap);
            }
            GTStr(res, l, r) => unimplemented!(),
            LTEFloat(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = LLVMBuildFCmp(self.builder, FPred::LLVMRealOLE, lv, rv, c_str!(""));
                self.bind_val(res.reflect(), ltv, tmap);
            }
            LTEInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = LLVMBuildICmp(self.builder, Pred::LLVMIntSLE, lv, rv, c_str!(""));
                self.bind_val(res.reflect(), ltv, tmap);
            }
            LTEStr(res, l, r) => unimplemented!(),
            GTEFloat(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = LLVMBuildFCmp(self.builder, FPred::LLVMRealOGE, lv, rv, c_str!(""));
                self.bind_val(res.reflect(), ltv, tmap);
            }
            GTEInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = LLVMBuildICmp(self.builder, Pred::LLVMIntSGE, lv, rv, c_str!(""));
                self.bind_val(res.reflect(), ltv, tmap);
            }
            GTEStr(res, l, r) => unimplemented!(),
            EQFloat(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = LLVMBuildFCmp(self.builder, FPred::LLVMRealOEQ, lv, rv, c_str!(""));
                self.bind_val(res.reflect(), ltv, tmap);
            }
            EQInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = LLVMBuildICmp(self.builder, Pred::LLVMIntEQ, lv, rv, c_str!(""));
                self.bind_val(res.reflect(), ltv, tmap);
            }
            EQStr(res, l, r) => unimplemented!(),
            SetColumn(dst, src) => unimplemented!(),
            GetColumn(dst, src) => unimplemented!(),
            SplitInt(flds, to_split, arr, pat) => unimplemented!(),
            SplitStr(flds, to_split, arr, pat) => unimplemented!(),
            PrintStdout(txt) => unimplemented!(),
            Print(txt, out, append) => unimplemented!(),
            LookupIntInt(res, arr, k) => unimplemented!(),
            LookupIntStr(res, arr, k) => unimplemented!(),
            LookupIntFloat(res, arr, k) => unimplemented!(),
            LookupStrInt(res, arr, k) => unimplemented!(),
            LookupStrStr(res, arr, k) => unimplemented!(),
            LookupStrFloat(res, arr, k) => unimplemented!(),
            ContainsIntInt(res, arr, k) => unimplemented!(),
            ContainsIntStr(res, arr, k) => unimplemented!(),
            ContainsIntFloat(res, arr, k) => unimplemented!(),
            ContainsStrInt(res, arr, k) => unimplemented!(),
            ContainsStrStr(res, arr, k) => unimplemented!(),
            ContainsStrFloat(res, arr, k) => unimplemented!(),
            DeleteIntInt(arr, k) => unimplemented!(),
            DeleteIntFloat(arr, k) => unimplemented!(),
            DeleteIntStr(arr, k) => unimplemented!(),
            DeleteStrInt(arr, k) => unimplemented!(),
            DeleteStrFloat(arr, k) => unimplemented!(),
            DeleteStrStr(arr, k) => unimplemented!(),
            LenIntInt(res, arr) => unimplemented!(),
            LenIntFloat(res, arr) => unimplemented!(),
            LenIntStr(res, arr) => unimplemented!(),
            LenStrInt(res, arr) => unimplemented!(),
            LenStrFloat(res, arr) => unimplemented!(),
            LenStrStr(res, arr) => unimplemented!(),
            StoreIntInt(arr, k, v) => unimplemented!(),
            StoreIntFloat(arr, k, v) => unimplemented!(),
            StoreIntStr(arr, k, v) => unimplemented!(),
            StoreStrInt(arr, k, v) => unimplemented!(),
            StoreStrFloat(arr, k, v) => unimplemented!(),
            StoreStrStr(arr, k, v) => unimplemented!(),
            LoadVarStr(dst, var) => unimplemented!(),
            StoreVarStr(var, src) => unimplemented!(),
            LoadVarInt(dst, var) => unimplemented!(),
            StoreVarInt(var, src) => unimplemented!(),
            LoadVarIntMap(dst, var) => unimplemented!(),
            StoreVarIntMap(var, src) => unimplemented!(),
            MovInt(dst, src) => unimplemented!(),
            MovFloat(dst, src) => unimplemented!(),
            MovStr(dst, src) => unimplemented!(),
            MovMapIntInt(dst, src) => unimplemented!(),
            MovMapIntFloat(dst, src) => unimplemented!(),
            MovMapIntStr(dst, src) => unimplemented!(),
            MovMapStrInt(dst, src) => unimplemented!(),
            MovMapStrFloat(dst, src) => unimplemented!(),
            MovMapStrStr(dst, src) => unimplemented!(),
            ReadErr(dst, file) => unimplemented!(),
            NextLine(dst, file) => unimplemented!(),
            ReadErrStdin(dst) => unimplemented!(),
            NextLineStdin(dst) => unimplemented!(),

            IterBeginIntInt(dst, arr) => unimplemented!(),
            IterBeginIntFloat(dst, arr) => unimplemented!(),
            IterBeginIntStr(dst, arr) => unimplemented!(),
            IterBeginStrInt(dst, arr) => unimplemented!(),
            IterBeginStrFloat(dst, arr) => unimplemented!(),
            IterBeginStrStr(dst, arr) => unimplemented!(),
            IterHasNextInt(dst, iter) => unimplemented!(),
            IterHasNextStr(dst, iter) => unimplemented!(),
            IterGetNextInt(dst, iter) => unimplemented!(),
            IterGetNextStr(dst, iter) => unimplemented!(),

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
    unsafe fn gen_hl_inst(&mut self, inst: &compile::HighLevel) -> Result<()> {
        // NB Phis skip the drop?
        unimplemented!()
    }
}

pub unsafe fn test_codegen() {
    if llvm::support::LLVMLoadLibraryPermanently(ptr::null()) != 0 {
        panic!("failed to load in-process library");
    }
    // TODO:
    // LLVM boilerplate
    //   * figure out issues with module verification.
    // Compilation metadata
    //  * build set of globals and locals used per function. Build up call-graph during
    //    construction. Use globals to get fixed point.
    //  * Use "typed ir" to first declare all relevant functions, storing their declarations in a
    //    map, then going through each instruction piecemeal.
    //
    // Runtime
    //   * Figure out extern-C-able versions of the runtime.
    //   * Make sure main uses all functions somehow.
    //   * Most of these are simple, but some things like "moving a string" could be tougher.
    //   * They all may require a "pointer to the runtime" passed in to handle the regex maps, etc.
    //   => We can put the string table in there.
    //   * Make sure to look up how to convert between ints and strings in LLVM.
    // Codegen
    //   * With all the metadata in place, we can do kaleidoscope chapters 5, 7 to implement
    //     everything we need?
    //   * Functions get relevant globals as arguments. Ints and Floats do just
    //     fine, though we still have to figure out that the plan is for Maps.
    //     Globals and locals are aloca'd in the entry block (of main only, for globals).
    //     * Computing relevant globals will require some sort of call graph traveral.
    //   * We need to figure out what to do about Str.
    //      - We may need a custom Rc that we can store in a pointer (having the ref-count "one
    //        word back" or some-such; then expose everything using that).
    //      - We could store all strings as offsets into a vector (a pointer to which we pass to
    //        every function). Then string indexes could be normal u64s, and all string functions
    //        could take the pointer as well.
    //        > It adds an extra layer of indirection
    //        > _but_ so does moving Rc to the toplevel, and many of the string operations are
    //          fairly heavy-duty.
    //        > This may be the best route.

    // Shared data-structures
    let ctx = LLVMContextCreate();
    let module = raw_guard(
        LLVMModuleCreateWithNameInContext(c_str!("main"), ctx),
        LLVMDisposeModule,
    );
    let builder = raw_guard(LLVMCreateBuilderInContext(ctx), LLVMDisposeBuilder);
    // Jit-specific setup
    LLVM_InitializeNativeTarget();
    LLVM_InitializeNativeAsmPrinter();
    LLVMLinkInMCJIT();
    let mut maybe_engine = MaybeUninit::<LLVMExecutionEngineRef>::uninit();
    let mut err: *mut c_char = ptr::null_mut();
    if LLVMCreateExecutionEngineForModule(maybe_engine.as_mut_ptr(), *module, &mut err) != 0 {
        // NB: In general, want to LLVMDisposeMessage if we weren't just going to crash.
        panic!(
            "failed to create program: {}",
            CStr::from_ptr(err).to_str().unwrap()
        );
    }
    let engine = maybe_engine.assume_init();
    let pass_manager = raw_guard(
        LLVMCreateFunctionPassManagerForModule(*module),
        LLVMDisposePassManager,
    );
    // Take some passes present in most of the tutorials
    {
        use llvm::transforms::scalar::*;
        llvm::transforms::util::LLVMAddPromoteMemoryToRegisterPass(*pass_manager);
        LLVMAddConstantPropagationPass(*pass_manager);
        LLVMAddInstructionCombiningPass(*pass_manager);
        LLVMAddReassociatePass(*pass_manager);
        LLVMAddGVNPass(*pass_manager);
        LLVMAddCFGSimplificationPass(*pass_manager);
        LLVMInitializeFunctionPassManager(*pass_manager);
    }

    // Code generation for __test_print
    let testprint = {
        let testprint_type = LLVMFunctionType(LLVMVoidType(), ptr::null_mut(), 0, 0);
        let tp = LLVMAddFunction(*module, c_str!("__test_print"), testprint_type);
        LLVMSetLinkage(tp, LLVMLinkage::LLVMExternalLinkage);
        tp
    };

    // Code generation for main
    let i64_type = LLVMInt64TypeInContext(ctx);
    let func_ty = LLVMFunctionType(i64_type, ptr::null_mut(), 0, /*is_var_arg=*/ 0);
    let func = LLVMAddFunction(*module, c_str!("main"), func_ty);
    LLVMSetLinkage(func, LLVMLinkage::LLVMExternalLinkage);
    let block = LLVMAppendBasicBlockInContext(ctx, func, c_str!(""));
    LLVMPositionBuilderAtEnd(*builder, block);
    let _ = LLVMBuildCall(*builder, testprint, ptr::null_mut(), 0, c_str!(""));
    LLVMBuildRet(*builder, LLVMConstInt(i64_type, 2, /*sign_extend=*/ 1));
    LLVMRunFunctionPassManager(*pass_manager, func);
    // LLVMVerifyModule(
    //     *module,
    //     LLVMVerifierFailureAction::LLVMAbortProcessAction,
    //     &mut err,
    // );

    // Now, get the code and go!
    let func_addr = LLVMGetFunctionAddress(engine, c_str!("main"));
    if func_addr == 0 {
        panic!("main function is just null!");
    }
    let jitted_func = mem::transmute::<u64, extern "C" fn() -> i64>(func_addr);
    println!("running jitted code");
    LLVMDumpModule(*module);
    let res = jitted_func();
    println!("result={}", res);
    // LLVMBuildCall
}
