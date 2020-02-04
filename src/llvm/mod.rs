use crate::builtins::Variable;
use crate::bytecode::{self, Accum};
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

// TODO add checking to ensure that no function gets a number of args greater than u32::max
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

struct View<'a> {
    f: &'a mut Function,
    tmap: &'a TypeMap,
    intrinsics: &'a HashMap<&'static str, LLVMValueRef>,
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
            let mut view = View {
                f: decl,
                tmap: &self.type_map,
                intrinsics: &self.intrinsics,
            };
            // TODO branches, etc.
            for inst in &bb.weight {
                match inst {
                    Either::Left(ll) => view.gen_ll_inst(ll)?,
                    Either::Right(hl) => view.gen_hl_inst(hl)?,
                }
            }
        }
        Ok(())
    }
}

impl<'a> View<'a> {
    unsafe fn get_local(&self, local: (NumTy, Ty)) -> Result<LLVMValueRef> {
        if let Some(v) = self.f.locals.get(&local) {
            Ok(*v)
        } else if let Some(ix) = self.f.globals.get(&local) {
            let gv = LLVMGetParam(self.f.val, *ix as libc::c_uint);
            Ok(if let Ty::Str = local.1 {
                // no point in loading the string directly. We manipulate them as pointers.
                gv
            } else {
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
        self.f.globals.get(&reg).is_some()
    }

    unsafe fn var_val(&self, v: &Variable) -> LLVMValueRef {
        LLVMConstInt(self.tmap.var_ty, *v as u64, /*sign_extend=*/ 0)
    }

    unsafe fn ref_reg(&mut self, reg: (NumTy, Ty)) -> Result<()> {
        let val = self.get_local(reg)?;
        self.ref_val(val, reg.1)
    }

    unsafe fn ref_val(&mut self, mut val: LLVMValueRef, ty: Ty) -> Result<()> {
        use Ty::*;
        match ty {
            MapIntInt | MapIntStr | MapIntFloat | MapStrInt | MapStrStr | MapStrFloat => {
                let func = self.intrinsics["ref_map"];
                LLVMBuildCall(self.f.builder, func, &mut val, 1, c_str!(""));
            }
            Str => {
                let func = self.intrinsics["ref_str"];
                LLVMBuildCall(self.f.builder, func, &mut val, 1, c_str!(""));
            }
            _ => {}
        };
        Ok(())
    }

    unsafe fn drop_reg(&mut self, reg: (NumTy, Ty)) -> Result<()> {
        let val = self.get_local(reg)?;
        self.drop_val(val, reg.1)
    }

    unsafe fn drop_val(&mut self, mut val: LLVMValueRef, ty: Ty) -> Result<()> {
        use Ty::*;
        match ty {
            MapIntInt | MapIntStr | MapIntFloat | MapStrInt | MapStrStr | MapStrFloat => {
                let func = self.intrinsics["drop_map"];
                LLVMBuildCall(self.f.builder, func, &mut val, 1, c_str!(""));
            }
            Str => {
                let func = self.intrinsics["drop_str"];
                LLVMBuildCall(self.f.builder, func, &mut val, 1, c_str!(""));
            }
            _ => {}
        };
        Ok(())
    }

    unsafe fn call(&mut self, func: &'static str, args: &mut [LLVMValueRef]) -> LLVMValueRef {
        let f = self.intrinsics[func];
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

    // TODO move intrinsics and tmap into some kind of view datastructure; too much param passing.
    unsafe fn bind_val(&mut self, val: (NumTy, Ty), to: LLVMValueRef) {
        // if val is global, then find the relevant parameter and store it directly.
        // if val is an existing local, fail
        // if val.ty is a string, alloca a new string, store it, then bind the result.
        // otherwise, just bind the result directly.
        #[cfg(debug_assertions)]
        {
            if let Ty::Str = val.1 {
                use llvm::LLVMTypeKind::*;
                // make sure we are passing string values, not pointers here.
                assert_eq!(LLVMGetTypeKind(LLVMTypeOf(to)), LLVMIntegerTypeKind);
            }
        }
        if let Some(ix) = self.f.globals.get(&val) {
            // We're storing into a global variable. If it's a string or map, that means we have to
            // alter the reference counts appropriately.
            //  - if Str, call drop, store, then ref on the global pointer directly.
            //  - if Map, load the value, drop it, ref `to` then store it
            //  - otherwise, just store it directly
            let param = LLVMGetParam(self.f.val, *ix as libc::c_uint);
            let new_global = to;
            use Ty::*;
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
        }
        debug_assert!(self.f.locals.get(&val).is_none());
        if let Ty::Str = val.1 {
            let str_ty = self.tmap.get_ty(Ty::Str);
            let loc = LLVMBuildAlloca(self.f.builder, str_ty, c_str!(""));
            LLVMBuildStore(self.f.builder, to, loc);
            self.f.locals.insert(val, loc);
        } else {
            self.f.locals.insert(val, to);
        }
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
        let resv = self.call(func, &mut [mapv, keyv, valv]);
        Ok(())
    }

    unsafe fn runtime_val(&self) -> LLVMValueRef {
        LLVMGetParam(self.f.val, self.f.num_args as libc::c_uint - 1)
    }

    unsafe fn gen_ll_inst<'b>(&mut self, inst: &compile::LL<'b>) -> Result<()> {
        use crate::bytecode::Instr::*;
        match inst {
            StoreConstStr(sr, s) => {
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
                let cmp = LLVMBuildICmp(self.f.builder, Pred::LLVMIntEQ, operand, zero, c_str!(""));
                self.bind_reg(res, cmp);
            }
            NotStr(res, sr) => {
                let mut sv = self.get_local(sr.reflect())?;
                let strlen = self.intrinsics["str_len"];
                let lenv = LLVMBuildCall(self.f.builder, strlen, &mut sv, 1, c_str!(""));
                let ty = self.tmap.get_ty(Ty::Int);
                let zero = LLVMConstInt(ty, 0, /*sign_extend=*/ 1);
                let cmp = LLVMBuildICmp(self.f.builder, Pred::LLVMIntEQ, lenv, zero, c_str!(""));
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
                let resv = self.call("match_pat", &mut [rt, lv, rv]);
                self.bind_reg(res, resv);
            }
            LenStr(res, s) => {
                let sv = self.get_local(s.reflect())?;
                let lenv = self.call("str_len", &mut [sv]);
                self.bind_reg(res, lenv);
            }
            LTFloat(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = LLVMBuildFCmp(self.f.builder, FPred::LLVMRealOLT, lv, rv, c_str!(""));
                self.bind_reg(res, ltv);
            }
            LTInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = LLVMBuildICmp(self.f.builder, Pred::LLVMIntSLT, lv, rv, c_str!(""));
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
                let ltv = LLVMBuildFCmp(self.f.builder, FPred::LLVMRealOGT, lv, rv, c_str!(""));
                self.bind_reg(res, ltv);
            }
            GTInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = LLVMBuildICmp(self.f.builder, Pred::LLVMIntSGT, lv, rv, c_str!(""));
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
                let ltv = LLVMBuildFCmp(self.f.builder, FPred::LLVMRealOLE, lv, rv, c_str!(""));
                self.bind_reg(res, ltv);
            }
            LTEInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = LLVMBuildICmp(self.f.builder, Pred::LLVMIntSLE, lv, rv, c_str!(""));
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
                let ltv = LLVMBuildFCmp(self.f.builder, FPred::LLVMRealOGE, lv, rv, c_str!(""));
                self.bind_reg(res, ltv);
            }
            GTEInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = LLVMBuildICmp(self.f.builder, Pred::LLVMIntSGE, lv, rv, c_str!(""));
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
                let ltv = LLVMBuildFCmp(self.f.builder, FPred::LLVMRealOEQ, lv, rv, c_str!(""));
                self.bind_reg(res, ltv);
            }
            EQInt(res, l, r) => {
                let lv = self.get_local(l.reflect())?;
                let rv = self.get_local(r.reflect())?;
                let ltv = LLVMBuildICmp(self.f.builder, Pred::LLVMIntEQ, lv, rv, c_str!(""));
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
            PrintStdout(txt) => {
                let txtv = self.get_local(txt.reflect())?;
                self.call("print_stdout", &mut [self.runtime_val(), txtv]);
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
                let res = self.call("load_var_str", &mut [v]);
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
                self.call("store_var_str", &mut [v, sv]);
            }
            LoadVarInt(dst, var) => {
                let v = self.var_val(var);
                let res = self.call("load_var_int", &mut [v]);
                self.bind_reg(dst, res);
            }
            StoreVarInt(var, src) => {
                let v = self.var_val(var);
                let sv = self.get_local(src.reflect())?;
                self.call("store_var_int", &mut [v, sv]);
            }
            LoadVarIntMap(dst, var) => {
                let v = self.var_val(var);
                let res = self.call("load_var_intmap", &mut [v]);
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
                self.call("store_var_intmap", &mut [v, sv]);
            }
            // TODO(ezr): does this work?
            MovInt(dst, src) => self.bind_reg(dst, self.get_local(src.reflect())?),
            MovFloat(dst, src) => self.bind_reg(dst, self.get_local(src.reflect())?),
            MovStr(dst, src) => self.bind_reg(dst, self.get_local(src.reflect())?),
            MovMapIntInt(dst, src) => self.bind_reg(dst, self.get_local(src.reflect())?),
            MovMapIntFloat(dst, src) => self.bind_reg(dst, self.get_local(src.reflect())?),
            MovMapIntStr(dst, src) => self.bind_reg(dst, self.get_local(src.reflect())?),
            MovMapStrInt(dst, src) => self.bind_reg(dst, self.get_local(src.reflect())?),
            MovMapStrFloat(dst, src) => self.bind_reg(dst, self.get_local(src.reflect())?),
            MovMapStrStr(dst, src) => self.bind_reg(dst, self.get_local(src.reflect())?),
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
