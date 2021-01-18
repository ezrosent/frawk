//! LLVM code generation for frawk programs.
mod attr;
pub(crate) mod builtin_functions;
pub(crate) mod intrinsics;

use crate::builtins;
use crate::bytecode::Accum;
use crate::codegen::{
    self, intrinsics::register_all, Backend, CodeGenerator, Jit, Ref, Sig, StrReg,
};
use crate::common::{Either, FileSpec, NodeIx, NumTy, Result, Stage};
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

use intrinsics::IntrinsicMap;

use std::ffi::{CStr, CString};
use std::mem::{self, MaybeUninit};
use std::ptr;

pub(crate) use codegen::Config;

type Pred = llvm_sys::LLVMIntPredicate;
type FPred = llvm_sys::LLVMRealPredicate;
type BuiltinFunc = builtin_functions::Function;

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

macro_rules! intrinsic {
    ($name:ident) => {
        crate::codegen::intrinsics::$name as *const u8
    };
}

struct View<'a> {
    f: &'a mut Function,
    decls: &'a Vec<FuncInfo>,
    tmap: &'a TypeMap,
    intrinsics: &'a mut IntrinsicMap,
    ctx: LLVMContextRef,
    module: LLVMModuleRef,
    printfs: &'a mut HashMap<(SmallVec<Ty>, PrintfKind), LLVMValueRef>,
    prints: &'a mut HashMap<(usize, /*stdout*/ bool), LLVMValueRef>,
    drop_str: LLVMValueRef,
    // We keep an extra builder always pointed at the start of the function. This is because
    // binding new string values requires an `alloca`; and we do not want to call `alloca` where a
    // string variable is referenced: for example, we do not want to call alloca in a loop.
    entry_builder: LLVMBuilderRef,
}

impl<'a> Backend for View<'a> {
    type Ty = LLVMTypeRef;
    fn void_ptr_ty(&self) -> Self::Ty {
        self.tmap.runtime_ty
    }
    fn u32_ty(&self) -> Self::Ty {
        unsafe { LLVMIntTypeInContext(self.ctx, 32) }
    }
    fn ptr_to(&self, ty: Self::Ty) -> Self::Ty {
        unsafe { LLVMPointerType(ty, 0) }
    }
    fn usize_ty(&self) -> Self::Ty {
        unsafe { LLVMIntTypeInContext(self.ctx, (mem::size_of::<*const u8>() * 8) as libc::c_uint) }
    }
    fn get_ty(&self, ty: compile::Ty) -> Self::Ty {
        self.tmap.get_ty(ty)
    }
    fn register_external_fn(
        &mut self,
        name: &'static str,
        name_c: *const u8,
        addr: *const u8,
        sig: Sig<Self>,
    ) -> Result<()> {
        let f_ty = unsafe {
            LLVMFunctionType(
                sig.ret.unwrap_or(LLVMVoidTypeInContext(self.ctx)),
                sig.args.as_mut_ptr(),
                sig.args.len() as u32,
                0,
            )
        };
        self.intrinsics
            .register(name, name_c as *const _, f_ty, sig.attrs, addr as *mut _);
        Ok(())
    }
}

impl<'a> CodeGenerator for View<'a> {
    type Val = LLVMValueRef;

    fn bind_val(&mut self, val: Ref, to: Self::Val) -> Result<()> {
        unsafe {
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
            if val.1 == Ty::Null {
                // We do not store null values explicitly
                return Ok(());
            }
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
                        self.drop_val(prev_global, val.1);
                        LLVMBuildStore(self.f.builder, new_global, param);
                        self.call(intrinsic!(ref_map), &mut [new_global]);
                    }
                    Str => {
                        self.drop_val(param, Ty::Str);
                        LLVMBuildStore(self.f.builder, new_global, param);
                        self.call(intrinsic!(ref_str), &mut [param]);
                    }
                    _ => {
                        LLVMBuildStore(self.f.builder, new_global, param);
                    }
                };
                return Ok(());
            }
            debug_assert!(
                self.f.locals.get(&val).is_none(),
                "we are inserting {:?}, but there is already something in there: {:?}",
                val,
                self.f.locals[&val]
            );
            match val.1 {
                MapIntInt | MapIntStr | MapIntFloat | MapStrInt | MapStrStr | MapStrFloat => {
                    // alloca only fails with an iterator or null type; but we have checked the type
                    // already.
                    let loc = self.alloca(val.1).unwrap();
                    let prev = LLVMBuildLoad(self.f.builder, loc, c_str!(""));
                    self.drop_val(prev, val.1);
                    LLVMBuildStore(self.f.builder, to, loc);
                    // NB: we used to have this here, but it leaked. See a segfault? It's possible we
                    // are missing some refs elsewhere.
                    //   self.call(intrinsic!(ref_map), &mut [to]);
                    // We had this for globals as well.
                    self.f.locals.insert(val, loc);
                    return Ok(());
                }
                Str => {
                    // Note: we ref strings ahead of time, either before calling bind_val in a
                    // MovStr, or as the result of a function call.

                    // unwrap justified like the above case for maps.
                    let loc = self.alloca(Ty::Str).unwrap();
                    self.drop_val(loc, Ty::Str);
                    LLVMBuildStore(self.f.builder, to, loc);
                    self.f.locals.insert(val, loc);
                    return Ok(());
                }
                _ => {}
            }
            self.f.locals.insert(val, to);
        }
        Ok(())
    }

    fn get_val(&mut self, r: Ref) -> Result<Self::Val> {
        match unsafe {
            self.get_local_inner(r, /*array_ptr=*/ false)
        } {
            Some(v) => Ok(v),
            None => err!("unbound variable {:?} (must call bind_val on it before)", r),
        }
    }

    fn runtime_val(&mut self) -> Self::Val {
        unsafe {
            LLVMGetParam(
                self.f.val,
                self.decls[self.f.id].num_args as libc::c_uint - 1,
            )
        }
    }

    fn const_int(&mut self, i: i64) -> Self::Val {
        unsafe {
            LLVMConstInt(self.get_ty(Ty::Int), i as u64, /*sign_extend=*/ 0)
        }
    }

    fn const_float(&mut self, f: f64) -> Self::Val {
        unsafe { LLVMConstReal(self.get_ty(Ty::Float), f) }
    }

    fn const_str<'b>(&mut self, s: &runtime::UniqueStr<'b>) -> Self::Val {
        // We don't know where we're storing this string literal. If it's in the middle of
        // a loop, we could be calling drop on it repeatedly. If the string is boxed, that
        // will lead to double-frees. In our current setup, these literals will all be
        // either empty, or references to word-aligned arena-allocated strings, so that's
        // actually fine.
        let as_str = s.clone_str();
        assert!(as_str.drop_is_trivial());
        let sc = as_str.into_bits();
        // There is no way to pass a 128-bit integer to LLVM directly. We have to convert
        // it to a string first.
        let as_hex = CString::new(format!("{:x}", sc)).unwrap();
        let ty = self.tmap.get_ty(Ty::Str);
        unsafe {
            LLVMConstIntOfString(ty, as_hex.as_ptr(), /*radix=*/ 16)
        }
    }

    fn const_ptr<'b, T>(&'b mut self, c: &'b T) -> Self::Val {
        let voidp = self.tmap.runtime_ty;
        let int_ty = self.tmap.get_ty(Ty::Int);
        unsafe {
            let bits = LLVMConstInt(int_ty, c as *const T as u64, /*sign_extend=*/ 0);
            LLVMBuildIntToPtr(self.f.builder, bits, voidp, c_str!(""))
        }
    }

    fn call_intrinsic(&mut self, func: codegen::Op, args: &mut [Self::Val]) -> Result<Self::Val> {
        use codegen::Op::*;
        fn to_pred(cmp: codegen::Cmp, is_float: bool) -> Either<Pred, FPred> {
            use codegen::Cmp::*;
            if is_float {
                Either::Right(match cmp {
                    // LLVM gives you `O` and `U` variants for float comparisons that "fail true"
                    // or "fail false" if either operand is NaN. `O` is  what matches the bytecode
                    // interpreter, but we may want to switch this around at some point.
                    EQ => FPred::LLVMRealOEQ,
                    LT => FPred::LLVMRealOLT,
                    LTE => FPred::LLVMRealOLE,
                    GT => FPred::LLVMRealOGT,
                    GTE => FPred::LLVMRealOGE,
                })
            } else {
                Either::Left(match cmp {
                    EQ => Pred::LLVMIntEQ,
                    LT => Pred::LLVMIntSLT,
                    LTE => Pred::LLVMIntSLE,
                    GT => Pred::LLVMIntSGT,
                    GTE => Pred::LLVMIntSGE,
                })
            }
        }
        fn translate_float_func(
            ff: builtins::FloatFunc,
        ) -> Either<*const u8, builtin_functions::Function> {
            use builtins::FloatFunc::*;
            match ff {
                Cos => Either::Right(builtin_functions::Function::Cos),
                Sin => Either::Right(builtin_functions::Function::Sin),
                Log => Either::Right(builtin_functions::Function::Log),
                Log2 => Either::Right(builtin_functions::Function::Log2),
                Log10 => Either::Right(builtin_functions::Function::Log10),
                Sqrt => Either::Right(builtin_functions::Function::Sqrt),
                Exp => Either::Right(builtin_functions::Function::Exp),
                Atan => Either::Left(codegen::intrinsics::_frawk_atan as _),
                Atan2 => Either::Left(codegen::intrinsics::_frawk_atan2 as _),
            }
        }
        unsafe {
            match func {
                Cmp { is_float, op } => Ok(self.cmp(to_pred(op, is_float), args[0], args[1])),
                Arith { is_float, op } => {
                    use codegen::Arith::*;
                    let res = if is_float {
                        match op {
                            Mul => LLVMBuildFMul(self.f.builder, args[0], args[1], c_str!("")),
                            Minus => LLVMBuildFSub(self.f.builder, args[0], args[1], c_str!("")),
                            Add => LLVMBuildFAdd(self.f.builder, args[0], args[1], c_str!("")),
                            Mod => LLVMBuildFRem(self.f.builder, args[0], args[1], c_str!("")),
                            Neg => LLVMBuildFNeg(self.f.builder, args[0], c_str!("")),
                        }
                    } else {
                        match op {
                            Mul => LLVMBuildMul(self.f.builder, args[0], args[1], c_str!("")),
                            Minus => LLVMBuildSub(self.f.builder, args[0], args[1], c_str!("")),
                            Add => LLVMBuildAdd(self.f.builder, args[0], args[1], c_str!("")),
                            Mod => LLVMBuildSRem(self.f.builder, args[0], args[1], c_str!("")),
                            Neg => {
                                let zero = self.const_int(0);
                                LLVMBuildSub(self.f.builder, zero, args[0], c_str!(""))
                            }
                        }
                    };
                    Ok(res)
                }
                Bitwise(bw) => {
                    use builtins::Bitwise::*;
                    Ok(match bw {
                        Complement => LLVMBuildXor(
                            self.f.builder,
                            args[0],
                            LLVMConstInt(self.get_ty(Ty::Int), !0, /*sign_extend=*/ 1),
                            c_str!(""),
                        ),
                        And => LLVMBuildAnd(self.f.builder, args[0], args[1], c_str!("")),
                        Or => LLVMBuildOr(self.f.builder, args[0], args[1], c_str!("")),
                        LogicalRightShift => {
                            LLVMBuildLShr(self.f.builder, args[0], args[1], c_str!(""))
                        }
                        ArithmeticRightShift => {
                            LLVMBuildAShr(self.f.builder, args[0], args[1], c_str!(""))
                        }
                        LeftShift => LLVMBuildShl(self.f.builder, args[0], args[1], c_str!("")),
                        Xor => LLVMBuildXor(self.f.builder, args[0], args[1], c_str!("")),
                    })
                }
                Math(ff) => Ok(match translate_float_func(ff) {
                    Either::Left(fname) => self.call(fname, args),
                    Either::Right(builtin) => self.call_builtin(builtin, args),
                }),
                Div => Ok(LLVMBuildFDiv(self.f.builder, args[0], args[1], c_str!(""))),
                Pow => Ok(self.call_builtin(BuiltinFunc::Pow, args)),
                FloatToInt => Ok(LLVMBuildFPToSI(
                    self.f.builder,
                    args[0],
                    self.get_ty(Ty::Int),
                    c_str!(""),
                )),
                IntToFloat => Ok(LLVMBuildSIToFP(
                    self.f.builder,
                    args[0],
                    self.get_ty(Ty::Float),
                    c_str!(""),
                )),
                Intrinsic(f) => Ok(self.call(f, args)),
            }
        }
    }
    fn printf(
        &mut self,
        output: &Option<(StrReg, FileSpec)>,
        fmt: &StrReg,
        args: &[Ref],
    ) -> Result<()> {
        unsafe {
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
            arg_vs.push(self.get_val(fmt.reflect())?);
            for a in args.iter().cloned() {
                arg_vs.push(self.get_val(a)?);
            }
            if let Some((path, append)) = output {
                arg_vs.push(self.get_val(path.reflect())?);
                let int_ty = self.tmap.get_ty(Ty::Int);
                arg_vs.push(LLVMConstInt(int_ty, *append as u64, 0));
            }
            LLVMBuildCall(
                self.f.builder,
                printf_fn,
                arg_vs.as_mut_ptr(),
                arg_vs.len() as libc::c_uint,
                c_str!(""),
            );
        }
        Ok(())
    }
    fn sprintf(&mut self, dst: &StrReg, fmt: &StrReg, args: &[Ref]) -> Result<()> {
        unsafe {
            let arg_tys: SmallVec<_> = args.iter().map(|x| x.1).collect();
            let sprintf_fn = self.wrapped_printf((arg_tys, PrintfKind::Sprintf));
            let mut arg_vs = SmallVec::with_capacity(args.len() + 1);
            arg_vs.push(self.runtime_val());
            arg_vs.push(self.get_val(fmt.reflect())?);
            for a in args.iter().cloned() {
                arg_vs.push(self.get_val(a)?);
            }
            let resv = LLVMBuildCall(
                self.f.builder,
                sprintf_fn,
                arg_vs.as_mut_ptr(),
                arg_vs.len() as libc::c_uint,
                c_str!(""),
            );
            self.bind_val(dst.reflect(), resv)
        }
    }
    fn print_all(&mut self, output: &Option<(StrReg, FileSpec)>, args: &[StrReg]) -> Result<()> {
        unsafe {
            let print_fn = self.print_all_fn(args.len(), /*is_stdout=*/ output.is_none())?;
            let mut args_v =
                SmallVec::with_capacity(args.len() + 1 + if output.is_some() { 2 } else { 0 });
            args_v.push(self.runtime_val());
            for a in args.iter() {
                args_v.push(self.get_val(a.reflect())?);
            }
            if let Some((out, fspec)) = output {
                args_v.push(self.get_val(out.reflect())?);
                let int_ty = self.tmap.get_ty(Ty::Int);
                args_v.push(LLVMConstInt(int_ty, *fspec as u64, /*sign_extend=*/ 0));
            }
            LLVMBuildCall(
                self.f.builder,
                print_fn,
                args_v.as_mut_ptr(),
                args_v.len() as libc::c_uint,
                c_str!(""),
            );
        }
        Ok(())
    }
    fn mov(&mut self, ty: compile::Ty, dst: NumTy, src: NumTy) -> Result<()> {
        unsafe {
            if let Ty::Str = ty {
                let sv = self.get_val((src, Ty::Str))?;
                self.call(intrinsic!(ref_str), &mut [sv]);
                let loaded = LLVMBuildLoad(self.f.builder, sv, c_str!(""));
                self.bind_val((dst, Ty::Str), loaded)
            } else {
                let sv = self.get_val((src, ty))?;
                if ty.is_array() {
                    self.call(intrinsic!(ref_map), &mut [sv]);
                }
                self.bind_val((dst, ty), sv)
            }
        }
    }
    fn iter_begin(&mut self, dst: Ref, map: Ref) -> Result<()> {
        unsafe {
            use Ty::*;
            let arrv = self.get_val(map)?;
            let (len_fn, begin_fn) = match map.1 {
                MapIntInt => (intrinsic!(len_intint), intrinsic!(iter_intint)),
                MapIntStr => (intrinsic!(len_intstr), intrinsic!(iter_intstr)),
                MapIntFloat => (intrinsic!(len_intfloat), intrinsic!(iter_intfloat)),
                MapStrInt => (intrinsic!(len_strint), intrinsic!(iter_strint)),
                MapStrStr => (intrinsic!(len_strstr), intrinsic!(iter_strstr)),
                MapStrFloat => (intrinsic!(len_strfloat), intrinsic!(iter_strfloat)),
                _ => return err!("iterating over non-map type: {:?}", map.1),
            };

            let iter_ptr = self.call(begin_fn, &mut [arrv]);
            let cur_index = self.alloca(Ty::Int)?;

            let ty = self.tmap.get_ty(Ty::Int);
            let zero = LLVMConstInt(ty, 0, /*sign_extend=*/ 1);
            LLVMBuildStore(self.f.builder, zero, cur_index);
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
    }
    fn iter_hasnext(&mut self, dst: Ref, iter: Ref) -> Result<()> {
        unsafe {
            let istate = self.get_iter(iter)?;
            let cur = LLVMBuildLoad(self.f.builder, istate.cur_index, c_str!(""));
            let len = istate.len;
            let hasnext = self.cmp(Either::Left(Pred::LLVMIntULT), cur, len);
            self.bind_val(dst, hasnext)
        }
    }
    fn iter_getnext(&mut self, dst: Ref, iter: Ref) -> Result<()> {
        let (res, res_loc) = unsafe {
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
            unsafe { self.call(intrinsic!(ref_str), &mut [res_loc]) };
        }
        self.bind_val(dst, res)
    }
    fn var_loaded(&mut self, dst: Ref) -> Result<()> {
        if (dst.1.is_array() || dst.1 == Ty::Str) && self.is_global(dst) {
            unsafe { self.drop_reg(dst)? };
        }
        Ok(())
    }
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
pub(crate) struct TypeMap {
    // Map from compile::Ty => TypeRef
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

    fn init(&mut self, ty: Ty, r: TypeRef) {
        self.table[ty as usize] = r;
    }

    fn get_ty(&self, ty: Ty) -> LLVMTypeRef {
        self.table[ty as usize].base
    }

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
    prints: HashMap<(usize, /*stdout*/ bool), LLVMValueRef>,
    // We pass raw regex pointers in the generated code. These ensure we do not free them
    // before the code is run.
    cfg: Config,

    // Specialized implementation of string destruction.
    drop_str: LLVMValueRef,
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
        // NB do we really need the NULL here? or could we omit calls
        Null | Int => LLVMConstInt(tmap.get_ty(Int), 0, /*sign_extend=*/ 1),
        Float => LLVMConstReal(tmap.get_ty(Float), 0.0),
        Str => {
            let str_ty = tmap.get_ty(Str);
            let v = LLVMConstInt(str_ty, 0, /*sign_extend=*/ 0);
            let v_loc = LLVMBuildAlloca(builder, str_ty, c_str!(""));
            LLVMBuildStore(builder, v, v_loc);
            v_loc
        }
        MapIntInt | MapIntStr | MapIntFloat | MapStrInt | MapStrStr | MapStrFloat => {
            let func = match ty {
                MapIntInt => intrinsic!(alloc_intint),
                MapIntFloat => intrinsic!(alloc_intfloat),
                MapIntStr => intrinsic!(alloc_intstr),
                MapStrInt => intrinsic!(alloc_strint),
                MapStrFloat => intrinsic!(alloc_strfloat),
                MapStrStr => intrinsic!(alloc_strstr),
                _ => unreachable!(),
            };
            let map_ty = tmap.get_ty(ty);
            let v = LLVMBuildCall(
                builder,
                intrinsics.get(func),
                ptr::null_mut(),
                0,
                c_str!(""),
            );
            let v_loc = LLVMBuildAlloca(builder, map_ty, c_str!(""));
            LLVMBuildStore(builder, v, v_loc);
            v_loc
        }
        IterInt | IterStr => return err!("we should not be default-allocating any iterators"),
    };
    Ok(val)
}

// We could just as easily make this a method on Generator, but we need the per-field tracking that
// we get from NLL that is blocked by a method like that.
macro_rules! view_at {
    ($slf:expr, $func_id:expr, $entry_builder:expr) => {
        View {
            f: &mut $slf.funcs[$func_id],
            tmap: &$slf.type_map,
            intrinsics: &mut $slf.intrinsics,
            decls: &$slf.decls,
            printfs: &mut $slf.printfs,
            prints: &mut $slf.prints,
            ctx: $slf.ctx,
            module: $slf.module,
            drop_str: $slf.drop_str,
            entry_builder: $entry_builder,
        }
    };
}

impl<'a, 'b> Jit for Generator<'a, 'b> {
    fn main_pointers(&mut self) -> Result<Stage<*const u8>> {
        unsafe {
            let main = self.gen_main()?;
            self.verify()?;
            self.optimize(main.iter().map(|(_, x)| x).cloned())?;
            Ok(main.map(|(name, _)| LLVMGetFunctionAddress(self.engine, name) as *const u8))
        }
    }
}

impl<'a, 'b> Generator<'a, 'b> {
    pub unsafe fn optimize(&mut self, mains: impl Iterator<Item = LLVMValueRef>) -> Result<()> {
        // Based on optimize_module in weld, in turn based on similar code in the LLVM opt tool.
        use llvm_sys::transforms::pass_manager_builder::*;
        let mpm = LLVMCreatePassManager();
        let fpm = LLVMCreateFunctionPassManagerForModule(self.module);

        let builder = LLVMPassManagerBuilderCreate();
        LLVMPassManagerBuilderSetOptLevel(builder, self.cfg.opt_level as u32);
        LLVMPassManagerBuilderSetSizeLevel(builder, 0);
        match self.cfg.opt_level {
            0 => {}
            1 => LLVMPassManagerBuilderUseInlinerWithThreshold(builder, 50),
            2 => LLVMPassManagerBuilderUseInlinerWithThreshold(builder, 100),
            3 => LLVMPassManagerBuilderUseInlinerWithThreshold(builder, 250),
            _ => return err!("unrecognized opt level"),
        };

        LLVMPassManagerBuilderPopulateFunctionPassManager(builder, fpm);
        LLVMPassManagerBuilderPopulateModulePassManager(builder, mpm);
        LLVMPassManagerBuilderDispose(builder);

        for f in self.decls.iter() {
            if f.val.is_null() {
                // unused functions are given null values.
                continue;
            }
            LLVMRunFunctionPassManager(fpm, f.val);
        }
        for fv in self.printfs.values() {
            LLVMRunFunctionPassManager(fpm, *fv);
        }
        for main in mains {
            LLVMRunFunctionPassManager(fpm, main);
        }

        LLVMFinalizeFunctionPassManager(fpm);
        LLVMRunPassManager(mpm, self.module);
        LLVMDisposePassManager(fpm);
        LLVMDisposePassManager(mpm);
        Ok(())
    }

    pub unsafe fn init(types: &'b mut Typer<'a>, cfg: Config) -> Result<Generator<'a, 'b>> {
        if llvm_sys::support::LLVMLoadLibraryPermanently(ptr::null()) != 0 {
            return err!("failed to load in-process library");
        }
        let ctx = LLVMContextCreate();
        let module = LLVMModuleCreateWithNameInContext(c_str!("frawk_main"), ctx);
        // JIT-specific initialization.
        LLVM_InitializeNativeTarget();
        LLVM_InitializeNativeAsmPrinter();
        LLVM_InitializeNativeAsmParser();
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
            intrinsics: IntrinsicMap::new(module, ctx),
            printfs: Default::default(),
            prints: Default::default(),
            cfg,
            drop_str: ptr::null_mut(),
        };
        res.build_map();
        res.build_decls();
        // Construct a placeholder `View` and use it to register intrinsics.
        register_all(&mut view_at!(res, 0, ptr::null_mut()))?;
        let drop_slow = res.intrinsics.get(intrinsic!(drop_str_slow));
        res.drop_str =
            builtin_functions::gen_drop_str(res.ctx, res.module, &res.type_map, drop_slow);
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
        let mains = self.gen_main()?;
        self.verify()?;
        self.optimize(mains.iter().map(|(_, x)| x).cloned())?;
        Ok(self.dump_module_inner())
    }

    // For benchmarking.
    #[cfg(all(test, feature = "unstable"))]
    pub unsafe fn compile_main(&mut self) -> Result<()> {
        let mains = self.gen_main()?;
        self.verify()?;
        self.optimize(mains.iter().map(|(_, x)| x).cloned())?;
        let addr = LLVMGetFunctionAddress(self.engine, c_str!("__frawk_main"));
        ptr::read_volatile(&addr);
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
        self.type_map
            .init(Ty::Null, make(self.type_map.get_ty(Ty::Int)));
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
            let is_called = self.types.frames[i].is_called;
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
            let builder = LLVMCreateBuilderInContext(self.ctx);
            let val = if is_called {
                let val = LLVMAddFunction(self.module, name.as_ptr(), ty);
                // We make these private, as we generate a separate main that calls into them. This
                // way, function bodies that get inlined into main do not have to show up in
                // generated code.
                LLVMSetLinkage(val, llvm_sys::LLVMLinkage::LLVMLinkerPrivateLinkage);
                val
            } else {
                ptr::null_mut()
            };
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

    unsafe fn gen_main_function(
        &mut self,
        main_offset: usize,
        name: *const libc::c_char,
    ) -> Result<(*const libc::c_char, LLVMValueRef)> {
        let ty = LLVMFunctionType(
            LLVMVoidTypeInContext(self.ctx),
            &mut self.type_map.runtime_ty,
            1,
            /*IsVarArg=*/ 0,
        );
        let decl = LLVMAddFunction(self.module, name, ty);
        let builder = LLVMCreateBuilderInContext(self.ctx);
        let bb = LLVMAppendBasicBlockInContext(self.ctx, decl, c_str!(""));
        LLVMPositionBuilderAtEnd(builder, bb);

        // For now, iterate over each element of the stage and call each component in sequence.
        // We need to allocate all of the global variables that our main function uses, and then
        // pass them as arguments, along with the runtime.
        let main_info = &self.decls[main_offset];
        let mut args: SmallVec<_> = smallvec![ptr::null_mut(); main_info.num_args];
        for ((_reg, ty), arg_ix) in main_info.globals.iter() {
            let local = self.alloc_local(builder, *ty)?;
            let param = if ty.is_array() || matches!(ty, Ty::Str) {
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
        Ok((name, decl))
    }

    unsafe fn gen_main(&mut self) -> Result<Stage<(*const libc::c_char, LLVMValueRef)>> {
        use crate::common::traverse;
        match self.types.stage() {
            Stage::Main(main) => Ok(Stage::Main(
                self.gen_main_function(main, c_str!("__frawk_main"))?,
            )),
            Stage::Par {
                begin,
                main_loop,
                end,
            } => Ok(Stage::Par {
                begin: traverse(
                    begin.map(|off| self.gen_main_function(off, c_str!("__frawk_begin"))),
                )?,
                main_loop: traverse(
                    main_loop.map(|off| self.gen_main_function(off, c_str!("__frawk_main_loop"))),
                )?,
                end: traverse(
                    end.map(|off| self.gen_main_function(off, c_str!("__frawk_end_loop"))),
                )?,
            }),
        }
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
            let prog_str = self.dump_module_inner();
            err!(
                "Module verification failed: {}\nFull Module: {}",
                err_str,
                prog_str
            )
        } else {
            Ok(())
        };
        LLVMDisposeMessage(error);
        res
    }

    unsafe fn gen_function(&mut self, func_id: usize) -> Result<()> {
        use compile::HighLevel::*;
        let frame = &self.types.frames[func_id];
        if !frame.is_called {
            return Ok(());
        }
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
        // handle arguments
        let enum_args: SmallVec<_> = self.funcs[func_id]
            .args
            .iter()
            .cloned()
            .enumerate()
            .collect();
        let arg_set: HashSet<_> = enum_args.iter().map(|(_, x)| *x).collect();
        for (local, (reg, ty)) in frame.locals.iter() {
            // implicitly-declared locals are just the ones with a subscript of 0.
            // Args are handled separately, skip them for now.
            if local.sub == 0 && !arg_set.contains(&(*reg, *ty)) {
                // For maps, we need these to go in entry
                let val = self.alloc_local(entry_builder, *ty)?;
                self.funcs[func_id].locals.insert((*reg, *ty), val);
            }
        }

        // As of writing; we'll only ever have a single return statement for a given function, but
        // we do not lose very much by having this function support multiple returns if we decide
        // to refactor some of the higher-level code in the future.
        let mut exits = Vec::with_capacity(1);
        let mut phis = Vec::new();
        let mut view = view_at!(self, func_id, entry_builder);
        for (i, arg) in enum_args.into_iter() {
            let argv = LLVMGetParam(view.f.val, i as libc::c_uint);
            // We insert into `locals` directly because we know these aren't globals, and we want
            // to avoid the extra ref/drop for string params. We _do_ use bind_val on array
            // parameters because they may need to be alloca'd here. Strings have an alloca path in
            // bind_val, but it isn't needed for params because strings are passed by pointer.
            if arg.1.is_array() {
                view.bind_val(arg, argv)?;
            } else {
                view.f.locals.insert(arg, argv);
            }
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
            let bb = &frame.cfg.node_weight(n).unwrap().insts;
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
                            Phi(_, ty, _) if ty != &Ty::Null => phis.push((i, j)),
                            Phi(_, _, _) | DropIter(_, _) | Call { .. } => {}
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
                    tcase = Some((e, bb));
                } else {
                    // NB, we used to disallow duplicate unconditional branches outbound from a
                    // basic block, but it does seem to happen in some cases, but only benignly
                    // (where taking either branch results in the same behavior).
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
        let node_weight = |bb, inst| &frame.cfg.node_weight(NodeIx::new(bb)).unwrap().insts[inst];
        let mut placeholder_intrinsics = IntrinsicMap::new(view.module, view.ctx);
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
                mem::swap(view.intrinsics, &mut placeholder_intrinsics);
                let val = alloc_local(
                    view.entry_builder,
                    ty,
                    &self.type_map,
                    &placeholder_intrinsics,
                )?;
                mem::swap(view.intrinsics, &mut placeholder_intrinsics);
                view.ret_val(val, ty)?
            }
        }

        // Now that we have initialized all local variables, we can wire in predecessors to phis.
        let mut preds = SmallVec::new();
        let mut blocks = SmallVec::new();
        for (phi_bb, phi_inst) in phis.into_iter() {
            if let Either::Right(Phi(reg, ty, ps)) = node_weight(phi_bb, phi_inst) {
                let phi_node = view.get_local_raw((*reg, *ty))?;
                for (pred_bb, pred_reg) in ps.iter() {
                    preds.push(view.get_local_raw((*pred_reg, *ty))?);
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

    // TODO: rename this; it gets globals too :)
    unsafe fn get_local_inner(&self, local: (NumTy, Ty), array_ptr: bool) -> Option<LLVMValueRef> {
        if local.1 == Ty::Null {
            // Null values, while largely erased from the picture, are occasionally loaded for
            // returns and for parameter passing. We could (as we do in the bytecode interpreter)
            // just erase these null parameters from existence and make all null returns void
            // returns. However, this could result in some special-casing and additional complexity
            // around parameter list lengths for little apparent gain.
            //
            // Contrast this with the bytecode case, where we need only omit pushes and pops from
            // the stack.
            Some(LLVMConstInt(
                self.tmap.get_ty(Ty::Null),
                0,
                /*sign_extend=*/ 0,
            ))
        } else if let Some(v) = self.f.locals.get(&local) {
            if local.1.is_array() && !array_ptr {
                Some(LLVMBuildLoad(self.f.builder, *v, c_str!("")))
            } else {
                Some(*v)
            }
        } else if let Some(ix) = self.decls[self.f.id].globals.get(&local) {
            let gv = LLVMGetParam(self.f.val, *ix as libc::c_uint);
            Some(if let Ty::Str = local.1 {
                // no point in loading the string directly. We manipulate them as pointers.
                gv
            } else {
                // XXX: do we need to ref maps here?
                // NB: depends on what we do when calling UDFs that take maps as arguments.
                //     but either way, no.
                //     We _should_ clarify calling convention re: maps though.
                LLVMBuildLoad(self.f.builder, gv, c_str!(""))
            })
        } else {
            None
        }
    }
    unsafe fn get_param(&mut self, local: (NumTy, Ty)) -> Result<LLVMValueRef> {
        if let Some(v) = self.get_local_inner(local, /*array_ptr=*/ false) {
            Ok(v)
        } else {
            // Some parameters are never mentioned in the source program, but are just added in as
            // placeholders by the `compile` module. In that case, we'll have an initialized value
            // that is never mentioned elsewhere. We still need to drop it, but we do not need to
            // perform any extra dropping.
            let v = if let Ty::Int | Ty::Str = local.1 {
                let str_ty = self.tmap.get_ty(local.1);
                LLVMConstInt(str_ty, 0, /*sign_extend=*/ 0)
            } else {
                alloc_local(self.f.builder, local.1, self.tmap, self.intrinsics)?
            };
            self.bind_val(local, v)?;
            Ok(v)
        }
    }

    // This is used for phi nodes with maps.
    // get_local_inner will load values by default, but that's the wrong thing to do because phi
    // nodes contain the pointers themselves. The output is the same as get_local for non-array
    // types.
    unsafe fn get_local_raw(&self, local: (NumTy, Ty)) -> Result<LLVMValueRef> {
        match self.get_local_inner(local, /*array_ptr=*/ true) {
            Some(v) => Ok(v),
            None => err!(
                "unbound variable {:?} (must call bind_val on it before)",
                local
            ),
        }
    }

    fn is_global(&self, reg: (NumTy, Ty)) -> bool {
        self.decls[self.f.id].globals.get(&reg).is_some()
    }

    unsafe fn drop_reg(&mut self, reg: (NumTy, Ty)) -> Result<()> {
        let val = self.get_val(reg)?;
        self.drop_val(val, reg.1);
        Ok(())
    }

    unsafe fn drop_val(&mut self, mut val: LLVMValueRef, ty: Ty) {
        use Ty::*;
        let func = match ty {
            MapIntInt => self.intrinsics.get(intrinsic!(drop_intint)),
            MapIntFloat => self.intrinsics.get(intrinsic!(drop_intfloat)),
            MapIntStr => self.intrinsics.get(intrinsic!(drop_intstr)),
            MapStrInt => self.intrinsics.get(intrinsic!(drop_strint)),
            MapStrFloat => self.intrinsics.get(intrinsic!(drop_strfloat)),
            MapStrStr => self.intrinsics.get(intrinsic!(drop_strstr)),
            Str => self.drop_str,
            _ => return,
        };
        LLVMBuildCall(self.f.builder, func, &mut val, 1, c_str!(""));
    }

    unsafe fn call_builtin(&mut self, f: BuiltinFunc, args: &mut [LLVMValueRef]) -> LLVMValueRef {
        let fv = f.get_val(self.module, self.tmap);
        LLVMBuildCall(
            self.f.builder,
            fv,
            args.as_mut_ptr(),
            args.len() as libc::c_uint,
            c_str!(""),
        )
    }

    unsafe fn call_at(
        &mut self,
        builder: LLVMBuilderRef,
        func: *const u8,
        args: &mut [LLVMValueRef],
    ) -> LLVMValueRef {
        let f = self.intrinsics.get(func);
        LLVMBuildCall(
            builder,
            f,
            args.as_mut_ptr(),
            args.len() as libc::c_uint,
            c_str!(""),
        )
    }

    unsafe fn call(&mut self, func: *const u8, args: &mut [LLVMValueRef]) -> LLVMValueRef {
        self.call_at(self.f.builder, func, args)
    }

    unsafe fn alloca(&mut self, ty: Ty) -> Result<LLVMValueRef> {
        let alloc_fn = match ty {
            Ty::Int | Ty::Float | Ty::Str => {
                let ty = self.tmap.get_ty(ty);
                let res = LLVMBuildAlloca(self.entry_builder, ty, c_str!(""));
                let v = LLVMConstInt(ty, 0, /*sign_extend=*/ 0);
                LLVMBuildStore(self.entry_builder, v, res);
                return Ok(res);
            }
            Ty::IterInt | Ty::Null | Ty::IterStr => {
                return err!("unexpected type passed to alloca: {:?}", ty);
            }
            // map
            Ty::MapIntInt => intrinsic!(alloc_intint),
            Ty::MapIntFloat => intrinsic!(alloc_intfloat),
            Ty::MapIntStr => intrinsic!(alloc_intstr),
            Ty::MapStrInt => intrinsic!(alloc_strint),
            Ty::MapStrFloat => intrinsic!(alloc_strfloat),
            Ty::MapStrStr => intrinsic!(alloc_strstr),
        };
        let llty = self.tmap.get_ty(ty);
        let res = LLVMBuildAlloca(self.entry_builder, llty, c_str!(""));
        let v = self.call_at(self.entry_builder, alloc_fn, &mut []);
        LLVMBuildStore(self.entry_builder, v, res);
        Ok(res)
    }

    fn get_iter(&self, iter: (NumTy, Ty)) -> Result<&IterState> {
        if let Some(istate) = self.f.iters.get(&iter) {
            Ok(istate)
        } else {
            err!("unbound iterator: {:?}", iter)
        }
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
            let val = self.get_val((reg, Ty::Int))?;
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

    unsafe fn ret(&mut self, val: (NumTy, Ty)) -> Result<()> {
        let ret = self.get_val(val)?;
        self.ret_val(ret, val.1)
    }

    unsafe fn ret_val(&mut self, to_return: LLVMValueRef, ty: Ty) -> Result<()> {
        // We can't iterate over self.f.locals directly because drop_val borrows all of `self`.
        let locals: SmallVec<_> = self
            .f
            .locals
            .iter()
            .map(|((reg, ty), _)| (*reg, *ty))
            .collect();
        for l in locals.into_iter() {
            if self.f.skip_drop.contains(&l) {
                continue;
            }
            let llval = self.get_val(l)?;
            if llval == to_return {
                continue;
            }
            self.drop_val(llval, l.1);
        }
        if let Ty::Str = ty {
            let loaded = LLVMBuildLoad(self.f.builder, to_return, c_str!(""));
            LLVMBuildRet(self.f.builder, loaded);
        } else {
            LLVMBuildRet(self.f.builder, to_return);
        }
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
                    argvs[i] = self.get_param(arg)?;
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
                self.bind_val((*dst_reg, *dst_ty), resv)?;
            }
            Phi(reg, ty, _preds) => {
                if let Ty::Null = ty {
                    return Ok(());
                }
                self.f.skip_drop.insert((*reg, *ty));
                let res = LLVMBuildPhi(
                    self.f.builder,
                    if ty.is_array() || ty == &Ty::Str {
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
                    Ty::IterInt => intrinsic!(drop_iter_int),
                    Ty::IterStr => intrinsic!(drop_iter_str),
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
        use PrintfKind::*;
        let kind = key.1;
        if let Some(v) = self.printfs.get(&key) {
            return *v;
        }
        let args = &key.0[..];

        let name_c = gen_name("_pf", self.printfs.len());

        // The var-arg portion + runtime + format spec
        //  (+ output + append, if named_output)
        let mut arg_lltys = smallvec::SmallVec::<[_; 8]>::with_capacity(args.len() + 4);
        arg_lltys.push(self.tmap.runtime_ty);
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
            let offset = 2;
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
                let intrinsic = self.intrinsics.get(intrinsic!(printf_impl_file));
                // runtime, spec, args, tys, num_args, output, append
                let mut args = [
                    LLVMGetParam(f, 0),
                    LLVMGetParam(f, 1),
                    args_ptr,
                    tys_ptr,
                    len_v,
                    LLVMGetParam(f, arg_lltys.len() as libc::c_uint - 2),
                    LLVMGetParam(f, arg_lltys.len() as libc::c_uint - 1),
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
                let intrinsic = self.intrinsics.get(intrinsic!(printf_impl_stdout));
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
                let intrinsic = self.intrinsics.get(intrinsic!(sprintf_impl));
                let mut args = [
                    LLVMGetParam(f, 0),
                    LLVMGetParam(f, 1),
                    args_ptr,
                    tys_ptr,
                    len_v,
                ];
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

    // A scaled-down version of the strategy we have for printf. This is for the var-args "print"
    // function that only prints strings. That means we don't have to pass an array of types, and
    // we don't have to cast anything: we just pass a single array of strings. Otherwise, the setup
    // is quite similar.
    unsafe fn print_all_fn(&mut self, n_args: usize, is_stdout: bool) -> Result<LLVMValueRef> {
        let key = (n_args, is_stdout);
        if let Some(res) = self.prints.get(&key) {
            return Ok(*res);
        }

        // Build the function type.
        let mut arg_lltys = smallvec::SmallVec::<[_; 8]>::with_capacity(n_args + 3);
        arg_lltys.push(self.tmap.runtime_ty);
        // var-arg portion
        let str_ty = self.tmap.get_ptr_ty(Ty::Str);
        let int_ty = self.tmap.get_ty(Ty::Int);
        let u32_ty = LLVMIntTypeInContext(self.ctx, 32);
        arg_lltys.extend((0..n_args).map(|_| str_ty));
        if !is_stdout {
            // file params
            arg_lltys.push(str_ty);
            arg_lltys.push(int_ty);
        }
        let ret = LLVMVoidTypeInContext(self.ctx);
        let func_ty = LLVMFunctionType(ret, arg_lltys.as_mut_ptr(), arg_lltys.len() as u32, 0);

        // Set up the function
        let name = gen_name("_pa", self.prints.len());
        let builder = LLVMCreateBuilderInContext(self.ctx);
        let f = LLVMAddFunction(self.module, name.as_ptr() as *const libc::c_char, func_ty);
        let bb = LLVMAppendBasicBlockInContext(self.ctx, f, c_str!(""));
        LLVMPositionBuilderAtEnd(builder, bb);
        let len = n_args as libc::c_uint;
        let args_ty = LLVMArrayType(str_ty, len);
        let args_array = LLVMBuildAlloca(builder, args_ty, c_str!(""));
        let zero = LLVMConstInt(u32_ty, 0, /*sign_extend=*/ 0);

        for i in 0..n_args {
            let mut index = [zero, LLVMConstInt(u32_ty, i as u64, /*sign_extend=*/ 0)];
            let arg_ptr = LLVMBuildGEP(builder, args_array, index.as_mut_ptr(), 2, c_str!(""));
            // We are storing index `i` in the array, which is going to be argument `i + 1`
            let argval = LLVMGetParam(f, i as libc::c_uint + 1);
            LLVMBuildStore(builder, argval, arg_ptr);
        }
        let mut start_index = [zero, zero];
        let args_ptr = LLVMBuildGEP(builder, args_array, start_index.as_mut_ptr(), 2, c_str!(""));
        let len_v = LLVMConstInt(int_ty, len as u64, /*sign_extend=*/ 0);
        if is_stdout {
            let intrinsic = self.intrinsics.get(intrinsic!(print_all_stdout));
            let mut args = [LLVMGetParam(f, 0), args_ptr, len_v];
            LLVMBuildCall(
                builder,
                intrinsic,
                args.as_mut_ptr(),
                args.len() as libc::c_uint,
                c_str!(""),
            );
            LLVMBuildRetVoid(builder);
        } else {
            let intrinsic = self.intrinsics.get(intrinsic!(print_all_file));
            let out_v = LLVMGetParam(f, 1 + len);
            let spec_v = LLVMGetParam(f, 1 + len + 1);
            let mut args = [LLVMGetParam(f, 0), args_ptr, len_v, out_v, spec_v];
            LLVMBuildCall(
                builder,
                intrinsic,
                args.as_mut_ptr(),
                args.len() as libc::c_uint,
                c_str!(""),
            );
            LLVMBuildRetVoid(builder);
        }
        LLVMSetLinkage(f, llvm_sys::LLVMLinkage::LLVMLinkerPrivateLinkage);
        LLVMDisposeBuilder(builder);
        self.prints.insert(key, f);
        Ok(f)
    }
}

fn gen_name(base_name: &str, index: usize) -> [u8; 32] {
    use std::io::{Cursor, Write};
    // 64 bit integers should only ever need 20 digits or so.
    assert!(base_name.len() < 4);
    let mut name_c = [0u8; 32];
    for (i, b) in base_name.as_bytes().iter().enumerate() {
        name_c[i] = *b;
    }
    let mut w = Cursor::new(&mut name_c[base_name.as_bytes().len()..]);
    write!(w, "{:x}", index).unwrap();
    assert_eq!(name_c[name_c.len() - 1], 0);
    name_c
}
