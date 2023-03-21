//! This module contains "function definitions" called by compiled frawk programs
//!
//! Unlike the intrinsics module, which exposes Rust implementations of some functionality through
//! some standard calling convention, these "builtins" are implemented in LLVM.

use super::TypeMap;

use crate::compile::Ty;

use lazy_static::lazy_static;
use libc::c_uint;
use llvm_sys::{core::*, prelude::*};

#[derive(Copy, Clone)]
pub enum Function {
    Pow,
    Sin,
    Cos,
    Sqrt,
    Log,
    Log2,
    Log10,
    Exp,
}

macro_rules! intrinsic_id {
    ($name:expr) => {
        unsafe {
            let len = $name.len();
            LLVMLookupIntrinsicID(c_str!($name), len)
        }
    };
}

lazy_static! {
    static ref POW_ID: c_uint = intrinsic_id!("llvm.pow");
    static ref SIN_ID: c_uint = intrinsic_id!("llvm.sin");
    static ref COS_ID: c_uint = intrinsic_id!("llvm.cos");
    static ref SQRT_ID: c_uint = intrinsic_id!("llvm.sqrt");
    static ref LOG_ID: c_uint = intrinsic_id!("llvm.log");
    static ref LOG2_ID: c_uint = intrinsic_id!("llvm.log2");
    static ref LOG10_ID: c_uint = intrinsic_id!("llvm.log10");
    static ref EXP_ID: c_uint = intrinsic_id!("llvm.exp");
}

/// Dropping a string is one of the more common operations performed by a frawk program. Strings
/// are often smalled and can thereby be stored inline, making drop "close to trivial." We
/// implement this fast path in LLVM so it can be inlined into the body of the program, still
/// relying on a Rust implementation for the "slow" variants that point to heap memory.
pub(crate) unsafe fn gen_drop_str(
    ctx: LLVMContextRef,
    module: LLVMModuleRef,
    tmap: &TypeMap,
    drop_slow: LLVMValueRef,
) -> LLVMValueRef {
    let fty = LLVMFunctionType(
        LLVMVoidTypeInContext(ctx),
        &mut tmap.get_ptr_ty(Ty::Str),
        1,
        0,
    );
    let decl = LLVMAddFunction(module, c_str!("drop_str_fast"), fty);
    LLVMSetLinkage(decl, llvm_sys::LLVMLinkage::LLVMLinkerPrivateLinkage);
    let builder = LLVMCreateBuilderInContext(ctx);
    let entry = LLVMAppendBasicBlockInContext(ctx, decl, c_str!(""));
    let fast = LLVMAppendBasicBlockInContext(ctx, decl, c_str!(""));
    let slow = LLVMAppendBasicBlockInContext(ctx, decl, c_str!(""));
    LLVMPositionBuilderAtEnd(builder, entry);
    // Load the string's representation
    let arg = LLVMGetParam(decl, 0);
    let str_rep = LLVMBuildLoad(builder, arg, c_str!(""));

    // TODO: Could we just truncate to int3 here? This is closer to the Rust version but that may
    // be more idiomatic llvm.
    let int64_ty = LLVMIntTypeInContext(ctx, 64);

    // Take the low 64 bits, then extract the tag
    let tag_mask = LLVMConstInt(int64_ty, 7, /*sign_extend=*/ 0);
    let low_64 = LLVMBuildTrunc(builder, str_rep, int64_ty, c_str!(""));
    let tag = LLVMBuildAnd(builder, low_64, tag_mask, c_str!(""));
    // test = tag < StrTag::Shared as u64
    let test = LLVMBuildICmp(
        builder,
        llvm_sys::LLVMIntPredicate::LLVMIntULT,
        tag,
        LLVMConstInt(int64_ty, 2, /*sign_extend=*/ 0),
        c_str!(""),
    );
    LLVMBuildCondBr(builder, test, fast, slow);

    // Fast path, just return
    LLVMPositionBuilderAtEnd(builder, fast);
    LLVMBuildRetVoid(builder);

    // Slow path, call the slow drop
    LLVMPositionBuilderAtEnd(builder, slow);
    LLVMBuildCall(builder, drop_slow, [arg, tag].as_mut_ptr(), 2, c_str!(""));
    LLVMBuildRetVoid(builder);
    LLVMDisposeBuilder(builder);
    decl
}

impl Function {
    pub(crate) unsafe fn get_val(self, module: LLVMModuleRef, tmap: &TypeMap) -> LLVMValueRef {
        match self {
            Function::Pow => {
                LLVMGetIntrinsicDeclaration(module, *POW_ID, &mut tmap.get_ty(Ty::Float), 1)
            }
            Function::Sin => {
                LLVMGetIntrinsicDeclaration(module, *SIN_ID, &mut tmap.get_ty(Ty::Float), 1)
            }
            Function::Cos => {
                LLVMGetIntrinsicDeclaration(module, *COS_ID, &mut tmap.get_ty(Ty::Float), 1)
            }
            Function::Sqrt => {
                LLVMGetIntrinsicDeclaration(module, *SQRT_ID, &mut tmap.get_ty(Ty::Float), 1)
            }
            Function::Log => {
                LLVMGetIntrinsicDeclaration(module, *LOG_ID, &mut tmap.get_ty(Ty::Float), 1)
            }
            Function::Log2 => {
                LLVMGetIntrinsicDeclaration(module, *LOG2_ID, &mut tmap.get_ty(Ty::Float), 1)
            }
            Function::Log10 => {
                LLVMGetIntrinsicDeclaration(module, *LOG10_ID, &mut tmap.get_ty(Ty::Float), 1)
            }
            Function::Exp => {
                LLVMGetIntrinsicDeclaration(module, *EXP_ID, &mut tmap.get_ty(Ty::Float), 1)
            }
        }
    }
}
