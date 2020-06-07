//! This module contains "function defintions" called by compiled frawk programs
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
