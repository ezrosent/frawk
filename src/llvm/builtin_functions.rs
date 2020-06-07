//! This module contains "function defintions" called by compiled frawk programs
//!
//! Unlike the intrinsics module, which exposes Rust implementations of some functionality through
//! some standard calling convention, these "builtins" are implemented in LLVM.

use super::TypeMap;

use crate::compile::Ty;

use llvm_sys::{core::*, prelude::*};

#[derive(Copy,Clone)]
pub enum Function {
    Pow,
}

impl Function {
    pub(crate) unsafe fn get_val(self, module: LLVMModuleRef, tmap: &TypeMap) -> LLVMValueRef {
        match self {
            Function::Pow => {
                // TODO: as we add more of these we can probably make a macro inside a
                // lazy_static.
                let id = LLVMLookupIntrinsicID(c_str!("llvm.pow"), "llvm.pow".len());
                LLVMGetIntrinsicDeclaration(module, id, &mut tmap.get_ty(Ty::Float), 1)
            }
        }
    }
}
