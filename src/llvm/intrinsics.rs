//! Plumbing used to expose external functions written in rust to LLVM.
//!
//! The core data-structure here is [`IntrinsicMap`], which lazily decalres external functions
//! based on a type signature.
use super::attr;
use crate::codegen::FunctionAttr;
use crate::common::Either;
use crate::libc::c_void;

use hashbrown::HashMap;
use llvm_sys::{
    self,
    prelude::{LLVMContextRef, LLVMModuleRef, LLVMTypeRef, LLVMValueRef},
    support::LLVMAddSymbol,
};
use smallvec;
use std::cell::RefCell;

struct Intrinsic {
    name: *const libc::c_char,
    data: RefCell<Either<LLVMTypeRef, LLVMValueRef>>,
    attrs: smallvec::SmallVec<[FunctionAttr; 1]>,
    func: *mut c_void,
}

// A map of intrinsics that lazily declares them when they are used in codegen.
pub(crate) struct IntrinsicMap {
    module: LLVMModuleRef,
    ctx: LLVMContextRef,
    map: HashMap<usize, Intrinsic>,
}

impl IntrinsicMap {
    pub(crate) fn new(module: LLVMModuleRef, ctx: LLVMContextRef) -> IntrinsicMap {
        IntrinsicMap {
            ctx,
            module,
            map: Default::default(),
        }
    }
    pub(crate) fn register(
        &mut self,
        cname: *const libc::c_char,
        ty: LLVMTypeRef,
        attrs: &[FunctionAttr],
        func: *mut c_void,
    ) {
        assert!(self
            .map
            .insert(
                func as usize,
                Intrinsic {
                    name: cname,
                    data: RefCell::new(Either::Left(ty)),
                    attrs: attrs.iter().cloned().collect(),
                    func,
                }
            )
            .is_none())
    }

    pub(crate) unsafe fn get(&self, func: *const u8) -> LLVMValueRef {
        use llvm_sys::core::*;
        let intr = &self.map[&(func as usize)];
        let mut val = intr.data.borrow_mut();

        let ty = match &mut *val {
            Either::Left(ty) => *ty,
            Either::Right(v) => return *v,
        };
        LLVMAddSymbol(intr.name, intr.func);
        let func = LLVMAddFunction(self.module, intr.name, ty);
        LLVMSetLinkage(func, llvm_sys::LLVMLinkage::LLVMExternalLinkage);
        if intr.attrs.len() > 0 {
            attr::add_function_attrs(self.ctx, func, &intr.attrs[..]);
        }
        *val = Either::Right(func);
        func
    }
}
