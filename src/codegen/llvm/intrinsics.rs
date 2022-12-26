//! Plumbing used to expose external functions written in rust to LLVM.
//!
//! The core data-structure here is [`IntrinsicMap`], which lazily declares external functions
//! based on a type signature.
use super::attr;
use crate::codegen::FunctionAttr;
use crate::common::Either;
use crate::compile::Ty;

use hashbrown::HashMap;
use libc::c_void;
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

    pub(crate) unsafe fn map_drop_fn(&self, ty: Ty) -> Option<LLVMValueRef> {
        use Ty::*;
        match ty {
            MapIntInt => Some(self.get(external!(drop_intint))),
            MapIntFloat => Some(self.get(external!(drop_intfloat))),
            MapIntStr => Some(self.get(external!(drop_intstr))),
            MapStrInt => Some(self.get(external!(drop_strint))),
            MapStrFloat => Some(self.get(external!(drop_strfloat))),
            MapStrStr => Some(self.get(external!(drop_strstr))),
            _ => None,
        }
    }

    pub(crate) fn register(
        &mut self,
        name: &str,
        cname: *const libc::c_char,
        ty: LLVMTypeRef,
        attrs: &[FunctionAttr],
        func: *mut c_void,
    ) {
        if let Some(old) = self.map.insert(
            func as usize,
            Intrinsic {
                name: cname,
                data: RefCell::new(Either::Left(ty)),
                attrs: attrs.iter().cloned().collect(),
                func,
            },
        ) {
            // Some functions that have distinct implementations may get merged in release builds
            // (e.g. alloc_intint vs. alloc_intfloat). In that case it's fine to just overwrite
            // the data. We still want this assert to run on debug builds, though, to guard against
            // typos/duplicate registrations.
            debug_assert!(
                false,
                "duplicate entry in intrinsics table for {} {:p}. Other entry: {}",
                name,
                func,
                unsafe { String::from_utf8_lossy(std::ffi::CStr::from_ptr(old.name).to_bytes()) }
            );
        }
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
        if !intr.attrs.is_empty() {
            attr::add_function_attrs(self.ctx, func, &intr.attrs[..]);
        }
        *val = Either::Right(func);
        func
    }
}
