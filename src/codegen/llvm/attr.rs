//! LLVM-specific functionality for the `FunctionAttr` construct in the `codegen` module.
use crate::codegen::FunctionAttr;
use llvm_sys::core::*;
use llvm_sys::prelude::*;

fn cstr(fa: &FunctionAttr) -> (*const libc::c_char, usize) {
    macro_rules! cstr_len {
        ($s:expr) => {
            (c_str!($s), $s.len())
        };
    }
    use FunctionAttr::*;
    match fa {
        ReadOnly => cstr_len!("readonly"),
        ArgmemOnly => cstr_len!("argmemonly"),
    }
}

pub unsafe fn add_function_attrs(ctx: LLVMContextRef, func: LLVMValueRef, attrs: &[FunctionAttr]) {
    let func_index = llvm_sys::LLVMAttributeFunctionIndex;
    for a in attrs {
        let (name, len) = cstr(a);
        let kind = LLVMGetEnumAttributeKindForName(name, len);
        assert_ne!(kind, 0);
        let attr = LLVMCreateEnumAttribute(ctx, kind, 0);
        LLVMAddAttributeAtIndex(func, func_index, attr);
    }
}
