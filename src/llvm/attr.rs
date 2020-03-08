/// Adds more ergonomic support for setting function attributes on functions. This is adapted from
/// the same functionality in the Weld project. For now we are not adding attributes to parameters,
/// just to the functions themselves.
use llvm_sys::core::*;
use llvm_sys::prelude::*;

#[derive(Debug, Copy, Clone)]
pub enum FunctionAttr {
    // There is more where that came from; we will add them as needed.
    ReadOnly,
    // Keeping this here as an example if we ever want to add more attributes.
    #[allow(unused)]
    ArgmemOnly,
}

impl FunctionAttr {
    fn cstr(&self) -> (*const libc::c_char, usize) {
        macro_rules! cstr_len {
            ($s:expr) => {
                (c_str!($s), $s.len())
            };
        }
        use FunctionAttr::*;
        match self {
            ReadOnly => cstr_len!("readonly"),
            ArgmemOnly => cstr_len!("argmemonly"),
        }
    }
}

pub unsafe fn add_function_attrs(ctx: LLVMContextRef, func: LLVMValueRef, attrs: &[FunctionAttr]) {
    let func_index = llvm_sys::LLVMAttributeFunctionIndex;
    for a in attrs {
        let (name, len) = a.cstr();
        let kind = LLVMGetEnumAttributeKindForName(name, len);
        assert_ne!(kind, 0);
        let attr = LLVMCreateEnumAttribute(ctx, kind, 0);
        LLVMAddAttributeAtIndex(func, func_index, attr);
    }
}
