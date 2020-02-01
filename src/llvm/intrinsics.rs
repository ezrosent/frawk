use super::llvm::{
    self,
    prelude::{LLVMContextRef, LLVMModuleRef, LLVMValueRef},
};
use crate::compile;
use crate::libc::c_void;
use crate::runtime::{
    self, FileRead, FileWrite, Float, Int, IntMap, Iter, LazyVec, RegexCache, Str, StrMap,
    Variables,
};
use hashbrown::HashMap;

use std::mem;

struct Runtime<'a> {
    vars: Variables<'a>,
    line: Str<'a>,
    split_line: LazyVec<Str<'a>>,
    regexes: RegexCache,
    write_files: FileWrite,
    read_files: FileRead,
}
impl<'a> Runtime<'a> {
    pub(crate) fn new(
        regs: impl Fn(compile::Ty) -> usize,
        stdin: impl std::io::Read + 'static,
        stdout: impl std::io::Write + 'static,
    ) -> Runtime<'a> {
        Runtime {
            vars: Default::default(),
            line: "".into(),
            split_line: LazyVec::new(),
            regexes: Default::default(),
            write_files: FileWrite::new(stdout),
            read_files: FileRead::new(stdin),
        }
    }
}

pub unsafe fn register(
    module: LLVMModuleRef,
    ctx: LLVMContextRef,
) -> HashMap<*mut () /* function address */, LLVMValueRef> {
    use llvm::core::*;
    let usize_ty = LLVMIntTypeInContext(ctx, (mem::size_of::<usize>() * 8) as libc::c_uint);
    let int_ty = LLVMIntTypeInContext(ctx, (mem::size_of::<Int>() * 8) as libc::c_uint);
    let float_ty = LLVMDoubleType();
    let void_ty = LLVMVoidType();
    let str_ty = LLVMIntTypeInContext(ctx, (mem::size_of::<Str>() * 8) as libc::c_uint);
    let rt_ty = LLVMPointerType(void_ty, 0);
    let mut table: HashMap<*mut (), LLVMValueRef> = Default::default();
    macro_rules! register_inner {
        ($name:ident, [ $($param:expr),* ], $ret:expr) => { {
            let mut params = [$($param),*];
            let ty = LLVMFunctionType($ret, params.as_mut_ptr(), params.len() as u32, 0);
            let func = LLVMAddFunction(module, c_str!(stringify!($name)), ty);
            LLVMSetLinkage(func, llvm::LLVMLinkage::LLVMExternalLinkage);
            if let Some(_) = table.insert($name as *mut (), func) {
                panic!("Duplicate registration for intrinsic {}", stringify!($name));
            }
        }};
    }
    macro_rules! register {
        ($name:ident ($($param:expr),*); $($rest:tt)*) => {
            register_inner!($name, [ $($param),* ], void_ty);
            register!($($rest)*);
        };
        ($name:ident ($($param:expr),*) -> $ret:expr; $($rest:tt)*) => {
            register_inner!($name, [ $($param),* ], $ret);
            register!($($rest)*);
        };
        () => {};
    }

    register! {
        ref_str(str_ty);
        drop_str(str_ty);
        ref_map(usize_ty);
        drop_map(usize_ty);
        int_to_str(int_ty) -> str_ty;
        float_to_str(float_ty) -> str_ty;
        str_to_int(str_ty) -> int_ty;
        str_to_float(str_ty) -> float_ty;
    };
    table
}

// TODO: NotStr => str_is_empty
// TODO: Concat => str_concat
// TODO: Match => match_pat
// TODO: LenStr => str_len
// TODO: <String comparison ops: LT, EQ, ...>
// TODO: Get/Set Column => get_col, set_col
// TODO: SplitInt/SplitStr => split_int, split_str
// TODO: Print/PrintStdout
// TODO: Map Lookup, Contains, Delete, Insert, Len, etc.
// TODO: Line handling
// TODO: Iterators

#[no_mangle]
pub unsafe extern "C" fn ref_str(s: u128) {
    mem::forget(mem::transmute::<u128, Str>(s).clone())
}

#[no_mangle]
pub unsafe extern "C" fn drop_str(s: u128) {
    mem::drop(mem::transmute::<u128, Str>(s))
}

unsafe fn ref_map_generic<K, V>(m: usize) {
    mem::forget(mem::transmute::<usize, runtime::SharedMap<K, V>>(m).clone())
}

unsafe fn drop_map_generic<K, V>(m: usize) {
    mem::drop(mem::transmute::<usize, runtime::SharedMap<K, V>>(m))
}

// XXX: relying on this doing the same thing regardless of type. We probably want a custom Rc to
// guarantee this.
#[no_mangle]
pub unsafe extern "C" fn ref_map(m: usize) {
    ref_map_generic::<Int, Int>(m)
}
#[no_mangle]
pub unsafe extern "C" fn drop_map(m: usize) {
    drop_map_generic::<Int, Int>(m)
}

#[no_mangle]
pub unsafe extern "C" fn int_to_str(i: Int) -> u128 {
    mem::transmute::<Str, u128>(runtime::convert::<Int, Str>(i))
}

#[no_mangle]
pub unsafe extern "C" fn float_to_str(f: Float) -> u128 {
    mem::transmute::<Str, u128>(runtime::convert::<Float, Str>(f))
}

#[no_mangle]
pub unsafe extern "C" fn str_to_int(s: u128) -> Int {
    let s = mem::transmute::<u128, Str>(s);
    let res = runtime::convert::<&Str, Int>(&s);
    mem::forget(s);
    res
}

#[no_mangle]
pub unsafe extern "C" fn str_to_float(s: u128) -> Float {
    let s = mem::transmute::<u128, Str>(s);
    let res = runtime::convert::<&Str, Float>(&s);
    mem::forget(s);
    res
}
