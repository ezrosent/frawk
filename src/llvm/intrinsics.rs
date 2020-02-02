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

macro_rules! fail {
    ($($es:expr),+) => {{
        #[cfg(test)]
        {
            panic!("failure in runtime {}. Halting execution", format!($($es),*))
        }
        #[cfg(not(test))]
        {
            eprintln!("failure in runtime {}. Halting execution", format!($($es),*));
            std::process::abort()
        }
    }}
}

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
    let str_ref_ty = rt_ty;
    let mut table: HashMap<*mut (), LLVMValueRef> = Default::default();
    macro_rules! register_inner {
        ($name:ident, [ $($param:expr),* ], $ret:expr) => { {
            // Try and make sure the linker doesn't strip the function out.
            std::ptr::read_volatile($name as *const u8);
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
        ref_str(str_ref_ty);
        drop_str(str_ref_ty);
        ref_map(usize_ty);
        drop_map(usize_ty);
        int_to_str(int_ty) -> str_ty;
        float_to_str(float_ty) -> str_ty;
        str_to_int(str_ref_ty) -> int_ty;
        str_to_float(str_ref_ty) -> float_ty;
        str_len(str_ref_ty) -> usize_ty;
        concat(str_ref_ty, str_ref_ty) -> str_ty;
        match_pat(rt_ty, str_ref_ty, str_ref_ty) -> int_ty;
        get_col(rt_ty, int_ty) -> str_ty;
        set_col(rt_ty, int_ty, str_ref_ty);
        split_int(rt_ty, str_ref_ty, usize_ty, str_ref_ty) -> int_ty;
        split_str(rt_ty, str_ref_ty, usize_ty, str_ref_ty) -> int_ty;
        print_stdout(rt_ty, str_ref_ty);
        print(rt_ty, str_ref_ty, str_ref_ty, int_ty);
        str_lt(str_ref_ty, str_ref_ty) -> int_ty;
        str_gt(str_ref_ty, str_ref_ty) -> int_ty;
        str_lte(str_ref_ty, str_ref_ty) -> int_ty;
        str_gte(str_ref_ty, str_ref_ty) -> int_ty;
        str_eq(str_ref_ty, str_ref_ty) -> int_ty;
    };
    table
}

// TODO: Map Lookup, Contains, Delete, Insert, Len, etc.
// TODO: Line handling
// TODO: Iterators
// TODO: IO Errors.
//  - we need to exit cleanly. Add a "checkIOerror" builtin to main? set a variable in the runtime
//    and exit cleanly?
//  - get this working along with iterators after everything else is working.

#[no_mangle]
pub unsafe extern "C" fn print_stdout(runtime: *mut c_void, txt: *mut c_void) {
    let newline: Str<'static> = "\n".into();
    let runtime = &mut *(runtime as *mut Runtime);
    let txt = &*(txt as *mut Str);
    runtime.write_files.write_str_stdout(txt);
    if runtime.write_files.write_str_stdout(&newline).is_err() {
        fail!("handle errors in file writing!")
    }
}

#[no_mangle]
pub unsafe extern "C" fn print(
    runtime: *mut c_void,
    txt: *mut c_void,
    out: *mut c_void,
    append: Int,
) {
    let runtime = &mut *(runtime as *mut Runtime);
    let txt = &*(txt as *mut Str);
    let out = &*(out as *mut Str);
    if runtime
        .write_files
        .write_line(out, txt, append != 0)
        .is_err()
    {
        fail!("handle errors in file writing!")
    }
}

#[no_mangle]
pub unsafe extern "C" fn split_str(
    runtime: *mut c_void,
    to_split: *mut c_void,
    into_arr: usize,
    pat: *mut c_void,
) -> Int {
    let runtime = &mut *(runtime as *mut Runtime);
    let into_arr = mem::transmute::<usize, StrMap<Str>>(into_arr);
    let to_split = &*(to_split as *mut Str);
    let pat = &*(pat as *mut Str);
    let old_len = into_arr.len();
    if let Err(e) = runtime
        .regexes
        .split_regex_strmap(&pat, &to_split, &into_arr)
    {
        fail!("failed to split string: {}", e);
    }
    let res = (into_arr.len() - old_len) as Int;
    mem::forget((into_arr, to_split, pat));
    res
}

#[no_mangle]
pub unsafe extern "C" fn split_int(
    runtime: *mut c_void,
    to_split: *mut c_void,
    into_arr: usize,
    pat: *mut c_void,
) -> Int {
    let runtime = &mut *(runtime as *mut Runtime);
    let into_arr = mem::transmute::<usize, IntMap<Str>>(into_arr);
    let to_split = &*(to_split as *mut Str);
    let pat = &*(pat as *mut Str);
    let old_len = into_arr.len();
    if let Err(e) = runtime
        .regexes
        .split_regex_intmap(&pat, &to_split, &into_arr)
    {
        fail!("failed to split string: {}", e);
    }
    let res = (into_arr.len() - old_len) as Int;
    mem::forget((into_arr, to_split, pat));
    res
}

#[no_mangle]
pub unsafe extern "C" fn get_col(runtime: *mut c_void, col: Int) -> u128 {
    if col < 0 {
        fail!("attempt to access negative column: {}", col);
    }
    let runtime = &mut *(runtime as *mut Runtime);
    if col == 0 {
        return mem::transmute::<Str, u128>(runtime.line.clone());
    }
    if runtime.split_line.len() == 0 {
        if let Err(e) =
            runtime
                .regexes
                .split_regex(&runtime.vars.fs, &runtime.line, &mut runtime.split_line)
        {
            fail!("failed to split line: {}", e);
        }
        runtime.vars.nf = runtime.split_line.len() as Int;
    }
    let res = runtime
        .split_line
        .get(col as usize - 1)
        .unwrap_or_else(Str::default);
    mem::transmute::<Str, u128>(res)
}

#[no_mangle]
pub unsafe extern "C" fn set_col(runtime: *mut c_void, col: Int, s: *mut c_void) {
    if col < 0 {
        fail!("attempt to set negative column: {}", col);
    }
    let runtime = &mut *(runtime as *mut Runtime);
    if col == 0 {
        runtime.split_line.clear();
        ref_str(s);
        runtime.line = (*(s as *mut Str)).clone();
        runtime.vars.nf = -1;
        return;
    }
    if runtime.split_line.len() == 0 {
        if let Err(e) =
            runtime
                .regexes
                .split_regex(&runtime.vars.fs, &runtime.line, &mut runtime.split_line)
        {
            fail!("failed to split line: {}", e);
        }
        runtime.vars.nf = runtime.split_line.len() as Int;
    }
    let s = &*(s as *mut Str);
    runtime.split_line.insert(col as usize - 1, s.clone());
}

#[no_mangle]
pub unsafe extern "C" fn str_len(s: *mut c_void) -> usize {
    let s = &*(s as *mut Str);
    let res = s.len();
    mem::forget(s);
    res
}

#[no_mangle]
pub unsafe extern "C" fn concat(s1: *mut c_void, s2: *mut c_void) -> u128 {
    let s1 = &*(s1 as *mut Str);
    let s2 = &*(s2 as *mut Str);
    let res = Str::concat(s1.clone(), s2.clone());
    mem::forget((s1, s2));
    mem::transmute::<Str, u128>(res)
}

// TODO: figure out error story.

#[no_mangle]
pub unsafe extern "C" fn match_pat(runtime: *mut c_void, s: *mut c_void, pat: *mut c_void) -> Int {
    let runtime = runtime as *mut Runtime;
    let s = &*(s as *mut Str);
    let pat = &*(pat as *mut Str);
    let res = match (*runtime).regexes.match_regex(&pat, &s) {
        Ok(res) => res as Int,
        Err(e) => fail!("match_pat: {}", e),
    };
    mem::forget((s, pat));
    res
}

#[no_mangle]
pub unsafe extern "C" fn ref_str(s: *mut c_void) {
    mem::forget((&*(s as *mut Str)).clone())
}

#[no_mangle]
pub unsafe extern "C" fn drop_str(s: *mut c_void) {
    std::ptr::drop_in_place(s as *mut Str);
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
pub unsafe extern "C" fn str_to_int(s: *mut c_void) -> Int {
    let s = &*(s as *mut Str);
    let res = runtime::convert::<&Str, Int>(&s);
    mem::forget(s);
    res
}

#[no_mangle]
pub unsafe extern "C" fn str_to_float(s: *mut c_void) -> Float {
    let s = &*(s as *mut Str);
    let res = runtime::convert::<&Str, Float>(&s);
    mem::forget(s);
    res
}

macro_rules! str_compare_inner {
    ($name:ident, $op:tt) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(s1: *mut c_void, s2: *mut c_void) -> Int {
            let s1 = &*(s1 as *mut Str);
            let s2 = &*(s2 as *mut Str);
            let res = s1.with_str(|s1| s2.with_str(|s2| s1 $op s2)) as Int;
            mem::forget((s1, s2));
            res
        }
    }
}
macro_rules! str_compare {
    ($($name:ident ($op:tt);)*) => { $( str_compare_inner!($name, $op); )* };
}

str_compare! {
    str_lt(<);
    str_gt(>);
    str_lte(<=);
    str_gte(>=);
    str_eq(==);
}
