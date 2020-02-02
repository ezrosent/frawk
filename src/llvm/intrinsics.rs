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
) -> HashMap<&'static str, LLVMValueRef> {
    use llvm::core::*;
    let usize_ty = LLVMIntTypeInContext(ctx, (mem::size_of::<usize>() * 8) as libc::c_uint);
    let int_ty = LLVMIntTypeInContext(ctx, (mem::size_of::<Int>() * 8) as libc::c_uint);
    let float_ty = LLVMDoubleType();
    let void_ty = LLVMVoidType();
    let str_ty = LLVMIntTypeInContext(ctx, (mem::size_of::<Str>() * 8) as libc::c_uint);
    let rt_ty = LLVMPointerType(void_ty, 0);
    let str_ref_ty = rt_ty;
    let mut table: HashMap<&'static str, LLVMValueRef> = Default::default();
    macro_rules! register_inner {
        ($name:ident, [ $($param:expr),* ], $ret:expr) => { {
            // Try and make sure the linker doesn't strip the function out.
            {
                let slice = &[$name];
                let len = slice.len();
                std::ptr::read_volatile(&len);
            }
            let mut params = [$($param),*];
            let ty = LLVMFunctionType($ret, params.as_mut_ptr(), params.len() as u32, 0);
            let func = LLVMAddFunction(module, c_str!(stringify!($name)), ty);
            LLVMSetLinkage(func, llvm::LLVMLinkage::LLVMExternalLinkage);
            if let Some(_) = table.insert(stringify!($name), func) {
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
        read_err(rt_ty, str_ref_ty) -> int_ty;
        read_err_stdin(rt_ty) -> int_ty;
        next_line(rt_ty, str_ref_ty) -> str_ty;
        next_line_stdin(rt_ty) -> str_ty;
        str_lt(str_ref_ty, str_ref_ty) -> int_ty;
        str_gt(str_ref_ty, str_ref_ty) -> int_ty;
        str_lte(str_ref_ty, str_ref_ty) -> int_ty;
        str_gte(str_ref_ty, str_ref_ty) -> int_ty;
        str_eq(str_ref_ty, str_ref_ty) -> int_ty;

        len_intint(usize_ty) -> int_ty;
        lookup_intint(usize_ty, int_ty) -> int_ty;
        contains_intint(usize_ty, int_ty) -> int_ty;
        insert_intint(usize_ty, int_ty, int_ty);
        delete_intint(usize_ty, int_ty);

        len_intfloat(usize_ty) -> int_ty;
        lookup_intfloat(usize_ty, int_ty) -> float_ty;
        contains_intfloat(usize_ty, int_ty) -> int_ty;
        insert_intfloat(usize_ty, int_ty, float_ty);
        delete_intfloat(usize_ty, int_ty);

        len_intstr(usize_ty) -> int_ty;
        lookup_intstr(usize_ty, int_ty) -> str_ty;
        contains_intstr(usize_ty, int_ty) -> int_ty;
        insert_intstr(usize_ty, int_ty, str_ref_ty);
        delete_intstr(usize_ty, int_ty);

        len_strint(usize_ty) -> int_ty;
        lookup_strint(usize_ty, str_ref_ty) -> int_ty;
        contains_strint(usize_ty, str_ref_ty) -> int_ty;
        insert_strint(usize_ty, str_ref_ty, int_ty);
        delete_strint(usize_ty, str_ref_ty);

        len_strfloat(usize_ty) -> int_ty;
        lookup_strfloat(usize_ty, str_ref_ty) -> float_ty;
        contains_strfloat(usize_ty, str_ref_ty) -> int_ty;
        insert_strfloat(usize_ty, str_ref_ty, float_ty);
        delete_strfloat(usize_ty, str_ref_ty);

        len_strstr(usize_ty) -> int_ty;
        lookup_strstr(usize_ty, str_ref_ty) -> str_ty;
        contains_strstr(usize_ty, str_ref_ty) -> int_ty;
        insert_strstr(usize_ty, str_ref_ty, str_ref_ty);
        delete_strstr(usize_ty, str_ref_ty);
    };
    table
}

// TODO: Iterators
// TODO: IO Errors.
//  - we need to exit cleanly. Add a "checkIOerror" builtin to main? set a variable in the runtime
//    and exit cleanly?
//  - get this working along with iterators after everything else is working.
//  - in gawk: redirecting to an output file that fails creates an error; but presumably we want to
//    handle stdout being closed gracefully.

#[no_mangle]
pub unsafe extern "C" fn read_err(runtime: *mut c_void, file: *mut c_void) -> Int {
    let runtime = &mut *(runtime as *mut Runtime);
    let res = match runtime.read_files.read_err(&*(file as *mut Str)) {
        Ok(res) => res,
        Err(e) => fail!("unexpected error when reading error status of file: {}", e),
    };
    res
}

#[no_mangle]
pub unsafe extern "C" fn read_err_stdin(runtime: *mut c_void) -> Int {
    let runtime = &mut *(runtime as *mut Runtime);
    runtime.read_files.read_err_stdin()
}

#[no_mangle]
pub unsafe extern "C" fn next_line_stdin(runtime: *mut c_void) -> u128 {
    let runtime = &mut *(runtime as *mut Runtime);
    match runtime
        .regexes
        .get_line_stdin(&runtime.vars.rs, &mut runtime.read_files)
    {
        Ok(res) => mem::transmute::<Str, u128>(res),
        Err(err) => fail!("unexpected error when reading line from stdin: {}", err),
    }
}

#[no_mangle]
pub unsafe extern "C" fn next_line(runtime: *mut c_void, file: *mut c_void) -> u128 {
    let runtime = &mut *(runtime as *mut Runtime);
    let file = &*(file as *mut Str);
    match runtime
        .regexes
        .get_line(file, &runtime.vars.rs, &mut runtime.read_files)
    {
        Ok(res) => mem::transmute::<Str, u128>(res),
        Err(_) => mem::transmute::<Str, u128>("".into()),
    }
}

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
    str_lt(<); str_gt(>); str_lte(<=); str_gte(>=); str_eq(==);
}

trait InTy {
    type In;
    type Out;
    fn convert_in(x: &Self::In) -> &Self;
    fn convert_out(x: Self) -> Self::Out;
}

impl<'a> InTy for Str<'a> {
    type In = *mut c_void;
    type Out = u128;
    fn convert_in(i: &*mut c_void) -> &Str<'a> {
        unsafe { &*((*i) as *mut Str<'a>) }
    }
    fn convert_out(s: Str<'a>) -> u128 {
        unsafe { mem::transmute::<Str, u128>(s) }
    }
}
impl InTy for Int {
    type In = Int;
    type Out = Int;
    fn convert_in(i: &Int) -> &Int {
        i
    }
    fn convert_out(i: Int) -> Int {
        i
    }
}

impl InTy for Float {
    type In = Float;
    type Out = Float;
    fn convert_in(f: &Float) -> &Float {
        f
    }
    fn convert_out(f: Float) -> Float {
        f
    }
}

macro_rules! map_impl_inner {
    ($lookup:ident, $len:ident, $insert:ident, $delete:ident, $contains:ident, $k:ty, $v:ty) => {
        #[no_mangle]
        pub unsafe extern "C" fn $len(map: usize) -> Int {
            let map = mem::transmute::<usize, runtime::SharedMap<$k, $v>>(map);
            let res = map.len();
            mem::forget(map);
            res as Int
        }
        #[no_mangle]
        pub unsafe extern "C" fn $lookup(map: usize, k: <$k as InTy>::In) -> <$v as InTy>::Out {
            let map = mem::transmute::<usize, runtime::SharedMap<$k, $v>>(map);
            let key = <$k as InTy>::convert_in(&k);
            let res = map.get(key).unwrap_or_else(Default::default);
            mem::forget(map);
            <$v as InTy>::convert_out(res)
        }
        #[no_mangle]
        pub unsafe extern "C" fn $contains(map: usize, k: <$k as InTy>::In) -> Int {
            let map = mem::transmute::<usize, runtime::SharedMap<$k, $v>>(map);
            let key = <$k as InTy>::convert_in(&k);
            let res = map.get(key).is_some() as Int;
            mem::forget(map);
            res
        }
        #[no_mangle]
        pub unsafe extern "C" fn $insert(map: usize, k: <$k as InTy>::In, v: <$v as InTy>::In) {
            let map = mem::transmute::<usize, runtime::SharedMap<$k, $v>>(map);
            let key = <$k as InTy>::convert_in(&k);
            let val = <$v as InTy>::convert_in(&v);
            map.insert(key.clone(), val.clone());
            mem::forget(map);
        }
        #[no_mangle]
        pub unsafe extern "C" fn $delete(map: usize, k: <$k as InTy>::In) {
            let map = mem::transmute::<usize, runtime::SharedMap<$k, $v>>(map);
            let key = <$k as InTy>::convert_in(&k);
            map.delete(key);
            mem::forget(map);
        }
    };
}

macro_rules! map_impl {
    ($($len:ident, $lookup:ident,
       $insert:ident, $delete:ident, $contains:ident, < $k:ty, $v:ty >;)*) => {
        $(
        map_impl_inner!($lookup, $len,$insert,$delete,$contains, $k, $v);
        )*
    }
}

map_impl! {
    len_intint, lookup_intint, insert_intint, delete_intint, contains_intint, <Int, Int>;
    len_intfloat, lookup_intfloat, insert_intfloat, delete_intfloat, contains_intfloat, <Int, Float>;
    len_intstr, lookup_intstr, insert_intstr, delete_intstr, contains_intstr, <Int, Str<'static>>;
    len_strint, lookup_strint, insert_strint, delete_strint, contains_strint, <Str<'static>, Int>;
    len_strfloat, lookup_strfloat, insert_strfloat, delete_strfloat, contains_strfloat, <Str<'static>, Float>;
    len_strstr, lookup_strstr, insert_strstr, delete_strstr, contains_strstr, <Str<'static>, Str<'static>>;
}
