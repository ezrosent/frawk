use super::attr::{self, FunctionAttr};
use crate::builtins::Variable;
use crate::common::Either;
use crate::compile::Ty;
use crate::libc::c_void;
use crate::runtime::{
    self,
    printf::{printf, FormatArg},
    FileRead, FileWrite, Float, Int, IntMap, LazyVec, RegexCache, Str, StrMap, Variables,
};

use hashbrown::HashMap;
use llvm_sys::{
    self,
    prelude::{LLVMContextRef, LLVMModuleRef, LLVMTypeRef, LLVMValueRef},
};
use smallvec;
type SmallVec<T> = smallvec::SmallVec<[T; 4]>;

use std::cell::RefCell;
use std::convert::TryFrom;
use std::mem;
use std::slice;

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

macro_rules! try_abort {
    ($e:expr, $msg:expr) => {
        match $e {
            Ok(res) => res,
            Err(e) => fail!(concat!($msg, " {}"), e),
        }
    };
    ($e:expr) => {
        try_abort!($e, "")
    };
}

macro_rules! exit {
    ($runtime:expr) => {{
        let rt = $runtime as *mut Runtime;
        std::ptr::drop_in_place(rt);
        std::process::exit(0)
    }};
}

pub(crate) struct Runtime<'a> {
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

struct Intrinsic {
    name: *const libc::c_char,
    data: RefCell<Either<LLVMTypeRef, LLVMValueRef>>,
    attrs: smallvec::SmallVec<[FunctionAttr; 1]>,
    _func: *mut c_void,
}

// A map of intrinsics that lazily declares them when they are used in codegen.
pub(crate) struct IntrinsicMap {
    module: LLVMModuleRef,
    ctx: LLVMContextRef,
    map: HashMap<&'static str, Intrinsic>,
}

impl IntrinsicMap {
    fn new(module: LLVMModuleRef, ctx: LLVMContextRef) -> IntrinsicMap {
        IntrinsicMap {
            ctx,
            module,
            map: Default::default(),
        }
    }
    fn register(
        &mut self,
        name: &'static str,
        cname: *const libc::c_char,
        ty: LLVMTypeRef,
        attrs: &[FunctionAttr],
        _func: *mut c_void,
    ) {
        assert!(self
            .map
            .insert(
                name,
                Intrinsic {
                    name: cname,
                    data: RefCell::new(Either::Left(ty)),
                    attrs: attrs.iter().cloned().collect(),
                    _func,
                }
            )
            .is_none())
    }

    pub(crate) unsafe fn get(&self, name: &'static str) -> LLVMValueRef {
        use llvm_sys::core::*;
        let intr = &self.map[name];
        let mut val = intr.data.borrow_mut();

        let ty = match &mut *val {
            Either::Left(ty) => *ty,
            Either::Right(v) => return *v,
        };
        let func = LLVMAddFunction(self.module, intr.name, ty);
        LLVMSetLinkage(func, llvm_sys::LLVMLinkage::LLVMExternalLinkage);
        if intr.attrs.len() > 0 {
            attr::add_function_attrs(self.ctx, func, &intr.attrs[..]);
        }
        *val = Either::Right(func);
        func
    }
}

pub(crate) unsafe fn register(module: LLVMModuleRef, ctx: LLVMContextRef) -> IntrinsicMap {
    use llvm_sys::core::*;
    let int_ty = LLVMIntTypeInContext(ctx, (mem::size_of::<Int>() * 8) as libc::c_uint);
    let float_ty = LLVMDoubleTypeInContext(ctx);
    let void_ty = LLVMVoidTypeInContext(ctx);
    let str_ty = LLVMIntTypeInContext(ctx, (mem::size_of::<Str>() * 8) as libc::c_uint);
    let rt_ty = LLVMPointerType(void_ty, 0);
    let fmt_args_ty = LLVMPointerType(int_ty, 0);
    let fmt_tys_ty = LLVMPointerType(LLVMIntTypeInContext(ctx, 32), 0);
    let map_ty = rt_ty;
    let str_ref_ty = LLVMPointerType(str_ty, 0);
    let iter_int_ty = LLVMPointerType(int_ty, 0);
    let iter_str_ty = LLVMPointerType(str_ty, 0);
    let mut table = IntrinsicMap::new(module, ctx);
    macro_rules! register_inner {
        ($name:ident, [ $($param:expr),* ], [$($attr:tt),*], $ret:expr) => { {
            // Try and make sure the linker doesn't strip the function out.
            let mut params = [$($param),*];
            let ty = LLVMFunctionType($ret, params.as_mut_ptr(), params.len() as u32, 0);
            table.register(
                stringify!($name),
                c_str!(stringify!($name)),
                ty,
                &[$(FunctionAttr::$attr),*],
                $name as *mut c_void,
            );
        }};
    }
    macro_rules! register {
        ($name:ident ($($param:expr),*); $($rest:tt)*) => {
            register!($name($($param),*) -> void_ty; $($rest)*);
        };
        ($name:ident ($($param:expr),*) -> $ret:expr; $($rest:tt)*) => {
            register!([] $name($($param),*) -> $ret; $($rest)*);
        };
        ([$($attr:tt),*] $name:ident ($($param:expr),*) -> $ret:expr; $($rest:tt)*) => {
            register_inner!($name, [ $($param),* ], [$($attr),*], $ret);
            register!($($rest)*);
        };

        () => {};
    }

    register! {
        ref_str(str_ref_ty);
        drop_str(str_ref_ty);
        ref_map(map_ty);
        [ReadOnly] int_to_str(int_ty) -> str_ty;
        [ReadOnly] float_to_str(float_ty) -> str_ty;
        [ReadOnly] str_to_int(str_ref_ty) -> int_ty;
        [ReadOnly] str_to_float(str_ref_ty) -> float_ty;
        [ReadOnly] str_len(str_ref_ty) -> int_ty;
        concat(str_ref_ty, str_ref_ty) -> str_ty;
        [ReadOnly] match_pat(rt_ty, str_ref_ty, str_ref_ty) -> int_ty;
        [ReadOnly] match_pat_loc(rt_ty, str_ref_ty, str_ref_ty) -> int_ty;
        subst_first(rt_ty, str_ref_ty, str_ref_ty, str_ref_ty) -> int_ty;
        subst_all(rt_ty, str_ref_ty, str_ref_ty, str_ref_ty) -> int_ty;
        substr(str_ref_ty, int_ty, int_ty) -> str_ty;
        get_col(rt_ty, int_ty) -> str_ty;
        set_col(rt_ty, int_ty, str_ref_ty);
        split_int(rt_ty, str_ref_ty, map_ty, str_ref_ty) -> int_ty;
        split_str(rt_ty, str_ref_ty, map_ty, str_ref_ty) -> int_ty;
        print_stdout(rt_ty, str_ref_ty);
        print(rt_ty, str_ref_ty, str_ref_ty, int_ty);
        sprintf_impl(str_ref_ty, fmt_args_ty, fmt_tys_ty, int_ty) -> str_ty;
        printf_impl_file(rt_ty, str_ref_ty, fmt_args_ty, fmt_tys_ty, int_ty, str_ref_ty, int_ty);
        printf_impl_stdout(rt_ty, str_ref_ty, fmt_args_ty, fmt_tys_ty, int_ty);
        close_file(rt_ty, str_ref_ty);
        read_err(rt_ty, str_ref_ty) -> int_ty;
        read_err_stdin(rt_ty) -> int_ty;
        next_line(rt_ty, str_ref_ty) -> str_ty;
        next_line_stdin(rt_ty) -> str_ty;

        load_var_str(rt_ty, int_ty) -> str_ty;
        store_var_str(rt_ty, int_ty, str_ref_ty);
        [ReadOnly] load_var_int(rt_ty, int_ty) -> int_ty;
        store_var_int(rt_ty, int_ty, int_ty);
        [ReadOnly] load_var_intmap(rt_ty, int_ty) -> map_ty;
        store_var_intmap(rt_ty, int_ty, map_ty);

        [ReadOnly] str_lt(str_ref_ty, str_ref_ty) -> int_ty;
        [ReadOnly] str_gt(str_ref_ty, str_ref_ty) -> int_ty;
        [ReadOnly] str_lte(str_ref_ty, str_ref_ty) -> int_ty;
        [ReadOnly] str_gte(str_ref_ty, str_ref_ty) -> int_ty;
        [ReadOnly] str_eq(str_ref_ty, str_ref_ty) -> int_ty;

        drop_iter_int(iter_int_ty, int_ty);
        drop_iter_str(iter_str_ty, int_ty);

        alloc_intint() -> map_ty;
        iter_intint(map_ty) -> iter_int_ty;
        [ReadOnly] len_intint(map_ty) -> int_ty;
        [ReadOnly] lookup_intint(map_ty, int_ty) -> int_ty;
        [ReadOnly] contains_intint(map_ty, int_ty) -> int_ty;
        insert_intint(map_ty, int_ty, int_ty);
        delete_intint(map_ty, int_ty);
        drop_intint(map_ty);

        alloc_intfloat() -> map_ty;
        iter_intfloat(map_ty) -> iter_int_ty;
        [ReadOnly] len_intfloat(map_ty) -> int_ty;
        [ReadOnly] lookup_intfloat(map_ty, int_ty) -> float_ty;
        [ReadOnly] contains_intfloat(map_ty, int_ty) -> int_ty;
        insert_intfloat(map_ty, int_ty, float_ty);
        delete_intfloat(map_ty, int_ty);
        drop_intfloat(map_ty);

        alloc_intstr() -> map_ty;
        iter_intstr(map_ty) -> iter_int_ty;
        [ReadOnly] len_intstr(map_ty) -> int_ty;
        [ReadOnly] lookup_intstr(map_ty, int_ty) -> str_ty;
        [ReadOnly] contains_intstr(map_ty, int_ty) -> int_ty;
        insert_intstr(map_ty, int_ty, str_ref_ty);
        delete_intstr(map_ty, int_ty);
        drop_intstr(map_ty);

        alloc_strint() -> map_ty;
        iter_strint(map_ty) -> iter_str_ty;
        [ReadOnly] len_strint(map_ty) -> int_ty;
        [ReadOnly] lookup_strint(map_ty, str_ref_ty) -> int_ty;
        [ReadOnly] contains_strint(map_ty, str_ref_ty) -> int_ty;
        insert_strint(map_ty, str_ref_ty, int_ty);
        delete_strint(map_ty, str_ref_ty);
        drop_strint(map_ty);

        alloc_strfloat() -> map_ty;
        iter_strfloat(map_ty) -> iter_str_ty;
        [ReadOnly] len_strfloat(map_ty) -> int_ty;
        [ReadOnly] lookup_strfloat(map_ty, str_ref_ty) -> float_ty;
        [ReadOnly] contains_strfloat(map_ty, str_ref_ty) -> int_ty;
        insert_strfloat(map_ty, str_ref_ty, float_ty);
        delete_strfloat(map_ty, str_ref_ty);
        drop_strfloat(map_ty);

        alloc_strstr() -> map_ty;
        iter_strstr(map_ty) -> iter_str_ty;
        [ReadOnly] len_strstr(map_ty) -> int_ty;
        [ReadOnly] lookup_strstr(map_ty, str_ref_ty) -> str_ty;
        [ReadOnly] contains_strstr(map_ty, str_ref_ty) -> int_ty;
        insert_strstr(map_ty, str_ref_ty, str_ref_ty);
        delete_strstr(map_ty, str_ref_ty);
        drop_strstr(map_ty);
    };
    table
}

#[no_mangle]
pub unsafe extern "C" fn read_err(runtime: *mut c_void, file: *mut c_void) -> Int {
    let runtime = &mut *(runtime as *mut Runtime);
    let res = try_abort!(
        runtime.read_files.read_err(&*(file as *mut Str)),
        "unexpected error when reading error status of file:"
    );
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
    let res = try_abort!(
        runtime
            .regexes
            .get_line_stdin(&runtime.vars.rs, &mut runtime.read_files),
        "unexpected error when reading line from stdin:"
    );
    mem::transmute::<Str, u128>(res)
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
    if runtime.write_files.write_str_stdout(txt).is_err() {
        exit!(runtime);
    }
    if runtime.write_files.write_str_stdout(&newline).is_err() {
        exit!(runtime);
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
        exit!(runtime);
    }
}

#[no_mangle]
pub unsafe extern "C" fn split_str(
    runtime: *mut c_void,
    to_split: *mut c_void,
    into_arr: *mut c_void,
    pat: *mut c_void,
) -> Int {
    let runtime = &mut *(runtime as *mut Runtime);
    let into_arr = mem::transmute::<*mut c_void, StrMap<Str>>(into_arr);
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
    into_arr: *mut c_void,
    pat: *mut c_void,
) -> Int {
    let runtime = &mut *(runtime as *mut Runtime);
    let into_arr = mem::transmute::<*mut c_void, IntMap<Str>>(into_arr);
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
    runtime.line = runtime.split_line.join(&runtime.vars.ofs);
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
    mem::transmute::<Str, u128>(res)
}

#[no_mangle]
pub unsafe extern "C" fn match_pat(runtime: *mut c_void, s: *mut c_void, pat: *mut c_void) -> Int {
    let runtime = runtime as *mut Runtime;
    let s = &*(s as *mut Str);
    let pat = &*(pat as *mut Str);
    let res = try_abort!((*runtime).regexes.is_regex_match(&pat, &s), "match_pat:");
    mem::forget((s, pat));
    res as Int
}

#[no_mangle]
pub unsafe extern "C" fn match_pat_loc(
    runtime: *mut c_void,
    s: *mut c_void,
    pat: *mut c_void,
) -> Int {
    let runtime = runtime as *mut Runtime;
    let s = &*(s as *mut Str);
    let pat = &*(pat as *mut Str);
    let res = try_abort!(
        (*runtime)
            .regexes
            .regex_match_loc(&mut (*runtime).vars, &pat, &s),
        "match_pat_loc:"
    );
    mem::forget((s, pat));
    res as Int
}

#[no_mangle]
pub unsafe extern "C" fn subst_first(
    runtime: *mut c_void,
    pat: *mut u128,
    s: *mut u128,
    in_s: *mut u128,
) -> Int {
    let runtime = &mut *(runtime as *mut Runtime);
    let s = &*(s as *mut Str);
    let pat = &*(pat as *mut Str);
    let in_s = &mut *(in_s as *mut Str);
    let (subbed, new) = try_abort!(runtime
        .regexes
        .with_regex(pat, |re| in_s.subst_first(re, s)));
    *in_s = subbed;
    new as Int
}

#[no_mangle]
pub unsafe extern "C" fn subst_all(
    runtime: *mut c_void,
    pat: *mut u128,
    s: *mut u128,
    in_s: *mut u128,
) -> Int {
    let runtime = &mut *(runtime as *mut Runtime);
    let s = &mut *(s as *mut Str);
    let pat = &*(pat as *mut Str);
    let in_s = &mut *(in_s as *mut Str);
    let (subbed, nsubs) = try_abort!(runtime.regexes.with_regex(pat, |re| in_s.subst_all(re, s)));
    *in_s = subbed;
    nsubs
}

#[no_mangle]
pub unsafe extern "C" fn substr(base: *mut u128, l: Int, r: Int) -> u128 {
    use std::cmp::{max, min};
    let base = &*(base as *mut Str);
    let len = base.len();
    let l = max(0, l - 1) as usize;
    let r = min(len as Int, r) as usize;
    mem::transmute::<Str, u128>(base.slice(l, r))
}

#[no_mangle]
pub unsafe extern "C" fn ref_str(s: *mut c_void) {
    mem::forget((&*(s as *mut Str)).clone())
}

#[no_mangle]
pub unsafe extern "C" fn drop_str(s: *mut c_void) {
    std::ptr::drop_in_place(s as *mut Str);
}

unsafe fn ref_map_generic<K, V>(m: *mut c_void) {
    mem::forget(mem::transmute::<&*mut c_void, &runtime::SharedMap<K, V>>(&m).clone())
}

unsafe fn drop_map_generic<K, V>(m: *mut c_void) {
    mem::drop(mem::transmute::<*mut c_void, runtime::SharedMap<K, V>>(m))
}

// XXX: relying on this doing the same thing regardless of type. We probably want a custom Rc to
// guarantee this.
//
// XXX: ... how could this ever work?
#[no_mangle]
pub unsafe extern "C" fn ref_map(m: *mut c_void) {
    ref_map_generic::<Int, Str>(m)
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

#[no_mangle]
pub unsafe extern "C" fn load_var_str(rt: *mut c_void, var: usize) -> u128 {
    let rt = &*(rt as *mut Runtime);
    if let Ok(var) = Variable::try_from(var) {
        let res = try_abort!(rt.vars.load_str(var));
        mem::transmute::<Str, u128>(res)
    } else {
        fail!("invalid variable code={}", var)
    }
}

#[no_mangle]
pub unsafe extern "C" fn store_var_str(rt: *mut c_void, var: usize, s: *mut c_void) {
    let rt = &mut *(rt as *mut Runtime);
    if let Ok(var) = Variable::try_from(var) {
        let s = (&*(s as *mut Str)).clone();
        try_abort!(rt.vars.store_str(var, s))
    } else {
        fail!("invalid variable code={}", var)
    }
}

#[no_mangle]
pub unsafe extern "C" fn load_var_int(rt: *mut c_void, var: usize) -> Int {
    let rt = &mut *(rt as *mut Runtime);
    if let Ok(var) = Variable::try_from(var) {
        if var == Variable::NF && rt.split_line.len() == 0 {
            try_abort!(
                rt.regexes
                    .split_regex(&rt.vars.fs, &rt.line, &mut rt.split_line),
                "failed to split line:"
            );
            rt.vars.nf = rt.split_line.len() as Int;
        }
        try_abort!(rt.vars.load_int(var))
    } else {
        fail!("invalid variable code={}", var)
    }
}

#[no_mangle]
pub unsafe extern "C" fn store_var_int(rt: *mut c_void, var: usize, i: Int) {
    let rt = &mut *(rt as *mut Runtime);
    if let Ok(var) = Variable::try_from(var) {
        try_abort!(rt.vars.store_int(var, i));
    } else {
        fail!("invalid variable code={}", var)
    }
}

#[no_mangle]
pub unsafe extern "C" fn load_var_intmap(rt: *mut c_void, var: usize) -> *mut c_void {
    let rt = &*(rt as *mut Runtime);
    if let Ok(var) = Variable::try_from(var) {
        let res = try_abort!(rt.vars.load_intmap(var));
        mem::transmute::<IntMap<_>, *mut c_void>(res)
    } else {
        fail!("invalid variable code={}", var)
    }
}

#[no_mangle]
pub unsafe extern "C" fn store_var_intmap(rt: *mut c_void, var: usize, map: *mut c_void) {
    let rt = &mut *(rt as *mut Runtime);
    if let Ok(var) = Variable::try_from(var) {
        let map = mem::transmute::<*mut c_void, IntMap<Str>>(map);
        try_abort!(rt.vars.store_intmap(var, map.clone()));
        mem::forget(map);
    } else {
        fail!("invalid variable code={}", var)
    }
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

#[no_mangle]
pub unsafe extern "C" fn drop_iter_int(iter: *mut Int, len: usize) {
    mem::drop(Box::from_raw(slice::from_raw_parts_mut(iter, len)))
}

#[no_mangle]
pub unsafe extern "C" fn drop_iter_str(iter: *mut u128, len: usize) {
    let p = iter as *mut Str;
    mem::drop(Box::from_raw(slice::from_raw_parts_mut(p, len)))
}

unsafe fn wrap_args<'a>(args: *mut usize, tys: *mut u32, num_args: Int) -> SmallVec<FormatArg<'a>> {
    let mut format_args = SmallVec::with_capacity(num_args as usize);
    for i in 0..num_args {
        let ty_code = *tys.offset(i as isize);
        let arg = *(args.offset(i as isize));
        let ty = if let Ok(ty) = Ty::try_from(ty_code) {
            ty
        } else {
            fail!("invalid type code passed to printf_impl_file: {}", ty_code)
        };
        let typed_arg: FormatArg = match ty {
            Ty::Int => mem::transmute::<usize, Int>(arg).into(),
            Ty::Float => mem::transmute::<usize, Float>(arg).into(),
            Ty::Str => mem::transmute::<usize, &Str>(arg).clone().into(),
            _ => fail!(
                "invalid format arg {:?} (this should have been caught earlier)",
                ty
            ),
        };
        format_args.push(typed_arg);
    }
    format_args
}

#[no_mangle]
pub unsafe extern "C" fn printf_impl_file(
    rt: *mut c_void,
    spec: *mut u128,
    args: *mut usize,
    tys: *mut u32,
    num_args: Int,
    output: *mut u128,
    append: Int,
) {
    let output_wrapped = Some((&*(output as *mut Str), append != 0));
    let format_args = wrap_args(args, tys, num_args);
    let res = (*(rt as *mut Runtime)).write_files.printf(
        output_wrapped,
        &*(spec as *mut Str),
        &format_args[..],
    );
    if res.is_err() {
        exit!(rt);
    }
}

#[no_mangle]
pub unsafe extern "C" fn sprintf_impl(
    spec: *mut u128,
    args: *mut usize,
    tys: *mut u32,
    num_args: Int,
) -> u128 {
    use runtime::str_impl::DynamicBuf;
    let mut buf = DynamicBuf::new(0);
    let format_args = wrap_args(args, tys, num_args);
    let spec = &*(spec as *mut Str);
    if let Err(e) = spec.with_str(|s| printf(&mut buf, s, &format_args[..])) {
        fail!("unexpected failure during sprintf: {}", e);
    }
    mem::transmute::<Str, u128>(buf.into_str())
}

#[no_mangle]
pub unsafe extern "C" fn printf_impl_stdout(
    rt: *mut c_void,
    spec: *mut u128,
    args: *mut usize,
    tys: *mut u32,
    num_args: Int,
) {
    let format_args = wrap_args(args, tys, num_args);
    let res =
        (*(rt as *mut Runtime))
            .write_files
            .printf(None, &*(spec as *mut Str), &format_args[..]);
    if res.is_err() {
        exit!(rt);
    }
}

#[no_mangle]
pub unsafe extern "C" fn close_file(rt: *mut c_void, file: *mut u128) {
    let rt = &mut *(rt as *mut Runtime);
    let file = &*(file as *mut Str);
    rt.read_files.close(file);
    rt.write_files.close(file);
}

// And now for the shenanigans for implementing map operations. There are 48 functions here; we
// have a bunch of macros to handle type-specific operations. Note: we initially had a trait for
// these operations:
//   pub trait InTy {
//       type In;
//       type Out;
//       fn convert_in(x: &Self::In) -> &Self;
//       fn convert_out(x: Self) -> Self::Out;
//   }
// But that didn't end up working out. We had intrinsic functions with parameter types like
// <Int as InTy>::In, which had strange consequences like not being able to take the address of a
// function. We need to take the address of these functions though,  otherwise the linker on some
// platform will spuriously strip out the symbol.  Instead, we replicate this trait in the form of
// macros that match on the input type.

macro_rules! in_ty {
    (Str) => { *mut c_void };
    (Int) => { Int };
    (Float) => { Float };
}

macro_rules! iter_ty {
    (Str) => { *mut c_void };
    (Int) => { *mut Int };
}

macro_rules! out_ty {
    (Str) => {
        u128
    };
    (Int) => {
        Int
    };
    (Float) => {
        Float
    };
}

macro_rules! convert_in {
    (Str, $e:expr) => {
        &*((*$e) as *mut Str)
    };
    (Int, $e:expr) => {
        $e
    };
    (Float, $e:expr) => {
        $e
    };
}

macro_rules! convert_out {
    (Str, $e:expr) => {
        mem::transmute::<Str, u128>($e)
    };
    (Int, $e:expr) => {
        $e
    };
    (Float, $e:expr) => {
        $e
    };
}
macro_rules! map_impl_inner {
    ($alloc:ident, $iter:ident, $lookup:ident, $len:ident,
     $insert:ident, $delete:ident, $contains:ident, $drop:ident, $k:tt, $v:tt) => {
        // XXX
        // What's going on with the read_volatile(&false) stuff?
        //
        // Put simply, on MacOS the symbols for these functions are stripped out of the function
        // without these lines.
        //
        // Linux seems much more fogiving in this regard. Without these, some tests will fail, only
        // on MacOS, and only in a release build.
        #[no_mangle]
        pub unsafe extern "C" fn $alloc() -> *mut c_void {
            if std::ptr::read_volatile(&false) {
                eprintln!("allocating from {}", stringify!($alloc));
            }
            let res: runtime::SharedMap<$k, $v> = Default::default();
            mem::transmute::<runtime::SharedMap<$k, $v>, *mut c_void>(res)
        }
        #[no_mangle]
        pub unsafe extern "C" fn $iter(map: *mut c_void) -> iter_ty!($k) {
            let map = mem::transmute::<*mut c_void, runtime::SharedMap<$k, $v>>(map);
            let iter: Vec<_> = map.to_vec();
            mem::forget(map);
            let b = iter.into_boxed_slice();
            Box::into_raw(b) as _
        }
        #[no_mangle]
        pub unsafe extern "C" fn $len(map: *mut c_void) -> Int {
            if std::ptr::read_volatile(&false) {
                eprintln!("allocating from {}", stringify!($alloc));
            }
            let map = mem::transmute::<*mut c_void, runtime::SharedMap<$k, $v>>(map);
            let res = map.len();
            mem::forget(map);
            res as Int
        }
        #[no_mangle]
        pub unsafe extern "C" fn $lookup(map: *mut c_void, k: in_ty!($k)) -> out_ty!($v) {
            let map = mem::transmute::<*mut c_void, runtime::SharedMap<$k, $v>>(map);
            let key = convert_in!($k, &k);
            let res = map.get(key).unwrap_or_else(Default::default);
            mem::forget(map);
            convert_out!($v, res)
        }
        #[no_mangle]
        pub unsafe extern "C" fn $contains(map: *mut c_void, k: in_ty!($k)) -> Int {
            let map = mem::transmute::<*mut c_void, runtime::SharedMap<$k, $v>>(map);
            let key = convert_in!($k, &k);
            let res = map.get(key).is_some() as Int;
            mem::forget(map);
            res
        }
        #[no_mangle]
        pub unsafe extern "C" fn $insert(map: *mut c_void, k: in_ty!($k), v: in_ty!($v)) {
            let map = mem::transmute::<*mut c_void, runtime::SharedMap<$k, $v>>(map);
            let key = convert_in!($k, &k);
            let val = convert_in!($v, &v);
            map.insert(key.clone(), val.clone());
            mem::forget(map);
        }
        #[no_mangle]
        pub unsafe extern "C" fn $delete(map: *mut c_void, k: in_ty!($k)) {
            let map = mem::transmute::<*mut c_void, runtime::SharedMap<$k, $v>>(map);
            let key = convert_in!($k, &k);
            map.delete(key);
            mem::forget(map);
        }
        #[no_mangle]
        pub unsafe extern "C" fn $drop(map: *mut c_void) {
            drop_map_generic::<$k, $v>(map)
        }
    };
}

macro_rules! map_impl {
    ($($iter:ident, $alloc:ident, $len:ident, $lookup:ident,
       $insert:ident, $delete:ident, $contains:ident, $drop:ident, < $k:tt, $v:tt >;)*) => {
        $(
            map_impl_inner!(
                $alloc,
                $iter,
                $lookup,
                $len,
                $insert,
                $delete,
                $contains,
                $drop,
                $k,
                $v
            );
        )*
    }
}

map_impl! {
    iter_intint, alloc_intint, len_intint, lookup_intint,
    insert_intint, delete_intint, contains_intint, drop_intint, <Int, Int>;

    iter_intfloat, alloc_intfloat, len_intfloat, lookup_intfloat,
    insert_intfloat, delete_intfloat, contains_intfloat, drop_intfloat, <Int, Float>;

    iter_intstr, alloc_intstr, len_intstr, lookup_intstr,
    insert_intstr, delete_intstr, contains_intstr, drop_intstr, <Int, Str>;

    iter_strint, alloc_strint, len_strint, lookup_strint,
    insert_strint, delete_strint, contains_strint, drop_strint, <Str, Int>;

    iter_strfloat, alloc_strfloat, len_strfloat, lookup_strfloat,
    insert_strfloat, delete_strfloat, contains_strfloat, drop_strfloat, <Str, Float>;

    iter_strstr, alloc_strstr, len_strstr, lookup_strstr,
    insert_strstr, delete_strstr, contains_strstr, drop_strstr, <Str, Str>;
}
