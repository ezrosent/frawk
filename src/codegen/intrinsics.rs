//! This module exposes core runtime functionality to generated code.
//!
//! There is quite a lot of code here at this point, but most of it is "glue". Where possible we
//! try and hew closely to the steps in the `interp` module, with most functionality in the
//! underlying runtime library.
use super::{Backend, FunctionAttr, Sig};
use crate::runtime::{
    self,
    printf::{printf, FormatArg},
    splitter::{
        batch::{ByteReader, CSVReader, WhitespaceOffsets},
        chunk::{ChunkProducer, OffsetChunk},
        regex::RegexSplitter,
    },
    ChainedReader, FileRead, Float, Int, IntMap, Line, LineReader, RegexCache, Str, StrMap,
};
use crate::{
    builtins::Variable,
    common::{CancelSignal, Cleanup, FileSpec, Notification, Result},
    compile::Ty,
    pushdown::FieldSet,
};

use libc::c_void;
use paste::paste;
use rand::{self, Rng};
use regex::bytes::Regex;
use smallvec;

use std::convert::TryFrom;
use std::io;
use std::mem;
use std::slice;

type SmallVec<T> = smallvec::SmallVec<[T; 4]>;

// We don't use u128 as there's a warning that the ABI is not stable. We're in big trouble if our
// version of LLVM disagrees with Rust on the representation of a 128-bit integer, but our tests
// _should_ catch any incompatibilities that arise on that front.
//
// TODO: what steps can we take to move the "should" above to "will"?
#[repr(C)]
pub struct U128(u64, u64);

/// Lazily registers all runtime functions with the given LLVM module and context.
pub(crate) fn register_all(cg: &mut impl Backend) -> Result<()> {
    let int_ty = cg.get_ty(Ty::Int);
    let float_ty = cg.get_ty(Ty::Float);
    let str_ty = cg.get_ty(Ty::Str);
    let rt_ty = cg.void_ptr_ty();
    let fmt_args_ty = cg.ptr_to(int_ty.clone());
    let fmt_tys_ty = cg.ptr_to(cg.u32_ty());
    // we assume that maps are all represented the same
    let map_ty = cg.get_ty(Ty::MapIntInt);
    let str_ref_ty = cg.ptr_to(str_ty.clone());
    let pa_args_ty = cg.ptr_to(str_ref_ty.clone());
    let iter_int_ty = cg.ptr_to(int_ty.clone());
    let iter_str_ty = str_ref_ty.clone();
    macro_rules! register_inner {
        ($name:ident, [ $($param:expr),* ], [$($attr:tt),*], $ret:expr) => {
            cg.register_external_fn(
                stringify!($name),
                c_str!(stringify!($name)) as *const _,
                $name as *const u8,
                Sig {
                    attrs: &[$(FunctionAttr::$attr),*],
                    args: &mut [$($param.clone()),*],
                    ret: $ret,
                }
            )?;
        };
    }
    macro_rules! wrap_ret {
        ([]) => {
            None
        };
        ($ret:tt) => {
            Some($ret.clone())
        };
    }
    macro_rules! register {
        ($name:ident ($($param:expr),*); $($rest:tt)*) => {
            register!($name($($param),*) -> []; $($rest)*);
        };
        ($name:ident ($($param:expr),*) -> $ret:tt; $($rest:tt)*) => {
            register!([] $name($($param),*) -> $ret; $($rest)*);
        };
        ([$($attr:tt),*] $name:ident ($($param:expr),*) -> $ret:tt; $($rest:tt)*) => {
            register_inner!($name, [ $($param),* ], [$($attr),*], wrap_ret!($ret));
            register!($($rest)*);
        };
        () => {};
    }

    register! {
        ref_str(str_ref_ty);
        drop_str(str_ref_ty);
        drop_str_slow(str_ref_ty, int_ty);
        ref_map(map_ty);
        [ReadOnly] int_to_str(int_ty) -> str_ty;
        [ReadOnly] float_to_str(float_ty) -> str_ty;
        [ReadOnly] str_to_int(str_ref_ty) -> int_ty;
        [ReadOnly] hex_str_to_int(str_ref_ty) -> int_ty;
        [ReadOnly] str_to_float(str_ref_ty) -> float_ty;
        [ReadOnly] str_len(str_ref_ty) -> int_ty;
        starts_with_const(str_ref_ty, rt_ty, int_ty) -> int_ty;
        concat(str_ref_ty, str_ref_ty) -> str_ty;
        [ReadOnly] match_pat(rt_ty, str_ref_ty, str_ref_ty) -> int_ty;
        [ReadOnly] match_const_pat(str_ref_ty, rt_ty) -> int_ty;
        [ReadOnly] match_pat_loc(rt_ty, str_ref_ty, str_ref_ty) -> int_ty;
        [ReadOnly] match_const_pat_loc(rt_ty, str_ref_ty, rt_ty) -> int_ty;
        [ReadOnly] substr_index(str_ref_ty, str_ref_ty) -> int_ty;
        subst_first(rt_ty, str_ref_ty, str_ref_ty, str_ref_ty) -> int_ty;
        subst_all(rt_ty, str_ref_ty, str_ref_ty, str_ref_ty) -> int_ty;
        escape_csv(str_ref_ty) -> str_ty;
        escape_tsv(str_ref_ty) -> str_ty;
        substr(str_ref_ty, int_ty, int_ty) -> str_ty;
        [ReadOnly] get_col(rt_ty, int_ty) -> str_ty;
        [ReadOnly] join_csv(rt_ty, int_ty, int_ty) -> str_ty;
        [ReadOnly] join_tsv(rt_ty, int_ty, int_ty) -> str_ty;
        [ReadOnly] join_cols(rt_ty, int_ty, int_ty, str_ref_ty) -> str_ty;
        [ReadOnly] to_upper_ascii(str_ref_ty) -> str_ty;
        [ReadOnly] to_lower_ascii(str_ref_ty) -> str_ty;
        set_col(rt_ty, int_ty, str_ref_ty);
        split_int(rt_ty, str_ref_ty, map_ty, str_ref_ty) -> int_ty;
        split_str(rt_ty, str_ref_ty, map_ty, str_ref_ty) -> int_ty;
        rand_float(rt_ty) -> float_ty;
        seed_rng(rt_ty, int_ty) -> int_ty;
        reseed_rng(rt_ty) -> int_ty;

        exit(rt_ty, int_ty);
        run_system(str_ref_ty) -> int_ty;
        print_all_stdout(rt_ty, pa_args_ty, int_ty);
        print_all_file(rt_ty, pa_args_ty, int_ty, str_ref_ty, int_ty);
        sprintf_impl(rt_ty, str_ref_ty, fmt_args_ty, fmt_tys_ty, int_ty) -> str_ty;
        printf_impl_file(rt_ty, str_ref_ty, fmt_args_ty, fmt_tys_ty, int_ty, str_ref_ty, int_ty);
        printf_impl_stdout(rt_ty, str_ref_ty, fmt_args_ty, fmt_tys_ty, int_ty);
        close_file(rt_ty, str_ref_ty);
        read_err(rt_ty, str_ref_ty, int_ty) -> int_ty;
        read_err_stdin(rt_ty) -> int_ty;
        next_line(rt_ty, str_ref_ty, int_ty) -> str_ty;
        next_line_stdin(rt_ty) -> str_ty;
        next_line_stdin_fused(rt_ty);
        next_file(rt_ty);
        update_used_fields(rt_ty);
        set_fi_entry(rt_ty, int_ty, int_ty);

        // TODO: we are no longer relying on avoiding collisions with exisint library symbols
        // (everything in this module was one no_mangle); we should look into removing the _frawk
        // prefix.

        // Floating-point functions. Note that aside from the last two operations, the LLVM backend
        // uses intrinsics for these, whereas we use standard functions here instead.
        [ReadOnly, ArgmemOnly] _frawk_fprem(float_ty, float_ty) -> float_ty;
        [ReadOnly, ArgmemOnly] _frawk_pow(float_ty, float_ty) -> float_ty;
        [ReadOnly, ArgmemOnly] _frawk_atan(float_ty) -> float_ty;
        [ReadOnly, ArgmemOnly] _frawk_cos(float_ty) -> float_ty;
        [ReadOnly, ArgmemOnly] _frawk_sin(float_ty) -> float_ty;
        [ReadOnly, ArgmemOnly] _frawk_log(float_ty) -> float_ty;
        [ReadOnly, ArgmemOnly] _frawk_log2(float_ty) -> float_ty;
        [ReadOnly, ArgmemOnly] _frawk_log10(float_ty) -> float_ty;
        [ReadOnly, ArgmemOnly] _frawk_exp(float_ty) -> float_ty;
        [ReadOnly, ArgmemOnly] _frawk_atan2(float_ty, float_ty) -> float_ty;

        load_var_str(rt_ty, int_ty) -> str_ty;
        store_var_str(rt_ty, int_ty, str_ref_ty);
        [ReadOnly] load_var_int(rt_ty, int_ty) -> int_ty;
        store_var_int(rt_ty, int_ty, int_ty);
        [ReadOnly] load_var_intmap(rt_ty, int_ty) -> map_ty;
        store_var_intmap(rt_ty, int_ty, map_ty);
        [ReadOnly] load_var_strmap(rt_ty, int_ty) -> map_ty;
        store_var_strmap(rt_ty, int_ty, map_ty);

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
        clear_intint(map_ty);
        drop_intint(map_ty);
        inc_int_intint(map_ty, int_ty, int_ty) -> int_ty;
        inc_float_intint(map_ty, int_ty, float_ty) -> int_ty;

        alloc_intfloat() -> map_ty;
        iter_intfloat(map_ty) -> iter_int_ty;
        [ReadOnly] len_intfloat(map_ty) -> int_ty;
        [ReadOnly] lookup_intfloat(map_ty, int_ty) -> float_ty;
        [ReadOnly] contains_intfloat(map_ty, int_ty) -> int_ty;
        insert_intfloat(map_ty, int_ty, float_ty);
        delete_intfloat(map_ty, int_ty);
        clear_intfloat(map_ty);
        drop_intfloat(map_ty);
        inc_int_intfloat(map_ty, int_ty, int_ty) -> float_ty;
        inc_float_intfloat(map_ty, int_ty, float_ty) -> float_ty;

        alloc_intstr() -> map_ty;
        iter_intstr(map_ty) -> iter_int_ty;
        [ReadOnly] len_intstr(map_ty) -> int_ty;
        [ReadOnly] lookup_intstr(map_ty, int_ty) -> str_ty;
        [ReadOnly] contains_intstr(map_ty, int_ty) -> int_ty;
        insert_intstr(map_ty, int_ty, str_ref_ty);
        delete_intstr(map_ty, int_ty);
        clear_intstr(map_ty);
        drop_intstr(map_ty);
        inc_int_intstr(map_ty, int_ty, int_ty) -> str_ty;
        inc_float_intstr(map_ty, int_ty, float_ty) -> str_ty;

        alloc_strint() -> map_ty;
        iter_strint(map_ty) -> iter_str_ty;
        [ReadOnly] len_strint(map_ty) -> int_ty;
        [ReadOnly] lookup_strint(map_ty, str_ref_ty) -> int_ty;
        [ReadOnly] contains_strint(map_ty, str_ref_ty) -> int_ty;
        insert_strint(map_ty, str_ref_ty, int_ty);
        delete_strint(map_ty, str_ref_ty);
        clear_strint(map_ty);
        drop_strint(map_ty);
        inc_int_strint(map_ty, str_ref_ty, int_ty) -> int_ty;
        inc_float_strint(map_ty, str_ref_ty, float_ty) -> int_ty;

        alloc_strfloat() -> map_ty;
        iter_strfloat(map_ty) -> iter_str_ty;
        [ReadOnly] len_strfloat(map_ty) -> int_ty;
        [ReadOnly] lookup_strfloat(map_ty, str_ref_ty) -> float_ty;
        [ReadOnly] contains_strfloat(map_ty, str_ref_ty) -> int_ty;
        insert_strfloat(map_ty, str_ref_ty, float_ty);
        delete_strfloat(map_ty, str_ref_ty);
        clear_strfloat(map_ty);
        drop_strfloat(map_ty);
        inc_int_strfloat(map_ty, str_ref_ty, int_ty) -> float_ty;
        inc_float_strfloat(map_ty, str_ref_ty, float_ty) -> float_ty;

        alloc_strstr() -> map_ty;
        iter_strstr(map_ty) -> iter_str_ty;
        [ReadOnly] len_strstr(map_ty) -> int_ty;
        [ReadOnly] lookup_strstr(map_ty, str_ref_ty) -> str_ty;
        [ReadOnly] contains_strstr(map_ty, str_ref_ty) -> int_ty;
        insert_strstr(map_ty, str_ref_ty, str_ref_ty);
        delete_strstr(map_ty, str_ref_ty);
        clear_strstr(map_ty);
        drop_strstr(map_ty);
        inc_int_strstr(map_ty, str_ref_ty, int_ty) -> str_ty;
        inc_float_strstr(map_ty, str_ref_ty, float_ty) -> str_ty;

        load_slot_int(rt_ty, int_ty) -> int_ty;
        load_slot_float(rt_ty, int_ty) -> float_ty;
        load_slot_str(rt_ty, int_ty) -> str_ty;
        load_slot_intint(rt_ty, int_ty) -> map_ty;
        load_slot_intfloat(rt_ty, int_ty) -> map_ty;
        load_slot_intstr(rt_ty, int_ty) -> map_ty;
        load_slot_strint(rt_ty, int_ty) -> map_ty;
        load_slot_strfloat(rt_ty, int_ty) -> map_ty;
        load_slot_strstr(rt_ty, int_ty) -> map_ty;

        store_slot_int(rt_ty, int_ty, int_ty);
        store_slot_float(rt_ty, int_ty, float_ty);
        store_slot_str(rt_ty, int_ty, str_ref_ty);
        store_slot_intint(rt_ty, int_ty, map_ty);
        store_slot_intfloat(rt_ty, int_ty, map_ty);
        store_slot_intstr(rt_ty, int_ty, map_ty);
        store_slot_strint(rt_ty, int_ty, map_ty);
        store_slot_strfloat(rt_ty, int_ty, map_ty);
        store_slot_strstr(rt_ty, int_ty, map_ty);
    };
    Ok(())
}

macro_rules! fail {
    ($rt:expr, $($es:expr),+) => {{
        #[cfg(test)]
        {
            eprintln_ignore!("failure in runtime {}. Halting execution", format!($($es),*));
            panic!("failure in runtime")
        }
        #[cfg(not(test))]
        {
            eprintln_ignore!("failure in runtime {}. Halting execution", format!($($es),*));
            exit!($rt, 1)
        }
    }}
}

macro_rules! try_abort {
    ($rt:expr, $e:expr, $msg:expr) => {
        match $e {
            Ok(res) => res,
            Err(e) => fail!($rt, concat!($msg, " {}"), e),
        }
    };
    ($rt:expr, $e:expr) => {
        try_abort!($rt, $e, "")
    };
}

// we use a "silent" abort for write errors to play nicely with unix tools like "head" which
// deliberately close pipes prematurely.
macro_rules! try_silent_abort {
    ($rt:expr, $e:expr) => {
        match $e {
            Ok(res) => res,
            Err(_) => exit!($rt),
        }
    };
}

macro_rules! exit {
    ($runtime:expr) => {
        exit!($runtime, 0)
    };
    ($runtime:expr, $code:expr) => {{
        // XXX: revisit if this is undefined behavior having &mut Runtime in scope even if it is
        // not used after the call to drop_in_place.
        let rt_raw = $runtime as *mut Runtime;
        let rt = &mut *rt_raw;
        let code = $code;
        if rt.concurrent {
            let pid = rt.core.vars.pid;
            rt.cancel_signal.cancel(code);
            std::ptr::drop_in_place(rt_raw);
            if pid == 1 {
                // We are the main thread. Drop on `rt` should have waited for other threads to exit.
                // All that's left is for us to abort.
                std::process::exit(code)
            } else {
                // Block forever. Let the main thread exit.
                let n = Notification::default();
                n.wait();
                unreachable!()
            }
        } else {
            std::ptr::drop_in_place(rt_raw);
            std::process::exit(code)
        }
    }};
}

macro_rules! with_input {
    ($inp:expr, |$p:pat| $body:expr) => {
        match $inp {
            $crate::codegen::intrinsics::InputData::V1($p) => $body,
            $crate::codegen::intrinsics::InputData::V2($p) => $body,
            $crate::codegen::intrinsics::InputData::V3($p) => $body,
            $crate::codegen::intrinsics::InputData::V4($p) => $body,
        }
    };
}

pub(crate) type InputTuple<LR> = (<LR as LineReader>::Line, FileRead<LR>);
pub(crate) enum InputData {
    V1(InputTuple<CSVReader<Box<dyn ChunkProducer<Chunk = OffsetChunk>>>>),
    V2(InputTuple<ByteReader<Box<dyn ChunkProducer<Chunk = OffsetChunk<WhitespaceOffsets>>>>>),
    V3(InputTuple<ByteReader<Box<dyn ChunkProducer<Chunk = OffsetChunk>>>>),
    V4(InputTuple<ChainedReader<RegexSplitter<Box<dyn io::Read + Send>>>>),
}

pub(crate) trait IntoRuntime {
    fn into_runtime<'a>(
        self,
        ff: impl runtime::writers::FileFactory,
        used_fields: &FieldSet,
        named_columns: Option<Vec<&[u8]>>,
        cancel_signal: CancelSignal,
    ) -> Runtime<'a>;
}

macro_rules! impl_into_runtime {
    ($ty:ty, $var:tt) => {
        impl IntoRuntime for $ty {
            fn into_runtime<'a>(
                self,
                ff: impl runtime::writers::FileFactory,
                used_fields: &FieldSet,
                named_columns: Option<Vec<&[u8]>>,
                cancel_signal: CancelSignal,
            ) -> Runtime<'a> {
                Runtime {
                    concurrent: false,
                    input_data: InputData::$var((
                        Default::default(),
                        FileRead::new(self, used_fields.clone(), named_columns),
                    )),
                    core: crate::interp::Core::new(ff),
                    cleanup: Cleanup::null(),
                    cancel_signal,
                }
            }
        }

        impl From<FileRead<$ty>> for InputData {
            fn from(v: FileRead<$ty>) -> InputData {
                InputData::$var((Default::default(), v))
            }
        }
    };
}

impl_into_runtime!(CSVReader<Box<dyn ChunkProducer<Chunk = OffsetChunk>>>, V1);
impl_into_runtime!(
    ByteReader<Box<dyn ChunkProducer<Chunk = OffsetChunk<WhitespaceOffsets>>>>,
    V2
);
impl_into_runtime!(ByteReader<Box<dyn ChunkProducer<Chunk = OffsetChunk>>>, V3);
impl_into_runtime!(ChainedReader<RegexSplitter<Box<dyn io::Read + Send>>>, V4);

pub(crate) struct Runtime<'a> {
    pub(crate) core: crate::interp::Core<'a>,
    pub(crate) input_data: InputData,
    #[allow(unused)]
    pub(crate) concurrent: bool,
    pub(crate) cancel_signal: CancelSignal,
    pub(crate) cleanup: Cleanup<Self>,
}

impl<'a> Runtime<'a> {
    fn reset_file_vars(&mut self) {
        self.core.vars.fnr = 0;
        self.core.vars.filename = with_input!(&mut self.input_data, |(_, read_files)| {
            read_files.stdin_filename().upcast()
        });
    }
}

impl<'a> Drop for Runtime<'a> {
    fn drop(&mut self) {
        mem::replace(&mut self.cleanup, Cleanup::null()).invoke(self);
    }
}

pub(crate) unsafe extern "C" fn exit(runtime: *mut c_void, code: Int) {
    exit!(runtime, code as i32);
}

pub(crate) unsafe extern "C" fn run_system(cmd: *mut U128) -> Int {
    let s: &Str = &*(cmd as *mut Str);
    s.with_bytes(runtime::run_command)
}

pub(crate) unsafe extern "C" fn rand_float(runtime: *mut c_void) -> f64 {
    let runtime = &mut *(runtime as *mut Runtime);
    runtime.core.rng.gen_range(0.0..=1.0)
}

pub(crate) unsafe extern "C" fn seed_rng(runtime: *mut c_void, seed: Int) -> Int {
    let runtime = &mut *(runtime as *mut Runtime);
    runtime.core.reseed(seed as u64) as Int
}

pub(crate) unsafe extern "C" fn reseed_rng(runtime: *mut c_void) -> Int {
    let runtime = &mut *(runtime as *mut Runtime);
    runtime.core.reseed_random() as Int
}

pub(crate) unsafe extern "C" fn read_err(
    runtime: *mut c_void,
    file: *mut c_void,
    is_file: Int,
) -> Int {
    let runtime = &mut *(runtime as *mut Runtime);
    let res = try_abort!(
        runtime,
        with_input!(&mut runtime.input_data, |(_, read_files)| {
            let file = &*(file as *mut Str);
            if is_file == 0 {
                read_files.read_err_cmd(file)
            } else {
                read_files.read_err(file)
            }
        }),
        "unexpected error when reading error status of file:"
    );
    res
}

pub(crate) unsafe extern "C" fn read_err_stdin(runtime: *mut c_void) -> Int {
    let runtime = &mut *(runtime as *mut Runtime);
    with_input!(&mut runtime.input_data, |(_, read_files)| read_files
        .read_err_stdin())
}

pub(crate) unsafe extern "C" fn next_line_stdin_fused(runtime: *mut c_void) {
    let runtime = &mut *(runtime as *mut Runtime);
    let changed = try_abort!(
        runtime,
        with_input!(&mut runtime.input_data, |(line, read_files)| {
            runtime
                .core
                .regexes
                .get_line_stdin_reuse(&runtime.core.vars.rs, read_files, line)
        }),
        "unexpected error when reading line from stdin:"
    );
    if changed {
        runtime.reset_file_vars();
    }
}

pub(crate) unsafe extern "C" fn next_file(runtime: *mut c_void) {
    let runtime = &mut *(runtime as *mut Runtime);
    try_abort!(
        runtime,
        with_input!(&mut runtime.input_data, |(_, read_files)| {
            read_files.next_file()
        })
    );
}

pub(crate) unsafe extern "C" fn next_line_stdin(runtime: *mut c_void) -> U128 {
    let runtime = &mut *(runtime as *mut Runtime);
    let (changed, res) = try_abort!(
        runtime,
        with_input!(&mut runtime.input_data, |(_, read_files)| {
            runtime
                .core
                .regexes
                .get_line_stdin(&runtime.core.vars.rs, read_files)
        }),
        "unexpected error when reading line from stdin:"
    );
    if changed {
        runtime.reset_file_vars();
    }
    mem::transmute::<Str, U128>(res)
}

pub(crate) unsafe extern "C" fn next_line(
    runtime: *mut c_void,
    file: *mut c_void,
    is_file: Int,
) -> U128 {
    let runtime = &mut *(runtime as *mut Runtime);
    let file = &*(file as *mut Str);
    let res = with_input!(&mut runtime.input_data, |(_, read_files)| {
        runtime
            .core
            .regexes
            .get_line(file, &runtime.core.vars.rs, read_files, is_file != 0)
    });
    match res {
        Ok(res) => mem::transmute::<Str, U128>(res),
        Err(_) => mem::transmute::<Str, U128>("".into()),
    }
}

pub(crate) unsafe extern "C" fn update_used_fields(runtime: *mut c_void) {
    let runtime = &mut *(runtime as *mut Runtime);
    let fi = &runtime.core.vars.fi;
    with_input!(&mut runtime.input_data, |(_, read_files)| {
        read_files.update_named_columns(fi);
    });
}

pub(crate) unsafe extern "C" fn set_fi_entry(runtime: *mut c_void, key: Int, val: Int) {
    let rt = &mut *(runtime as *mut Runtime);
    let fi = &rt.core.vars.fi;
    let k = mem::transmute::<U128, Str>(get_col(runtime, key));
    fi.insert(k, val);
}

pub(crate) unsafe extern "C" fn split_str(
    runtime: *mut c_void,
    to_split: *mut c_void,
    into_arr: *mut c_void,
    pat: *mut c_void,
) -> Int {
    let runtime = &mut *(runtime as *mut Runtime);
    let into_arr = mem::transmute::<*mut c_void, StrMap<Str>>(into_arr);
    let to_split = &*(to_split as *mut Str);
    let pat = &*(pat as *mut Str);
    if let Err(e) = runtime
        .core
        .regexes
        .split_regex_strmap(pat, to_split, &into_arr)
    {
        fail!(runtime, "failed to split string: {}", e);
    }
    let res = into_arr.len() as Int;
    mem::forget((into_arr, to_split, pat));
    res
}

pub(crate) unsafe extern "C" fn split_int(
    runtime: *mut c_void,
    to_split: *mut c_void,
    into_arr: *mut c_void,
    pat: *mut c_void,
) -> Int {
    let runtime = &mut *(runtime as *mut Runtime);
    let into_arr = mem::transmute::<*mut c_void, IntMap<Str>>(into_arr);
    let to_split = &*(to_split as *mut Str);
    let pat = &*(pat as *mut Str);
    if let Err(e) = runtime
        .core
        .regexes
        .split_regex_intmap(&pat, &to_split, &into_arr)
    {
        fail!(runtime, "failed to split string: {}", e);
    }
    let res = into_arr.len() as Int;
    mem::forget((into_arr, to_split, pat));
    res
}

pub(crate) unsafe extern "C" fn get_col(runtime: *mut c_void, col: Int) -> U128 {
    let runtime = &mut *(runtime as *mut Runtime);
    let col_str = with_input!(&mut runtime.input_data, |(line, _)| {
        line.get_col(
            col,
            &runtime.core.vars.fs,
            &runtime.core.vars.ofs,
            &mut runtime.core.regexes,
        )
    });
    let res = match col_str {
        Ok(s) => s,
        Err(e) => fail!(runtime, "get_col: {}", e),
    };
    mem::transmute::<Str, U128>(res)
}

pub(crate) unsafe extern "C" fn join_csv(runtime: *mut c_void, start: Int, end: Int) -> U128 {
    let sep: Str<'static> = ",".into();
    let runtime = &mut *(runtime as *mut Runtime);
    let res = try_abort!(
        runtime,
        with_input!(&mut runtime.input_data, |(line, _)| {
            let nf = try_abort!(
                runtime,
                line.nf(&runtime.core.vars.fs, &mut runtime.core.regexes),
                "nf:"
            );
            line.join_cols(start, end, &sep, nf, |s| runtime::escape_csv(&s))
        }),
        "join_csv:"
    );
    mem::transmute::<Str, U128>(res)
}

pub(crate) unsafe extern "C" fn join_tsv(runtime: *mut c_void, start: Int, end: Int) -> U128 {
    let sep: Str<'static> = "\t".into();
    let runtime = &mut *(runtime as *mut Runtime);
    let res = try_abort!(
        runtime,
        with_input!(&mut runtime.input_data, |(line, _)| {
            let nf = try_abort!(
                runtime,
                line.nf(&runtime.core.vars.fs, &mut runtime.core.regexes),
                "nf:"
            );
            line.join_cols(start, end, &sep, nf, |s| runtime::escape_tsv(&s))
        }),
        "join_tsv:"
    );
    mem::transmute::<Str, U128>(res)
}

pub(crate) unsafe extern "C" fn join_cols(
    runtime: *mut c_void,
    start: Int,
    end: Int,
    sep: *mut U128,
) -> U128 {
    let runtime = &mut *(runtime as *mut Runtime);
    let res = try_abort!(
        runtime,
        with_input!(&mut runtime.input_data, |(line, _)| {
            let nf = try_abort!(
                runtime,
                line.nf(&runtime.core.vars.fs, &mut runtime.core.regexes),
                "nf:"
            );
            line.join_cols(start, end, &*(sep as *mut Str), nf, |s| s)
        }),
        "join_cols:"
    );
    mem::transmute::<Str, U128>(res)
}

pub(crate) unsafe extern "C" fn to_upper_ascii(s: *mut U128) -> U128 {
    let res = (&*(s as *mut Str as *const Str)).to_upper_ascii();
    mem::transmute::<Str, U128>(res)
}

pub(crate) unsafe extern "C" fn to_lower_ascii(s: *mut U128) -> U128 {
    let res = (&*(s as *mut Str as *const Str)).to_lower_ascii();
    mem::transmute::<Str, U128>(res)
}

pub(crate) unsafe extern "C" fn set_col(runtime: *mut c_void, col: Int, s: *mut c_void) {
    let runtime = &mut *(runtime as *mut Runtime);
    let s = &*(s as *mut Str);
    if let Err(e) = with_input!(&mut runtime.input_data, |(line, _)| line.set_col(
        col,
        s,
        &runtime.core.vars.ofs,
        &mut runtime.core.regexes,
    )) {
        fail!(runtime, "set_col: {}", e);
    }
}

pub(crate) unsafe extern "C" fn str_len(s: *mut c_void) -> usize {
    let s = &*(s as *mut Str);
    s.len()
}

pub(crate) unsafe extern "C" fn starts_with_const(
    s1: *mut c_void,
    base: *const u8,
    len: Int,
) -> Int {
    debug_assert!(len >= 0);
    let other = slice::from_raw_parts(base, len as usize);
    let s1 = &*(s1 as *const Str);
    let s1_bytes = &*s1.get_bytes();
    ((s1_bytes.len() >= other.len()) && &s1_bytes[..other.len()] == other) as Int
}

pub(crate) unsafe extern "C" fn concat(s1: *mut c_void, s2: *mut c_void) -> U128 {
    let s1 = &*(s1 as *mut Str);
    let s2 = &*(s2 as *mut Str);
    let res = Str::concat(s1.clone(), s2.clone());
    mem::transmute::<Str, U128>(res)
}

pub(crate) unsafe extern "C" fn match_pat(
    runtime: *mut c_void,
    s: *mut c_void,
    pat: *mut c_void,
) -> Int {
    let runtime = runtime as *mut Runtime;
    let s = &*(s as *mut Str);
    let pat = &*(pat as *mut Str);
    let res = try_abort!(
        runtime,
        (*runtime).core.is_match_regex(s, pat),
        "match_pat:"
    );
    res as Int
}

pub(crate) unsafe extern "C" fn match_const_pat(s: *mut c_void, pat: *mut c_void) -> Int {
    let s = &*(s as *mut Str);
    let pat = &*(pat as *const Regex);
    RegexCache::regex_const_match(pat, s) as Int
}

pub(crate) unsafe extern "C" fn match_pat_loc(
    runtime: *mut c_void,
    s: *mut c_void,
    pat: *mut c_void,
) -> Int {
    let runtime = runtime as *mut Runtime;
    let s = &*(s as *mut Str);
    let pat = &*(pat as *mut Str);
    let res = try_abort!(
        runtime,
        (*runtime).core.match_regex(s, pat),
        "match_pat_loc:"
    );
    res as Int
}

pub(crate) unsafe extern "C" fn match_const_pat_loc(
    runtime: *mut c_void,
    s: *mut c_void,
    pat: *mut c_void,
) -> Int {
    let runtime = runtime as *mut Runtime;
    let s = &*(s as *mut Str);
    let pat = &*(pat as *const Regex);
    try_abort!(
        runtime,
        (*runtime).core.match_const_regex(s, pat),
        "match_const_pat_loc:"
    )
}

pub(crate) unsafe extern "C" fn substr_index(s: *mut U128, t: *mut U128) -> Int {
    let s = &*(s as *mut Str);
    let t = &*(t as *mut Str);
    runtime::string_search::index_substr(/*needle*/ t, /*haystack*/ s)
}

pub(crate) unsafe extern "C" fn subst_first(
    runtime: *mut c_void,
    pat: *mut U128,
    s: *mut U128,
    in_s: *mut U128,
) -> Int {
    let runtime = &mut *(runtime as *mut Runtime);
    let s = &*(s as *mut Str);
    let pat = &*(pat as *mut Str);
    let in_s = &mut *(in_s as *mut Str);
    let (subbed, new) = try_abort!(
        runtime,
        runtime
            .core
            .regexes
            .with_regex(pat, |re| in_s.subst_first(re, s))
    );
    *in_s = subbed;
    new as Int
}

pub(crate) unsafe extern "C" fn subst_all(
    runtime: *mut c_void,
    pat: *mut U128,
    s: *mut U128,
    in_s: *mut U128,
) -> Int {
    let runtime = &mut *(runtime as *mut Runtime);
    let s = &mut *(s as *mut Str);
    let pat = &*(pat as *mut Str);
    let in_s = &mut *(in_s as *mut Str);
    let (subbed, nsubs) = try_abort!(
        runtime,
        runtime
            .core
            .regexes
            .with_regex(pat, |re| in_s.subst_all(re, s))
    );
    *in_s = subbed;
    nsubs
}

pub(crate) unsafe extern "C" fn escape_csv(s: *mut U128) -> U128 {
    mem::transmute::<Str, U128>(runtime::escape_csv(&*(s as *mut Str)))
}

pub(crate) unsafe extern "C" fn escape_tsv(s: *mut U128) -> U128 {
    mem::transmute::<Str, U128>(runtime::escape_tsv(&*(s as *mut Str)))
}

pub(crate) unsafe extern "C" fn substr(base: *mut U128, l: Int, r: Int) -> U128 {
    use std::cmp::{max, min};
    let base = &*(base as *mut Str);
    let len = base.len();
    let l = max(0, l - 1);
    if l as usize >= len {
        mem::transmute::<Str, U128>(Str::default())
    } else {
        let r = min(len as Int, l.saturating_add(r)) as usize;
        mem::transmute::<Str, U128>(base.slice(l as usize, r))
    }
}

pub(crate) unsafe extern "C" fn ref_str(s: *mut c_void) {
    mem::forget((&*(s as *mut Str)).clone())
}

// This is a "slow path" drop, used by cranelift only for the time being.
pub(crate) unsafe extern "C" fn drop_str(s: *mut U128) {
    std::ptr::drop_in_place(s as *mut Str)
}

pub(crate) unsafe extern "C" fn drop_str_slow(s: *mut U128, tag: u64) {
    (&*(s as *mut Str)).drop_with_tag(tag)
}

unsafe fn ref_map_generic<K, V>(m: *mut c_void) {
    mem::forget(mem::transmute::<&*mut c_void, &runtime::SharedMap<K, V>>(&m).clone())
}

unsafe fn drop_map_generic<K, V>(m: *mut c_void) {
    let map_ref = mem::transmute::<*mut c_void, runtime::SharedMap<K, V>>(m);
    debug_assert!(std::rc::Rc::strong_count(&map_ref.0) > 0);
    mem::drop(map_ref)
}

// XXX: relying on this doing the same thing regardless of type. We probably want a custom Rc to
// guarantee this.

pub(crate) unsafe extern "C" fn ref_map(m: *mut c_void) {
    ref_map_generic::<Int, Str>(m)
}

pub(crate) unsafe extern "C" fn int_to_str(i: Int) -> U128 {
    mem::transmute::<Str, U128>(runtime::convert::<Int, Str>(i))
}

pub(crate) unsafe extern "C" fn float_to_str(f: Float) -> U128 {
    mem::transmute::<Str, U128>(runtime::convert::<Float, Str>(f))
}

pub(crate) unsafe extern "C" fn str_to_int(s: *mut c_void) -> Int {
    let s = &*(s as *mut Str);
    runtime::convert::<&Str, Int>(s)
}

pub(crate) unsafe extern "C" fn hex_str_to_int(s: *mut c_void) -> Int {
    let s = &*(s as *mut Str);
    s.with_bytes(runtime::hextoi)
}

pub(crate) unsafe extern "C" fn str_to_float(s: *mut c_void) -> Float {
    let s = &*(s as *mut Str);
    runtime::convert::<&Str, Float>(s)
}

pub(crate) unsafe extern "C" fn load_var_str(rt: *mut c_void, var: usize) -> U128 {
    let runtime = &mut *(rt as *mut Runtime);
    if let Ok(var) = Variable::try_from(var) {
        let res = try_abort!(runtime, runtime.core.vars.load_str(var));
        mem::transmute::<Str, U128>(res)
    } else {
        fail!(runtime, "invalid variable code={}", var)
    }
}

pub(crate) unsafe extern "C" fn store_var_str(rt: *mut c_void, var: usize, s: *mut c_void) {
    let runtime = &mut *(rt as *mut Runtime);
    if let Ok(var) = Variable::try_from(var) {
        let s = (&*(s as *mut Str)).clone();
        try_abort!(runtime, runtime.core.vars.store_str(var, s))
    } else {
        fail!(runtime, "invalid variable code={}", var)
    }
}

pub(crate) unsafe extern "C" fn load_var_int(rt: *mut c_void, var: usize) -> Int {
    let runtime = &mut *(rt as *mut Runtime);
    if let Ok(var) = Variable::try_from(var) {
        if let Variable::NF = var {
            runtime.core.vars.nf = match with_input!(&mut runtime.input_data, |(line, _)| line
                .nf(&runtime.core.vars.fs, &mut runtime.core.regexes))
            {
                Ok(nf) => nf as Int,
                Err(e) => fail!(runtime, "nf: {}", e),
            };
        }
        try_abort!(runtime, runtime.core.vars.load_int(var))
    } else {
        fail!(runtime, "invalid variable code={}", var)
    }
}

pub(crate) unsafe extern "C" fn store_var_int(rt: *mut c_void, var: usize, i: Int) {
    let runtime = &mut *(rt as *mut Runtime);
    if let Ok(var) = Variable::try_from(var) {
        try_abort!(runtime, runtime.core.vars.store_int(var, i));
    } else {
        fail!(runtime, "invalid variable code={}", var)
    }
}

pub(crate) unsafe extern "C" fn load_var_intmap(rt: *mut c_void, var: usize) -> *mut c_void {
    let runtime = &mut *(rt as *mut Runtime);
    if let Ok(var) = Variable::try_from(var) {
        let res = try_abort!(runtime, runtime.core.vars.load_intmap(var));
        mem::transmute::<IntMap<_>, *mut c_void>(res)
    } else {
        fail!(runtime, "invalid variable code={}", var)
    }
}

pub(crate) unsafe extern "C" fn store_var_intmap(rt: *mut c_void, var: usize, map: *mut c_void) {
    let runtime = &mut *(rt as *mut Runtime);
    if let Ok(var) = Variable::try_from(var) {
        let map = mem::transmute::<*mut c_void, IntMap<Str>>(map);
        try_abort!(runtime, runtime.core.vars.store_intmap(var, map.clone()));
        mem::forget(map);
    } else {
        fail!(runtime, "invalid variable code={}", var)
    }
}

pub(crate) unsafe extern "C" fn load_var_strmap(rt: *mut c_void, var: usize) -> *mut c_void {
    let runtime = &mut *(rt as *mut Runtime);
    if let Ok(var) = Variable::try_from(var) {
        let res = try_abort!(runtime, runtime.core.vars.load_strmap(var));
        mem::transmute::<StrMap<_>, *mut c_void>(res)
    } else {
        fail!(runtime, "invalid variable code={}", var)
    }
}

pub(crate) unsafe extern "C" fn store_var_strmap(rt: *mut c_void, var: usize, map: *mut c_void) {
    let runtime = &mut *(rt as *mut Runtime);
    if let Ok(var) = Variable::try_from(var) {
        let map = mem::transmute::<*mut c_void, StrMap<Int>>(map);
        try_abort!(runtime, runtime.core.vars.store_strmap(var, map.clone()));
        mem::forget(map);
    } else {
        fail!(runtime, "invalid variable code={}", var)
    }
}

macro_rules! str_compare_inner {
    ($name:ident, $op:tt) => {
        pub(crate) unsafe extern "C" fn $name(s1: *mut c_void, s2: *mut c_void) -> Int {
            let s1 = &*(s1 as *mut Str);
            let s2 = &*(s2 as *mut Str);
            let res = s1.with_bytes(|bs1| s2.with_bytes(|bs2| bs1 $op bs2)) as Int;
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

pub(crate) unsafe extern "C" fn drop_iter_int(iter: *mut Int, len: usize) {
    mem::drop(Box::from_raw(slice::from_raw_parts_mut(iter, len)))
}

pub(crate) unsafe extern "C" fn drop_iter_str(iter: *mut U128, len: usize) {
    let p = iter as *mut Str;
    mem::drop(Box::from_raw(slice::from_raw_parts_mut(p, len)))
}

unsafe fn wrap_args<'a>(
    _rt: &mut Runtime<'a>,
    args: *mut usize,
    tys: *mut u32,
    num_args: Int,
) -> SmallVec<FormatArg<'a>> {
    let mut format_args = SmallVec::with_capacity(num_args as usize);
    for i in 0..num_args {
        let ty_code = *tys.offset(i as isize);
        let arg = *(args.offset(i as isize));
        let ty = if let Ok(ty) = Ty::try_from(ty_code) {
            ty
        } else {
            fail!(
                _rt,
                "invalid type code passed to printf_impl_file: {}",
                ty_code
            )
        };
        let typed_arg: FormatArg = match ty {
            Ty::Int => mem::transmute::<usize, Int>(arg).into(),
            Ty::Float => Float::from_bits(arg as u64).into(),
            Ty::Str => mem::transmute::<usize, &Str>(arg).clone().into(),
            _ => fail!(
                _rt,
                "invalid format arg {:?} (this should have been caught earlier)",
                ty
            ),
        };
        format_args.push(typed_arg);
    }
    format_args
}

pub(crate) unsafe extern "C" fn print_all_stdout(rt: *mut c_void, args: *mut usize, num_args: Int) {
    let args_wrapped: &[&Str] =
        slice::from_raw_parts(args as *const usize as *const &Str, num_args as usize);
    let rt = rt as *mut Runtime;
    try_silent_abort!(rt, (*rt).core.write_files.write_all(args_wrapped, None))
}

pub(crate) unsafe extern "C" fn print_all_file(
    rt: *mut c_void,
    args: *mut usize,
    num_args: Int,
    output: *const U128,
    append: Int,
) {
    let args_wrapped: &[&Str] =
        slice::from_raw_parts(args as *const usize as *const &Str, num_args as usize);
    let rt = rt as *mut Runtime;
    let output_wrapped = Some((
        &*(output as *mut Str),
        try_abort!(rt, FileSpec::try_from(append)),
    ));

    try_silent_abort!(
        rt,
        (*rt)
            .core
            .write_files
            .write_all(args_wrapped, output_wrapped)
    )
}

pub(crate) unsafe extern "C" fn printf_impl_file(
    rt: *mut c_void,
    spec: *mut U128,
    args: *mut usize,
    tys: *mut u32,
    num_args: Int,
    output: *mut U128,
    append: Int,
) {
    let output_wrapped = Some((
        &*(output as *mut Str),
        try_abort!(rt, FileSpec::try_from(append)),
    ));
    let format_args = wrap_args(&mut *(rt as *mut _), args, tys, num_args);
    let rt = rt as *mut Runtime;
    try_abort!(
        rt,
        (*rt)
            .core
            .write_files
            .printf(output_wrapped, &*(spec as *mut Str), &format_args[..],)
    )
}

pub(crate) unsafe extern "C" fn sprintf_impl(
    rt: *mut c_void,
    spec: *mut U128,
    args: *mut usize,
    tys: *mut u32,
    num_args: Int,
) -> U128 {
    use runtime::str_impl::DynamicBuf;
    let mut buf = DynamicBuf::new(0);
    let rt = &mut *(rt as *mut _);
    let format_args = wrap_args(rt, args, tys, num_args);
    let spec = &*(spec as *mut Str);
    if let Err(e) = spec.with_bytes(|bs| printf(&mut buf, bs, &format_args[..])) {
        fail!(rt, "unexpected failure during sprintf: {}", e);
    }
    mem::transmute::<Str, U128>(buf.into_str())
}

pub(crate) unsafe extern "C" fn printf_impl_stdout(
    rt: *mut c_void,
    spec: *mut U128,
    args: *mut usize,
    tys: *mut u32,
    num_args: Int,
) {
    let format_args = wrap_args(&mut *(rt as *mut _), args, tys, num_args);
    let res = (*(rt as *mut Runtime)).core.write_files.printf(
        None,
        &*(spec as *mut Str),
        &format_args[..],
    );
    if res.is_err() {
        exit!(rt);
    }
}

pub(crate) unsafe extern "C" fn close_file(rt: *mut c_void, file: *mut U128) {
    let rt = &mut *(rt as *mut Runtime);
    let file = &*(file as *mut Str);
    with_input!(&mut rt.input_data, |(_, read_files)| read_files.close(file));
    try_abort!(rt, rt.core.write_files.close(file));
}

pub(crate) unsafe extern "C" fn _frawk_cos(f: Float) -> Float {
    f.cos()
}

pub(crate) unsafe extern "C" fn _frawk_sin(f: Float) -> Float {
    f.sin()
}

pub(crate) unsafe extern "C" fn _frawk_log(f: Float) -> Float {
    f.ln()
}

pub(crate) unsafe extern "C" fn _frawk_log2(f: Float) -> Float {
    f.log2()
}

pub(crate) unsafe extern "C" fn _frawk_log10(f: Float) -> Float {
    f.log10()
}

pub(crate) unsafe extern "C" fn _frawk_exp(f: Float) -> Float {
    f.exp()
}

pub(crate) unsafe extern "C" fn _frawk_atan(f: Float) -> Float {
    f.atan()
}

pub(crate) unsafe extern "C" fn _frawk_atan2(x: Float, y: Float) -> Float {
    x.atan2(y)
}

pub(crate) unsafe extern "C" fn _frawk_pow(x: Float, y: Float) -> Float {
    Float::powf(x, y)
}

pub(crate) unsafe extern "C" fn _frawk_fprem(x: Float, y: Float) -> Float {
    x % y
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
// But that didn't end up working out. We had intrinsic functions with parameter types like <Int as
// InTy>::In, which had strange consequences like not being able to take the address of a function.
// We need to take the address of these functions though, because we pass them to generated code.
// Instead, we replicate this trait in the form of macros that match on the input type.

macro_rules! in_ty {
    (Str) => { *mut c_void };
    (Int) => { Int };
    (Float) => { Float };
    (Map) => { *mut c_void };
}

macro_rules! iter_ty {
    (Str) => { *mut c_void };
    (Int) => { *mut Int };
}

macro_rules! out_ty {
    (Str) => {
        U128
    };
    (Int) => {
        Int
    };
    (Float) => {
        Float
    };
    (Map) => { *mut c_void }
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

macro_rules! convert_in_val {
    (Str, $e:expr) => {
        (&*($e as *mut Str)).clone()
    };
    (Int, $e:expr) => {
        $e
    };
    (Float, $e:expr) => {
        $e
    };
    (Map, $e:expr) => {
        mem::transmute::<&*mut c_void, &runtime::SharedMap<_, _>>(&$e).clone()
    };
}

macro_rules! convert_out {
    (Str, $e:expr) => {
        mem::transmute::<Str, U128>($e)
    };
    (Int, $e:expr) => {
        $e
    };
    (Float, $e:expr) => {
        $e
    };
    (Map, $e:expr) => {{
        let map: runtime::SharedMap<_, _> = $e;
        mem::transmute::<_, *mut c_void>(map)
    }};
}

macro_rules! map_impl {
    ($ty:ident, $k:tt, $v:tt) => {
        paste! {
            pub(crate) unsafe extern "C" fn [< alloc_ $ty >]() -> *mut c_void {
                let res: runtime::SharedMap<$k, $v> = Default::default();
                mem::transmute::<runtime::SharedMap<$k, $v>, *mut c_void>(res)
            }

            pub(crate) unsafe extern "C" fn [< iter_ $ty >](map: *mut c_void) -> iter_ty!($k) {
                debug_assert!(!map.is_null());
                let map = mem::transmute::<*mut c_void, runtime::SharedMap<$k, $v>>(map);
                let iter: Vec<_> = map.to_vec();
                mem::forget(map);
                let b = iter.into_boxed_slice();
                Box::into_raw(b) as _
            }

            pub(crate) unsafe extern "C" fn [<len_ $ty>](map: *mut c_void) -> Int {
                debug_assert!(!map.is_null());
                let map = mem::transmute::<*mut c_void, runtime::SharedMap<$k, $v>>(map);
                let res = map.len();
                mem::forget(map);
                res as Int
            }

            pub(crate) unsafe extern "C" fn [<lookup_ $ty>](map: *mut c_void, k: in_ty!($k)) -> out_ty!($v) {
                // TODO: this should probably insert the value as well!
                debug_assert!(!map.is_null());
                let map = mem::transmute::<*mut c_void, runtime::SharedMap<$k, $v>>(map);
                let key = convert_in!($k, &k);
                let res = map.get(key);
                mem::forget(map);
                convert_out!($v, res)
            }

            pub(crate) unsafe extern "C" fn [<contains_ $ty>](map: *mut c_void, k: in_ty!($k)) -> Int {
                debug_assert!(!map.is_null());
                let map = mem::transmute::<*mut c_void, runtime::SharedMap<$k, $v>>(map);
                let key = convert_in!($k, &k);
                let res = map.contains(key) as Int;
                mem::forget(map);
                res
            }

            pub(crate) unsafe extern "C" fn [<insert_ $ty>](map: *mut c_void, k: in_ty!($k), v: in_ty!($v)) {
                debug_assert!(!map.is_null());
                let map = mem::transmute::<*mut c_void, runtime::SharedMap<$k, $v>>(map);
                let key = convert_in!($k, &k);
                let val = convert_in!($v, &v);
                map.insert(key.clone(), val.clone());
                mem::forget(map);
            }

            pub(crate) unsafe extern "C" fn [<delete_ $ty>](map: *mut c_void, k: in_ty!($k)) {
                debug_assert!(!map.is_null());
                let map = mem::transmute::<*mut c_void, runtime::SharedMap<$k, $v>>(map);
                let key = convert_in!($k, &k);
                map.delete(key);
                mem::forget(map);
            }

            pub(crate) unsafe extern "C" fn [<clear_ $ty>](map: *mut c_void) {
                debug_assert!(!map.is_null());
                let map = mem::transmute::<*mut c_void, runtime::SharedMap<$k, $v>>(map);
                map.clear();
                mem::forget(map);
            }

            pub(crate) unsafe extern "C" fn [<drop_ $ty>](map: *mut c_void) {
                debug_assert!(!map.is_null());
                drop_map_generic::<$k, $v>(map)
            }

            pub(crate) unsafe extern "C" fn [<inc_int_ $ty>](map: *mut c_void, k: in_ty!($k), by: Int) -> out_ty!($v) {
                debug_assert!(!map.is_null());
                let map = mem::transmute::<*mut c_void, runtime::SharedMap<$k, $v>>(map);
                let key = convert_in!($k, &k);
                let res = map.inc_int(key, by);
                mem::forget(map);
                convert_out!($v, res)
            }

            pub(crate) unsafe extern "C" fn [<inc_float_ $ty>](map: *mut c_void, k: in_ty!($k), by: Float) -> out_ty!($v) {
                debug_assert!(!map.is_null());
                let map = mem::transmute::<*mut c_void, runtime::SharedMap<$k, $v>>(map);
                let key = convert_in!($k, &k);
                let res = map.inc_float(key, by);
                mem::forget(map);
                convert_out!($v, res)
            }
        }
    };
}

map_impl!(intint, Int, Int);
map_impl!(intfloat, Int, Float);
map_impl!(intstr, Int, Str);
map_impl!(strint, Str, Int);
map_impl!(strfloat, Str, Float);
map_impl!(strstr, Str, Str);

macro_rules! slot_impl {
    ($name:ident, $ty:tt) => {
        paste! {
            pub(crate) unsafe extern "C" fn [<load_slot_ $name>](runtime: *mut c_void, slot: Int) -> out_ty!($ty) {
                let runtime = &mut *(runtime as *mut Runtime);
                convert_out!($ty, runtime.core.[<load_ $name>](slot as usize))
            }

            pub(crate) unsafe extern "C" fn [<store_slot_ $name>](runtime: *mut c_void, slot: Int, v: in_ty!($ty)) {
                let runtime = &mut *(runtime as *mut Runtime);
                runtime
                    .core
                    .[<store_ $name>](slot as usize, convert_in_val!($ty, v));
            }
        }
    };
}

slot_impl!(int, Int);
slot_impl!(float, Float);
slot_impl!(str, Str);
slot_impl!(intint, Map);
slot_impl!(intfloat, Map);
slot_impl!(intstr, Map);
slot_impl!(strint, Map);
slot_impl!(strfloat, Map);
slot_impl!(strstr, Map);
