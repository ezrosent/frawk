use super::llvm::{
    self,
    prelude::{LLVMModuleRef, LLVMValueRef},
};
use crate::compile;
use crate::libc::c_void;
use crate::runtime::{
    FileRead, FileWrite, Float, Int, IntMap, Iter, LazyVec, RegexCache, Str, StrMap, Variables,
};
use hashbrown::HashMap;

struct Runtime<'a> {
    strs: Vec<Str<'a>>,
    maps_int_float: Vec<IntMap<Float>>,
    maps_int_int: Vec<IntMap<Int>>,
    maps_int_str: Vec<IntMap<Str<'a>>>,

    maps_str_float: Vec<StrMap<'a, Float>>,
    maps_str_int: Vec<StrMap<'a, Int>>,
    maps_str_str: Vec<StrMap<'a, Str<'a>>>,

    iters_int: Vec<Iter<Int>>,
    iters_str: Vec<Iter<Str<'a>>>,

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
        use compile::Ty::*;
        // Lambdas aren't polymorphic, otherwise we could use that.
        macro_rules! default_of {
            ($ty:expr) => {{
                let mut res = Vec::new();
                res.resize_with(regs($ty), Default::default);
                res
            }};
        };

        Runtime {
            strs: default_of!(Str),
            vars: Default::default(),

            line: "".into(),
            split_line: LazyVec::new(),
            regexes: Default::default(),
            write_files: FileWrite::new(stdout),
            read_files: FileRead::new(stdin),

            maps_int_float: default_of!(MapIntFloat),
            maps_int_int: default_of!(MapIntInt),
            maps_int_str: default_of!(MapIntStr),

            maps_str_float: default_of!(MapStrFloat),
            maps_str_int: default_of!(MapStrInt),
            maps_str_str: default_of!(MapStrStr),

            iters_int: default_of!(IterInt),
            iters_str: default_of!(IterStr),
        }
    }
}

pub unsafe fn register(
    module: LLVMModuleRef,
) -> HashMap<*mut () /* function address */, LLVMValueRef> {
    use llvm::core::*;
    let usize_ty = LLVMIntType((std::mem::size_of::<usize>() * 8) as libc::c_uint);
    let int_ty = LLVMIntType((std::mem::size_of::<Int>() * 8) as libc::c_uint);
    let float_ty = LLVMDoubleType();
    let void_ty = LLVMVoidType();
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
        lookup_map_intint(rt_ty, usize_ty, int_ty) -> int_ty;
        store_map_intint(rt_ty, usize_ty, int_ty, int_ty);

        lookup_map_intfloat(rt_ty, usize_ty, int_ty) -> float_ty;
        store_map_intfloat(rt_ty, usize_ty, int_ty, float_ty);

        lookup_map_intstr(rt_ty, usize_ty, int_ty, usize_ty);
        store_map_intstr(rt_ty, usize_ty, int_ty, usize_ty);

        lookup_map_strint(rt_ty, usize_ty, usize_ty) -> int_ty;
        store_map_strint(rt_ty, usize_ty, usize_ty, int_ty);

        lookup_map_strfloat(rt_ty, usize_ty, usize_ty) -> float_ty;
        lookup_map_strstr(rt_ty, usize_ty, usize_ty, usize_ty);
    };
    table
}

macro_rules! make_lookup {
    ($lookup:ident, $store:ident, $fld:tt, $k:ty, $v:ty) => {
        #[no_mangle]
        pub unsafe extern "C" fn $lookup(runtime: *mut c_void, map: usize, key: $k) -> $v {
            let runtime = &mut *(runtime as *mut Runtime);
            debug_assert!(map < runtime.$fld.len());
            runtime
                .$fld
                .get_unchecked(map)
                .get(&key)
                .unwrap_or(Default::default())
        }

        #[no_mangle]
        pub unsafe extern "C" fn $store(runtime: *mut c_void, map: usize, key: $k, val: $v) {
            let runtime = &mut *(runtime as *mut Runtime);
            debug_assert!(map < runtime.$fld.len());
            runtime.$fld.get_unchecked(map).insert(key, val)
        }
    };
}

make_lookup!(lookup_map_intint, store_map_intint, maps_int_int, Int, Int);
make_lookup!(
    lookup_map_intfloat,
    store_map_intfloat,
    maps_int_float,
    Int,
    Float
);

#[no_mangle]
pub unsafe extern "C" fn lookup_map_intstr(runtime: *mut c_void, map: usize, key: Int, dst: usize) {
    let runtime = &mut *(runtime as *mut Runtime);
    debug_assert!(map < runtime.maps_int_str.len());
    debug_assert!(dst < runtime.strs.len());
    let dst_ref = runtime.strs.get_unchecked_mut(dst);
    *dst_ref = runtime
        .maps_int_str
        .get_unchecked(map)
        .get(&key)
        .unwrap_or(Default::default());
}

#[no_mangle]
pub unsafe extern "C" fn store_map_intstr(runtime: *mut c_void, map: usize, key: Int, val: usize) {
    let runtime = &mut *(runtime as *mut Runtime);
    debug_assert!(map < runtime.maps_int_str.len());
    debug_assert!(val < runtime.strs.len());
    let val = runtime.strs.get_unchecked(val).clone();
    runtime.maps_int_str.get_unchecked(map).insert(key, val);
}

#[no_mangle]
pub unsafe extern "C" fn lookup_map_strint(runtime: *mut c_void, map: usize, key: usize) -> Int {
    let runtime = &mut *(runtime as *mut Runtime);
    debug_assert!(map < runtime.maps_str_int.len());
    debug_assert!(key < runtime.strs.len());
    let key_ref = runtime.strs.get_unchecked(key);
    runtime
        .maps_str_int
        .get_unchecked(map)
        .get(key_ref)
        .unwrap_or(Default::default())
}

#[no_mangle]
pub unsafe extern "C" fn store_map_strint(runtime: *mut c_void, map: usize, key: usize, val: Int) {
    let runtime = &mut *(runtime as *mut Runtime);
    debug_assert!(map < runtime.maps_str_int.len());
    debug_assert!(key < runtime.strs.len());
    let key = runtime.strs.get_unchecked(key).clone();
    runtime.maps_str_int.get_unchecked(map).insert(key, val);
}

#[no_mangle]
pub unsafe extern "C" fn lookup_map_strfloat(
    runtime: *mut c_void,
    map: usize,
    key: usize,
) -> Float {
    let runtime = &mut *(runtime as *mut Runtime);
    debug_assert!(map < runtime.maps_str_float.len());
    debug_assert!(key < runtime.strs.len());
    let key_ref = runtime.strs.get_unchecked(key);
    runtime
        .maps_str_float
        .get_unchecked(map)
        .get(key_ref)
        .unwrap_or(Default::default())
}

#[no_mangle]
pub unsafe extern "C" fn lookup_map_strstr(
    runtime: *mut c_void,
    map: usize,
    key: usize,
    dst: usize,
) {
    let runtime = &mut *(runtime as *mut Runtime);
    debug_assert!(map < runtime.maps_str_str.len());
    debug_assert!(dst < runtime.strs.len());
    debug_assert!(key < runtime.strs.len());
    let val = {
        let key_ref = runtime.strs.get_unchecked(key);
        runtime
            .maps_str_str
            .get_unchecked(map)
            .get(key_ref)
            .unwrap_or(Default::default())
    };

    let dst_ref = runtime.strs.get_unchecked_mut(dst);
    *dst_ref = val;
}
