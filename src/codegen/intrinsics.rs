use super::{CodeGenerator, FunctionAttr, Sig};
use crate::{common::Result, compile::Ty};

/// Lazily registers all runtime functions with the given LLVM module and context.
pub(crate) fn register_all(cg: &mut impl CodeGenerator) -> Result<()> {
    let int_ty = cg.get_ty(Ty::Int);
    let float_ty = cg.get_ty(Ty::Float);
    let str_ty = cg.get_ty(Ty::Float);
    let rt_ty = cg.void_ptr_ty();
    // do we want to make these more advanced (they are actually *usize and *u32s)
    let fmt_args_ty = cg.void_ptr_ty();
    let fmt_tys_ty = cg.void_ptr_ty();
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
                crate::llvm::intrinsics::$name as *const u8,
                Sig {
                    attrs: &[$(FunctionAttr::$attr),*],
                    args: &[$($param.clone()),*],
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
        drop_str_slow(str_ref_ty, int_ty);
        ref_map(map_ty);
        [ReadOnly] int_to_str(int_ty) -> str_ty;
        [ReadOnly] float_to_str(float_ty) -> str_ty;
        [ReadOnly] str_to_int(str_ref_ty) -> int_ty;
        [ReadOnly] hex_str_to_int(str_ref_ty) -> int_ty;
        [ReadOnly] str_to_float(str_ref_ty) -> float_ty;
        [ReadOnly] str_len(str_ref_ty) -> int_ty;
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
        set_col(rt_ty, int_ty, str_ref_ty);
        split_int(rt_ty, str_ref_ty, map_ty, str_ref_ty) -> int_ty;
        split_str(rt_ty, str_ref_ty, map_ty, str_ref_ty) -> int_ty;
        rand_float(rt_ty) -> float_ty;
        seed_rng(rt_ty, int_ty) -> int_ty;
        reseed_rng(rt_ty) -> int_ty;

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

        [ReadOnly, ArgmemOnly] _frawk_atan(float_ty) -> float_ty;
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
