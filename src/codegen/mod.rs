use crate::{
    builtins,
    bytecode::{self, Accum},
    common::{FileSpec, NumTy, Result},
    compile,
    runtime::{self, UniqueStr},
};

pub(crate) mod intrinsics;

// TODO: move intrinsics module over to codegen, make it CodeGenerator-generic.
//  (fine that some runtime libraries are left behind for the time being in llvm)
// TODO: continue stubbing out gen_inst; filling in missing items. I think the idea is that for
// difficult stuff, we'll add a method that implementors will override, and that this will
// encapsulate code generation "without the branches"
// TODO: migrate LLVM over to gen_inst, get everything passing
// TODO: move LLVM into codegen module
// TODO: start on clir

type SmallVec<T> = smallvec::SmallVec<[T; 4]>;
pub(crate) type Ref = (NumTy, compile::Ty);
pub(crate) type StrReg<'a> = bytecode::Reg<runtime::Str<'a>>;

pub(crate) struct Sig<'a, C: CodeGenerator + ?Sized> {
    pub attrs: &'a [FunctionAttr],
    pub args: &'a mut [C::Ty],
    pub ret: Option<C::Ty>,
}

// TODO: fill in the intrinsics stuf...

macro_rules! intrinsic {
    ($name:ident) => {
        Op::Intrinsic(crate::llvm::intrinsics::$name as *const u8)
    };
}

pub(crate) enum Cmp {
    EQ,
    NEQ,
    LTE,
    LT,
    GTE,
    GT,
}

pub(crate) enum Arith {
    Mul,
    Minus,
    Add,
    Mod,
    Neg,
}

pub(crate) enum Op {
    // TODO: we probably don't need the is_float here? we could infer based on the operands?
    // otoh it's more explicit...
    Cmp { is_float: bool, op: Cmp },
    Arith { is_float: bool, op: Arith },
    Bitwise(builtins::Bitwise),
    Math(builtins::FloatFunc),
    Div,
    Pow,
    FloatToInt,
    IntToFloat,
    Intrinsic(*const u8),
}

fn op(op: Arith, is_float: bool) -> Op {
    Op::Arith { is_float, op }
}
fn cmp(op: Cmp, is_float: bool) -> Op {
    Op::Cmp { is_float, op }
}

#[derive(Debug, Copy, Clone)]
pub enum FunctionAttr {
    ReadOnly,
    ArgmemOnly,
}

/// CodeGenerator encapsulates common functionality needed to generate instructions across multiple
/// backends. This trait is not currently sufficient to abstract over any backend "end to end" from
/// bytecode instructions all the way to machine code, but it allows us to keep much of the more
/// mundane plumbing work common across all backends (as well as separate safe "glue code" from
/// unsafe calls to the LLVM C API).
pub(crate) trait CodeGenerator {
    type Ty: Clone;
    type Val;

    /// Register a function with address `addr` and name `name` (/ `name_c`, the null-terminated
    /// variant) with signature `Sig` to be called.
    fn register_external_fn(
        &mut self,
        name: &'static str,
        name_c: *const u8,
        addr: *const u8,
        sig: Sig<Self>,
    ) -> Result<()>;

    // mappings from compile::Ty to Self::Ty
    fn void_ptr_ty(&self) -> Self::Ty;
    fn ptr_to(&self, ty: Self::Ty) -> Self::Ty;
    fn usize_ty(&self) -> Self::Ty;
    fn get_ty(&self, ty: compile::Ty) -> Self::Ty;

    // mappings to and from bytecode-level registers to IR-level values
    fn bind_val(&mut self, r: Ref, v: Self::Val) -> Result<()>;
    fn get_val(&mut self, r: Ref) -> Result<Self::Val>;

    // backend-specific handling of constants and low-level operations.
    fn runtime_val(&self) -> Self::Val;
    fn const_int(&self, i: i64) -> Self::Val;
    fn const_float(&self, f: f64) -> Self::Val;
    fn const_str<'a>(&self, s: &UniqueStr<'a>) -> Self::Val;
    fn const_ptr<'a, T>(&'a self, c: &'a T) -> Self::Val;

    /// Call an intrinsic, given a pointer to the [`intrinsics`] module and a list of arguments.
    fn call_intrinsic(&mut self, func: Op, args: &mut [Self::Val]) -> Result<Self::Val>;

    // var-arg printing functions. The arguments here directly parallel the instruction
    // definitions.

    fn printf(
        &mut self,
        output: &Option<(StrReg, FileSpec)>,
        fmt: &StrReg,
        args: &Vec<Ref>,
    ) -> Result<()>;

    fn sprintf(&mut self, dst: &StrReg, fmt: &StrReg, args: &Vec<Ref>) -> Result<()>;

    fn print_all(&mut self, output: &Option<(StrReg, FileSpec)>, args: &Vec<StrReg>) -> Result<()>;

    /// Moves the contents of `src` into `dst`, taking refcounts into consideration if necessary.
    fn mov(&mut self, ty: compile::Ty, dst: NumTy, src: NumTy) -> Result<()>;

    /// Constructs an iterator over the keys of `map` and stores it in `dst`.
    fn iter_begin(&mut self, dst: Ref, map: Ref) -> Result<()>;

    /// Queries the iterator in `iter` as to whether any elements remain, stores the result in the
    /// `dst` register.
    fn iter_hasnext(&mut self, dst: Ref, iter: Ref) -> Result<()>;

    /// Advances the iterator in `iter` to the next element and stores the current element in `dst`
    fn iter_getnext(&mut self, dst: Ref, iter: Ref) -> Result<()>;

    // The plumbing for builtin variable manipulation is mostly pretty wrote ... anything we can do
    // here?

    /// Method called after loading a builtin variable into `dst`.
    ///
    /// This is included to help clean up ref-counts on string or map builtins, if necessary.
    fn var_loaded(&mut self, dst: Ref) -> Result<()>;

    // derived functions

    /// Loads contents of given slot into dst.
    ///
    /// Assumes that dst.1 is a type we can store in a slot (i.e. it cannot be an iterator)
    fn load_slot(&mut self, dst: Ref, slot: i64) -> Result<()> {
        use compile::Ty::*;
        let slot_v = self.const_int(slot);
        let func = match dst.1 {
            Int => intrinsic!(load_slot_int),
            Float => intrinsic!(load_slot_float),
            Str => intrinsic!(load_slot_str),
            MapIntInt => intrinsic!(load_slot_intint),
            MapIntFloat => intrinsic!(load_slot_intfloat),
            MapIntStr => intrinsic!(load_slot_intstr),
            MapStrInt => intrinsic!(load_slot_strint),
            MapStrFloat => intrinsic!(load_slot_strfloat),
            MapStrStr => intrinsic!(load_slot_strstr),
            _ => unreachable!(),
        };
        let resv = self.call_intrinsic(func, &mut [self.runtime_val(), slot_v])?;
        self.bind_val(dst, resv)
    }

    /// Stores contents of src into a given slot.
    ///
    /// Assumes that src.1 is a type we can store in a slot (i.e. it cannot be an iterator)
    fn store_slot(&mut self, src: Ref, slot: i64) -> Result<()> {
        use compile::Ty::*;
        let slot_v = self.const_int(slot);
        let func = match src.1 {
            Int => intrinsic!(store_slot_int),
            Float => intrinsic!(store_slot_float),
            Str => intrinsic!(store_slot_str),
            MapIntInt => intrinsic!(store_slot_intint),
            MapIntFloat => intrinsic!(store_slot_intfloat),
            MapIntStr => intrinsic!(store_slot_intstr),
            MapStrInt => intrinsic!(store_slot_strint),
            MapStrFloat => intrinsic!(store_slot_strfloat),
            MapStrStr => intrinsic!(store_slot_strstr),
            _ => unreachable!(),
        };
        let arg = self.get_val(src)?;
        self.call_intrinsic(func, &mut [self.runtime_val(), slot_v, arg])?;
        Ok(())
    }

    /// Retrieves the contents of `map` at `key` and stores them in `dst`.
    ///
    /// These are "awk lookups" that insert a default value into the map if it is not presetn.
    /// Assumes that types of map, key, dst match up.
    fn lookup_map(&mut self, map: Ref, key: Ref, dst: Ref) -> Result<()> {
        use compile::Ty::*;
        map_valid(map.1, key.1, dst.1)?;
        let func = match map.1 {
            MapIntInt => intrinsic!(lookup_intint),
            MapIntFloat => intrinsic!(lookup_intfloat),
            MapIntStr => intrinsic!(lookup_intstr),
            MapStrInt => intrinsic!(lookup_strint),
            MapStrFloat => intrinsic!(lookup_strfloat),
            MapStrStr => intrinsic!(lookup_strstr),
            ty => return err!("non-map type: {:?}", ty),
        };
        let mapv = self.get_val(map)?;
        let keyv = self.get_val(key)?;
        let resv = self.call_intrinsic(func, &mut [mapv, keyv])?;
        self.bind_val(dst, resv)?;
        Ok(())
    }

    /// Deletes the contents of `map` at `key`.
    ///
    /// Assumes that map and key types match up.
    fn delete_map(&mut self, map: Ref, key: Ref) -> Result<()> {
        use compile::Ty::*;
        map_key_valid(map.1, key.1)?;
        let func = match map.1 {
            MapIntInt => intrinsic!(delete_intint),
            MapIntFloat => intrinsic!(delete_intfloat),
            MapIntStr => intrinsic!(delete_intstr),
            MapStrInt => intrinsic!(delete_strint),
            MapStrFloat => intrinsic!(delete_strfloat),
            MapStrStr => intrinsic!(delete_strstr),
            ty => return err!("non-map type: {:?}", ty),
        };
        let mapv = self.get_val(map)?;
        let keyv = self.get_val(key)?;
        self.call_intrinsic(func, &mut [mapv, keyv])?;
        Ok(())
    }

    /// Determines if `map` contains `key` and stores the result (0 or 1) in `dst`.
    ///
    /// Assumes that map and key types match up.
    fn contains_map(&mut self, map: Ref, key: Ref, dst: Ref) -> Result<()> {
        use compile::Ty::*;
        map_key_valid(map.1, key.1)?;
        let func = match map.1 {
            MapIntInt => intrinsic!(contains_intint),
            MapIntFloat => intrinsic!(contains_intfloat),
            MapIntStr => intrinsic!(contains_intstr),
            MapStrInt => intrinsic!(contains_strint),
            MapStrFloat => intrinsic!(contains_strfloat),
            MapStrStr => intrinsic!(contains_strstr),
            ty => return err!("non-map type: {:?}", ty),
        };
        let mapv = self.get_val(map)?;
        let keyv = self.get_val(key)?;
        let resv = self.call_intrinsic(func, &mut [mapv, keyv])?;
        self.bind_val(dst, resv)?;
        Ok(())
    }

    /// Stores the size of `map` in `dst`.
    fn len_map(&mut self, map: Ref, dst: Ref) -> Result<()> {
        use compile::Ty::*;
        let func = match map.1 {
            MapIntInt => intrinsic!(len_intint),
            MapIntFloat => intrinsic!(len_intfloat),
            MapIntStr => intrinsic!(len_intstr),
            MapStrInt => intrinsic!(len_strint),
            MapStrFloat => intrinsic!(len_strfloat),
            MapStrStr => intrinsic!(len_strstr),
            ty => return err!("non-map type: {:?}", ty),
        };
        let mapv = self.get_val(map)?;
        let resv = self.call_intrinsic(func, &mut [mapv])?;
        self.bind_val(dst, resv)?;
        Ok(())
    }

    /// Stores `val` into `map` at key `key`.
    ///
    /// Assumes that the types of the input registers match up.
    fn store_map(&mut self, map: Ref, key: Ref, val: Ref) -> Result<()> {
        assert_eq!(map.1.key()?, key.1);
        assert_eq!(map.1.val()?, val.1);
        use compile::Ty::*;
        map_valid(map.1, key.1, val.1)?;
        let func = match map.1 {
            MapIntInt => intrinsic!(insert_intint),
            MapIntFloat => intrinsic!(insert_intfloat),
            MapIntStr => intrinsic!(insert_intstr),
            MapStrInt => intrinsic!(insert_strint),
            MapStrFloat => intrinsic!(insert_strfloat),
            MapStrStr => intrinsic!(insert_strstr),
            ty => return err!("non-map type: {:?}", ty),
        };
        let mapv = self.get_val(map)?;
        let keyv = self.get_val(key)?;
        let valv = self.get_val(val)?;
        self.call_intrinsic(func, &mut [mapv, keyv, valv])?;
        Ok(())
    }

    /// Wraps `call_intrinsic` for [`Op`]s that have two arguments and return a value.
    fn binop(&mut self, op: Op, dst: &impl Accum, l: &impl Accum, r: &impl Accum) -> Result<()> {
        let lv = self.get_val(l.reflect())?;
        let rv = self.get_val(r.reflect())?;
        let res = self.call_intrinsic(op, &mut [lv, rv])?;
        self.bind_val(dst.reflect(), res)
    }

    /// Wraps `call_intrinsic` for [`Op`]s that have one argument and return a value.
    fn unop(&mut self, op: Op, dst: &impl Accum, x: &impl Accum) -> Result<()> {
        let xv = self.get_val(x.reflect())?;
        let res = self.call_intrinsic(op, &mut [xv])?;
        self.bind_val(dst.reflect(), res)
    }

    fn gen_ll_inst(&mut self, inst: &compile::LL) -> Result<()> {
        use crate::bytecode::Instr::*;
        match inst {
            StoreConstStr(sr, s) => {
                let sv = self.const_str(s);
                self.bind_val(sr.reflect(), sv)
            }
            StoreConstInt(ir, i) => {
                let iv = self.const_int(*i);
                self.bind_val(ir.reflect(), iv)
            }
            StoreConstFloat(fr, f) => {
                let fv = self.const_float(*f);
                self.bind_val(fr.reflect(), fv)
            }
            IntToStr(sr, ir) => self.unop(intrinsic!(int_to_str), sr, ir),
            FloatToStr(sr, fr) => self.unop(intrinsic!(float_to_str), sr, fr),
            StrToInt(ir, sr) => self.unop(intrinsic!(str_to_int), ir, sr),
            HexStrToInt(ir, sr) => self.unop(intrinsic!(hex_str_to_int), ir, sr),
            StrToFloat(fr, sr) => self.unop(intrinsic!(str_to_float), fr, sr),
            FloatToInt(ir, fr) => self.unop(Op::FloatToInt, ir, fr),
            IntToFloat(fr, ir) => self.unop(Op::IntToFloat, fr, ir),
            AddInt(res, l, r) => self.binop(op(Arith::Add, false), res, l, r),
            AddFloat(res, l, r) => self.binop(op(Arith::Add, true), res, l, r),
            MinusInt(res, l, r) => self.binop(op(Arith::Minus, false), res, l, r),
            MinusFloat(res, l, r) => self.binop(op(Arith::Minus, true), res, l, r),
            MulInt(res, l, r) => self.binop(op(Arith::Mul, false), res, l, r),
            MulFloat(res, l, r) => self.binop(op(Arith::Mul, true), res, l, r),
            ModInt(res, l, r) => self.binop(op(Arith::Mod, false), res, l, r),
            ModFloat(res, l, r) => self.binop(op(Arith::Mod, true), res, l, r),
            Div(res, l, r) => self.binop(Op::Div, res, l, r),
            Pow(res, l, r) => self.binop(Op::Pow, res, l, r),
            Not(res, ir) => {
                let iv = self.get_val(ir.reflect())?;
                let zero = self.const_int(0);
                let cmp = self.call_intrinsic(
                    Op::Cmp {
                        is_float: false,
                        op: Cmp::EQ,
                    },
                    &mut [iv, zero],
                )?;
                self.bind_val(res.reflect(), cmp)
            }
            NotStr(res, sr) => {
                let sv = self.get_val(sr.reflect())?;
                let lenv = self.call_intrinsic(intrinsic!(str_len), &mut [sv])?;
                let zero = self.const_int(0);
                let cmp = self.call_intrinsic(
                    Op::Cmp {
                        is_float: false,
                        op: Cmp::EQ,
                    },
                    &mut [lenv, zero],
                )?;
                self.bind_val(res.reflect(), cmp)
            }
            NegInt(res, ir) => self.unop(op(Arith::Neg, false), res, ir),
            NegFloat(res, fr) => self.unop(op(Arith::Neg, true), res, fr),
            Float1(ff, dst, src) => self.unop(Op::Math(*ff), dst, src),
            Float2(ff, dst, l, r) => self.binop(Op::Math(*ff), dst, l, r),
            Int1(bw, dst, src) => self.unop(Op::Bitwise(*bw), dst, src),
            Int2(bw, dst, l, r) => self.binop(Op::Bitwise(*bw), dst, l, r),
            Rand(dst) => {
                let res = self.call_intrinsic(intrinsic!(rand_float), &mut [self.runtime_val()])?;
                self.bind_val(dst.reflect(), res)
            }
            Srand(dst, seed) => {
                let seedv = self.get_val(seed.reflect())?;
                let res =
                    self.call_intrinsic(intrinsic!(seed_rng), &mut [self.runtime_val(), seedv])?;
                self.bind_val(dst.reflect(), res)
            }
            ReseedRng(dst) => {
                let res = self.call_intrinsic(intrinsic!(reseed_rng), &mut [self.runtime_val()])?;
                self.bind_val(dst.reflect(), res)
            }
            Concat(dst, l, r) => self.binop(intrinsic!(concat), dst, l, r),
            Match(dst, l, r) => {
                let lv = self.get_val(l.reflect())?;
                let rv = self.get_val(r.reflect())?;
                let rt = self.runtime_val();
                let res = self.call_intrinsic(intrinsic!(match_pat_loc), &mut [rt, lv, rv])?;
                self.bind_val(dst.reflect(), res)
            }
            IsMatch(dst, l, r) => {
                let lv = self.get_val(l.reflect())?;
                let rv = self.get_val(r.reflect())?;
                let rt = self.runtime_val();
                let res = self.call_intrinsic(intrinsic!(match_pat), &mut [rt, lv, rv])?;
                self.bind_val(dst.reflect(), res)
            }
            MatchConst(res, src, pat) => {
                let rt = self.runtime_val();
                let srcv = self.get_val(src.reflect())?;
                let patv = self.const_ptr(&**pat);
                let resv =
                    self.call_intrinsic(intrinsic!(match_const_pat_loc), &mut [rt, srcv, patv])?;
                self.bind_val(res.reflect(), resv)
            }
            IsMatchConst(res, src, pat) => {
                let rt = self.runtime_val();
                let srcv = self.get_val(src.reflect())?;
                let patv = self.const_ptr(&**pat);
                let resv =
                    self.call_intrinsic(intrinsic!(match_const_pat), &mut [rt, srcv, patv])?;
                self.bind_val(res.reflect(), resv)
            }
            SubstrIndex(dst, s, t) => self.binop(intrinsic!(substr_index), dst, s, t),
            LenStr(dst, x) => self.unop(intrinsic!(str_len), dst, x),
            Sub(res, pat, s, in_s) => {
                let rt = self.runtime_val();
                let patv = self.get_val(pat.reflect())?;
                let sv = self.get_val(s.reflect())?;
                let in_sv = self.get_val(in_s.reflect())?;
                let resv =
                    self.call_intrinsic(intrinsic!(subst_first), &mut [rt, patv, sv, in_sv])?;
                self.bind_val(res.reflect(), resv)
            }
            GSub(res, pat, s, in_s) => {
                let rt = self.runtime_val();
                let patv = self.get_val(pat.reflect())?;
                let sv = self.get_val(s.reflect())?;
                let in_sv = self.get_val(in_s.reflect())?;
                let resv =
                    self.call_intrinsic(intrinsic!(subst_all), &mut [rt, patv, sv, in_sv])?;
                self.bind_val(res.reflect(), resv)
            }
            EscapeCSV(dst, s) => self.unop(intrinsic!(escape_csv), dst, s),
            EscapeTSV(dst, s) => self.unop(intrinsic!(escape_tsv), dst, s),
            Substr(res, base, l, r) => {
                let basev = self.get_val(base.reflect())?;
                let lv = self.get_val(l.reflect())?;
                let rv = self.get_val(r.reflect())?;
                let resv = self.call_intrinsic(intrinsic!(substr), &mut [basev, lv, rv])?;
                self.bind_val(res.reflect(), resv)
            }
            LTInt(res, l, r) => self.binop(cmp(Cmp::LT, false), res, l, r),
            GTInt(res, l, r) => self.binop(cmp(Cmp::GT, false), res, l, r),
            LTEInt(res, l, r) => self.binop(cmp(Cmp::LTE, false), res, l, r),
            GTEInt(res, l, r) => self.binop(cmp(Cmp::GTE, false), res, l, r),
            EQInt(res, l, r) => self.binop(cmp(Cmp::EQ, false), res, l, r),
            LTFloat(res, l, r) => self.binop(cmp(Cmp::LT, true), res, l, r),
            GTFloat(res, l, r) => self.binop(cmp(Cmp::GT, true), res, l, r),
            LTEFloat(res, l, r) => self.binop(cmp(Cmp::LTE, true), res, l, r),
            GTEFloat(res, l, r) => self.binop(cmp(Cmp::GTE, true), res, l, r),
            EQFloat(res, l, r) => self.binop(cmp(Cmp::EQ, true), res, l, r),
            LTStr(res, l, r) => self.binop(intrinsic!(str_lt), res, l, r),
            GTStr(res, l, r) => self.binop(intrinsic!(str_gt), res, l, r),
            LTEStr(res, l, r) => self.binop(intrinsic!(str_lte), res, l, r),
            GTEStr(res, l, r) => self.binop(intrinsic!(str_gte), res, l, r),
            EQStr(res, l, r) => self.binop(intrinsic!(str_eq), res, l, r),
            SetColumn(dst, src) => {
                let srcv = self.get_val(src.reflect())?;
                let dstv = self.get_val(dst.reflect())?;
                self.call_intrinsic(intrinsic!(set_col), &mut [self.runtime_val(), dstv, srcv])?;
                Ok(())
            }
            GetColumn(dst, src) => {
                let srcv = self.get_val(src.reflect())?;
                let dstv =
                    self.call_intrinsic(intrinsic!(get_col), &mut [self.runtime_val(), srcv])?;
                self.bind_val(dst.reflect(), dstv)
            }
            JoinCSV(dst, start, end) => {
                let startv = self.get_val(start.reflect())?;
                let endv = self.get_val(end.reflect())?;
                let resv = self.call_intrinsic(
                    intrinsic!(join_csv),
                    &mut [self.runtime_val(), startv, endv],
                )?;
                self.bind_val(dst.reflect(), resv)
            }
            JoinTSV(dst, start, end) => {
                let startv = self.get_val(start.reflect())?;
                let endv = self.get_val(end.reflect())?;
                let resv = self.call_intrinsic(
                    intrinsic!(join_tsv),
                    &mut [self.runtime_val(), startv, endv],
                )?;
                self.bind_val(dst.reflect(), resv)
            }
            JoinColumns(dst, start, end, sep) => {
                let startv = self.get_val(start.reflect())?;
                let endv = self.get_val(end.reflect())?;
                let sepv = self.get_val(sep.reflect())?;
                let resv = self.call_intrinsic(
                    intrinsic!(join_cols),
                    &mut [self.runtime_val(), startv, endv, sepv],
                )?;
                self.bind_val(dst.reflect(), resv)
            }
            SplitInt(flds, to_split, arr, pat) => {
                let tsv = self.get_val(to_split.reflect())?;
                let arrv = self.get_val(arr.reflect())?;
                let patv = self.get_val(pat.reflect())?;
                let fldsv = self.call_intrinsic(
                    intrinsic!(split_int),
                    &mut [self.runtime_val(), tsv, arrv, patv],
                )?;
                self.bind_val(flds.reflect(), fldsv)
            }
            SplitStr(flds, to_split, arr, pat) => {
                let tsv = self.get_val(to_split.reflect())?;
                let arrv = self.get_val(arr.reflect())?;
                let patv = self.get_val(pat.reflect())?;
                let fldsv = self.call_intrinsic(
                    intrinsic!(split_str),
                    &mut [self.runtime_val(), tsv, arrv, patv],
                )?;
                self.bind_val(flds.reflect(), fldsv)
            }
            Printf { output, fmt, args } => self.printf(output, fmt, args),
            Sprintf { dst, fmt, args } => self.sprintf(dst, fmt, args),
            PrintAll { output, args } => self.print_all(output, args),
            Close(file) => {
                let filev = self.get_val(file.reflect())?;
                self.call_intrinsic(intrinsic!(close_file), &mut [self.runtime_val(), filev])?;
                Ok(())
            }
            RunCmd(dst, cmd) => self.unop(intrinsic!(run_system), dst, cmd),
            ReadErr(dst, file, is_file) => {
                let filev = self.get_val(file.reflect())?;
                let is_filev = self.const_int(*is_file as i64);
                let resv = self.call_intrinsic(
                    intrinsic!(read_err),
                    &mut [self.runtime_val(), filev, is_filev],
                )?;
                self.bind_val(dst.reflect(), resv)
            }
            NextLine(dst, file, is_file) => {
                let filev = self.get_val(file.reflect())?;
                let is_filev = self.const_int(*is_file as i64);
                let resv = self.call_intrinsic(
                    intrinsic!(next_line),
                    &mut [self.runtime_val(), filev, is_filev],
                )?;
                self.bind_val(dst.reflect(), resv)
            }
            ReadErrStdin(dst) => {
                let resv =
                    self.call_intrinsic(intrinsic!(read_err_stdin), &mut [self.runtime_val()])?;
                self.bind_val(dst.reflect(), resv)
            }
            NextLineStdin(dst) => {
                let resv =
                    self.call_intrinsic(intrinsic!(next_line_stdin), &mut [self.runtime_val()])?;
                self.bind_val(dst.reflect(), resv)
            }
            NextLineStdinFused() => {
                self.call_intrinsic(intrinsic!(next_line_stdin_fused), &mut [self.runtime_val()])?;
                Ok(())
            }
            NextFile() => {
                self.call_intrinsic(intrinsic!(next_file), &mut [self.runtime_val()])?;
                Ok(())
            }
            UpdateUsedFields() => {
                self.call_intrinsic(intrinsic!(update_used_fields), &mut [self.runtime_val()])?;
                Ok(())
            }
            SetFI(key, val) => {
                // We could probably get away without an extra intrinsic here, but this way we can
                // avoid repeated refs and drops of the FI variable outside of the existing
                // framework for performing refs and drops.
                let keyv = self.get_val(key.reflect())?;
                let valv = self.get_val(val.reflect())?;
                self.call_intrinsic(
                    intrinsic!(set_fi_entry),
                    &mut [self.runtime_val(), keyv, valv],
                )?;
                Ok(())
            }
            Lookup {
                map_ty,
                dst,
                map,
                key,
            } => self.lookup_map(
                (*map, *map_ty),
                (*key, map_ty.key()?),
                (*dst, map_ty.val()?),
            ),
            Contains {
                map_ty,
                dst,
                map,
                key,
            } => self.contains_map(
                (*map, *map_ty),
                (*key, map_ty.key()?),
                (*dst, compile::Ty::Int),
            ),
            Delete { map_ty, map, key } => self.delete_map((*map, *map_ty), (*key, map_ty.key()?)),
            Len { map_ty, map, dst } => self.len_map((*map, *map_ty), (*dst, compile::Ty::Int)),
            Store {
                map_ty,
                map,
                key,
                val,
            } => self.store_map(
                (*map, *map_ty),
                (*key, map_ty.key()?),
                (*val, map_ty.val()?),
            ),
            LoadVarStr(dst, var) => {
                let varv = self.const_int(*var as i64);
                let res =
                    self.call_intrinsic(intrinsic!(load_var_str), &mut [self.runtime_val(), varv])?;
                let dref = dst.reflect();
                self.bind_val(dref, res)?;
                self.var_loaded(dref)
            }
            StoreVarStr(var, src) => {
                let varv = self.const_int(*var as i64);
                let srcv = self.get_val(src.reflect())?;
                self.call_intrinsic(
                    intrinsic!(store_var_str),
                    &mut [self.runtime_val(), varv, srcv],
                )?;
                Ok(())
            }
            LoadVarInt(dst, var) => {
                let varv = self.const_int(*var as i64);
                let res =
                    self.call_intrinsic(intrinsic!(load_var_int), &mut [self.runtime_val(), varv])?;
                let dref = dst.reflect();
                self.bind_val(dref, res)?;
                self.var_loaded(dref)
            }
            StoreVarInt(var, src) => {
                let varv = self.const_int(*var as i64);
                let srcv = self.get_val(src.reflect())?;
                self.call_intrinsic(
                    intrinsic!(store_var_int),
                    &mut [self.runtime_val(), varv, srcv],
                )?;
                Ok(())
            }
            LoadVarIntMap(dst, var) => {
                let varv = self.const_int(*var as i64);
                let res = self
                    .call_intrinsic(intrinsic!(load_var_intmap), &mut [self.runtime_val(), varv])?;
                let dref = dst.reflect();
                self.bind_val(dref, res)?;
                self.var_loaded(dref)
            }
            StoreVarIntMap(var, src) => {
                let varv = self.const_int(*var as i64);
                let srcv = self.get_val(src.reflect())?;
                self.call_intrinsic(
                    intrinsic!(store_var_intmap),
                    &mut [self.runtime_val(), varv, srcv],
                )?;
                Ok(())
            }
            LoadVarStrMap(dst, var) => {
                let varv = self.const_int(*var as i64);
                let res = self
                    .call_intrinsic(intrinsic!(load_var_strmap), &mut [self.runtime_val(), varv])?;
                let dref = dst.reflect();
                self.bind_val(dref, res)?;
                self.var_loaded(dref)
            }
            StoreVarStrMap(var, src) => {
                let varv = self.const_int(*var as i64);
                let srcv = self.get_val(src.reflect())?;
                self.call_intrinsic(
                    intrinsic!(store_var_strmap),
                    &mut [self.runtime_val(), varv, srcv],
                )?;
                Ok(())
            }
            LoadSlot { ty, dst, slot } => self.load_slot((*dst, *ty), *slot),
            StoreSlot { ty, src, slot } => self.store_slot((*src, *ty), *slot),
            Mov(ty, dst, src) => self.mov(*ty, *dst, *src),
            IterBegin { map_ty, map, dst } => {
                self.iter_begin((*dst, map_ty.key_iter()?), (*map, *map_ty))
            }
            IterHasNext { iter_ty, dst, iter } => {
                self.iter_hasnext((*dst, compile::Ty::Int), (*iter, *iter_ty))
            }
            IterGetNext { iter_ty, dst, iter } => {
                self.iter_getnext((*dst, iter_ty.iter()?), (*iter, *iter_ty))
            }
            Push(_, _) | Pop(_, _) => err!("unexpected explicit push/pop in llvm"),
            AllocMap(_, _) => {
                err!("unexpected AllocMap (allocs are handled differently in LLVM)")
            }
            Ret | Halt | Jmp(_) | JmpIf(_, _) | Call(_) => {
                err!("unexpected bytecode-level control flow")
            }
        }
    }
}

fn map_key_valid(map: compile::Ty, key: compile::Ty) -> Result<()> {
    if map.key()? != key {
        return err!("map key type does not match: {:?} vs {:?}", map, key);
    }
    Ok(())
}

fn map_valid(map: compile::Ty, key: compile::Ty, val: compile::Ty) -> Result<()> {
    map_key_valid(map, key)?;
    if map.val()? != val {
        return err!("map value type does not match: {:?} vs {:?}", map, val);
    }
    Ok(())
}
