use crate::{
    builtins,
    bytecode::Accum,
    common::{NumTy, Result},
    compile,
    runtime::UniqueStr,
};

// TODO: move intrinsics module over to codegen, make it CodeGenerator-generic.
//  (fine that some runtime libraries are left behind for the time being in llvm)
// TODO: continue stubbing out gen_inst; filling in missing items. I think the idea is that for
// difficult stuff, we'll add a method that implementors will override, and that this will
// encapsulate code generation "without the branches"
// TODO: migrate LLVM over to gen_inst, get everything passing
// TODO: move LLVM into codegen module
// TODO: start on clir

type SmallVec<T> = smallvec::SmallVec<[T; 4]>;
type Ref = (NumTy, compile::Ty);

struct Sig<C: CodeGenerator> {
    // cstr? that we assert is utf8? bytes that we assert are both utf8 and nul-terminated?
    name: &'static str,
    args: SmallVec<C::Ty>,
    ret: C::Ty,
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

/// CodeGenerator encapsulates common functionality needed to generate instructions across multiple
/// backends. This trait is not currently sufficient to abstract over any backend "end to end" from
/// bytecode instructions all the way to machine code, but it allows us to keep much of the more
/// mundane plumbing work common across all backends (as well as separate safe "glue code" from
/// unsafe calls to the LLVM C API).
pub(crate) trait CodeGenerator {
    type Ty;
    type Val;

    // mappings from compile::Ty to Self::Ty
    fn void_ptr_ty(&self) -> Self::Ty;
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

    /// Call an intrinsic, given a pointer to the [`intrinsics`] module and a list of arguments.
    fn call_intrinsic(&mut self, func: Op, args: &mut [Self::Val]) -> Result<Self::Val>;

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
            IntToStr(sr, ir) => {
                let arg = self.get_val(ir.reflect())?;
                let res = self.call_intrinsic(intrinsic!(int_to_str), &mut [arg])?;
                self.bind_val(sr.reflect(), res)
            }
            StrToInt(ir, sr) => {
                let arg = self.get_val(sr.reflect())?;
                let res = self.call_intrinsic(intrinsic!(str_to_int), &mut [arg])?;
                self.bind_val(ir.reflect(), res)
            }
            HexStrToInt(ir, sr) => {
                let arg = self.get_val(sr.reflect())?;
                let res = self.call_intrinsic(intrinsic!(hex_str_to_int), &mut [arg])?;
                self.bind_val(ir.reflect(), res)
            }
            StrToFloat(fr, sr) => {
                let arg = self.get_val(sr.reflect())?;
                let res = self.call_intrinsic(intrinsic!(str_to_float), &mut [arg])?;
                self.bind_val(fr.reflect(), res)
            }
            FloatToInt(ir, fr) => {
                let arg = self.get_val(fr.reflect())?;
                let res = self.call_intrinsic(Op::FloatToInt, &mut [arg])?;
                self.bind_val(ir.reflect(), res)
            }
            IntToFloat(fr, ir) => {
                let arg = self.get_val(ir.reflect())?;
                let res = self.call_intrinsic(Op::IntToFloat, &mut [arg])?;
                self.bind_val(fr.reflect(), res)
            }
            AddInt(res, l, r) => self.binop(op(Arith::Add, false), res, l, r),
            AddFloat(res, l, r) => self.binop(op(Arith::Add, true), res, l, r),
            MinusInt(res, l, r) => self.binop(op(Arith::Minus, false), res, l, r),
            MinusFloat(res, l, r) => self.binop(op(Arith::Minus, true), res, l, r),
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
            _ => unimplemented!(),
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
