use crate::{
    bytecode::Instr,
    common::{NumTy, Result},
    compile,
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
        crate::llvm::intrinsics::$name as *const u8
    };
}

/// CodeGenerator encapsulates common functionality needed to generate instructions across multiple
/// backends. This trait is not currently sufficient to abstract over any backend "end to end" from
/// bytecode instructions all the way to machine code, but it allows us to keep much of the more
/// mundane plumbing work common across all backends.
pub(crate) trait CodeGenerator {
    type Ty;
    type Val;
    fn void_ptr_ty(&self) -> Self::Ty;
    fn usize_ty(&self) -> Self::Ty;
    fn get_ty(&self, ty: compile::Ty) -> Self::Ty;
    fn bind_val(&mut self, r: Ref, v: Self::Val) -> Result<()>;
    fn get_val(&mut self, r: Ref) -> Result<Self::Val>;
    fn runtime_val(&self) -> Self::Val;
    fn call_intrinsic(&mut self, func: *const u8, args: &mut [Self::Val]) -> Result<Self::Val>;
    fn const_int(&self, i: i64) -> Self::Val;

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

fn gen_inst(cg: &mut impl CodeGenerator, instr: &Instr) -> Result<()> {
    match instr {
        _ => unimplemented!(),
    }
}
