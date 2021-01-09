//! Cranelift code generation for frawk programs.
use cranelift::prelude::*;
use cranelift_codegen::ir::StackSlot;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use hashbrown::{HashMap, HashSet};
use smallvec::SmallVec;

use crate::builtins;
use crate::bytecode::Accum;
use crate::codegen::{Backend, CodeGenerator, Op, Ref, Sig, StrReg};
use crate::common::{CompileError, FileSpec, NumTy, Result};
use crate::compile;
use crate::runtime::{self, UniqueStr};

use std::convert::TryFrom;
use std::mem;

// TODO:
// * figure out declarations, stack slots for strings, and frame setup.
// * control flow, drops, etc.
//
// TODO (cleanup; after tests are passing):
// * move floatfunc/bitwise stuff into llvm module
// * move llvm module under codegen
// * make sure cargo doc builds
// * doc fixups

/// Information about a user-defined function needed by callers.
struct FuncInfo {
    globals: SmallVec<[Ref; 2]>,
    func_id: FuncId,
}

/// Function-independent data used in compilation
struct Shared {
    codegen_ctx: codegen::Context,
    module: JITModule,
    func_ids: Vec<FuncInfo>,
    external_funcs: HashMap<*const u8, FuncId>,
    // We need cranelift Signatures for declaring external functions. We put them here to reuse
    // them across calls to `register_external_fn`.
    sig: Signature,
}

/// A cranelift [`Variable`] with frawk-specific metadata
#[derive(Clone)]
struct VarRef {
    var: Variable,
    is_global: bool,
    skip_drop: bool,
}

/// Iterator-specific variable state. This is treated differently from [`VarRef`] because iterators
/// function in a much more restricted context when compared with variables.
#[derive(Clone)]
struct IterState {
    // NB: a more compact representation would just be to store `start` and `end`, but we need to
    // hold onto the true `start` in order to free the memory.
    len: Variable,  // int
    cur: Variable,  // int
    base: Variable, // pointer
}

/// Function-level state
struct Frame {
    vars: HashMap<Ref, VarRef>,
    iters: HashMap<Ref, IterState>,
    runtime: Variable,
    entry_block: Block,
    n_params: usize,
    n_vars: usize,
}

/// Toplevel information
struct GlobalContext {
    shared: Shared,
    ctx: FunctionBuilderContext,
    funcs: Vec<Frame>,
}

/// The state required for generating code for the function at `f`.
struct View<'a> {
    f: &'a mut Frame,
    builder: FunctionBuilder<'a>,
    shared: &'a mut Shared,
}

macro_rules! external {
    ($name:ident) => {
        crate::codegen::intrinsics::$name as *const u8
    };
}

impl<'a> View<'a> {
    fn stack_slot_bytes(&mut self, bytes: u32) -> StackSlot {
        let data = StackSlotData::new(StackSlotKind::ExplicitSlot, bytes);
        self.builder.create_stack_slot(data)
    }
    fn switch_to_entry(&mut self) -> Result<Block> {
        let last_block = self
            .builder
            .current_block()
            .map(Ok)
            .unwrap_or_else(|| err!("generating instructions without a current block"))?;
        self.builder.switch_to_block(self.f.entry_block);
        Ok(last_block)
    }

    /// Initialize the `Variable` associated with a non-iterator and non-null local variable of
    /// type `ty`.
    fn declare_local(&mut self, ty: compile::Ty) -> Result<Variable> {
        use compile::Ty::*;
        let next_var = Variable::new(self.f.n_vars);
        self.f.n_vars += 1;
        let cl_ty = self.get_ty(ty);
        // NB: should we preinitialize maps? We may have to do that in a separate pass after we
        // finish the pass over the rest of the functions.
        match ty {
            Int | Float => self.builder.declare_var(next_var, cl_ty),
            Str => {
                // allocate a stack slot for the string, then assign next_var to point to that
                // slot.
                let last_block = self.switch_to_entry()?;
                let ptr_ty = self.ptr_to(cl_ty);
                let slot = self.stack_slot_bytes(mem::size_of::<runtime::Str>() as u32);
                self.builder.declare_var(next_var, ptr_ty);
                let addr = self.builder.ins().stack_addr(ptr_ty, slot, 0);
                self.builder.def_var(next_var, addr);
                // For good measure, write zeros here,
                let zeros = self.builder.ins().iconst(cl_ty, 0);
                self.builder.ins().stack_store(zeros, slot, 0);
                self.builder.switch_to_block(last_block);
            }
            MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat | MapStrStr => {
                self.builder.declare_var(next_var, cl_ty);
                let alloc_fn = match ty {
                    MapIntInt => external!(alloc_intint),
                    MapIntFloat => external!(alloc_intfloat),
                    MapIntStr => external!(alloc_intstr),
                    MapStrInt => external!(alloc_strint),
                    MapStrFloat => external!(alloc_strfloat),
                    MapStrStr => external!(alloc_strstr),
                    _ => unreachable!(),
                };
                let last_block = self.switch_to_entry()?;
                let new_map = self.call_external(alloc_fn, &[]);
                self.builder.def_var(next_var, new_map);
                self.builder.switch_to_block(last_block);
            }
            IterInt | IterStr | Null => {
                return err!("invalid type for declare local (iterator or null)")
            }
        }
        Ok(next_var)
    }

    /// Construct an [`IterState`] corresponding to an iterator of type `ty`.
    fn declare_iterator(&mut self, ty: compile::Ty) -> Result<IterState> {
        use compile::Ty::*;
        match ty {
            IterStr | IterInt => {
                let len = Variable::new(self.f.n_vars);
                let cur = Variable::new(self.f.n_vars + 1);
                let base = Variable::new(self.f.n_vars + 2);
                self.f.n_vars += 3;
                self.builder.declare_var(len, types::I64);
                self.builder.declare_var(cur, types::I64);
                self.builder.declare_var(base, self.void_ptr_ty());
                Ok(IterState { len, cur, base })
            }
            Null | Int | Float | Str | MapIntInt | MapIntFloat | MapIntStr | MapStrInt
            | MapStrFloat | MapStrStr => err!(
                "attempting to declare iterator variable for non-iterator type: {:?}",
                ty
            ),
        }
    }

    /// Increment the refcount of the value `v` of type `ty`.
    ///
    /// If `ty` is not an array or string type, this method is a noop.
    fn ref_val(&mut self, ty: compile::Ty, v: Value) {
        use compile::Ty::*;
        let func = match ty {
            MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat | MapStrStr => {
                external!(ref_map)
            }
            Str => external!(ref_str),
            Null | Int | Float | IterInt | IterStr => return,
        };
        self.call_external_void(func, &[v]);
    }

    /// Decrement the refcount of the value `v` of type `ty`.
    ///
    /// If `ty` is not an array or string type, this method is a noop.
    fn drop_val(&mut self, ty: compile::Ty, v: Value) {
        use compile::Ty::*;
        let func = match ty {
            MapIntInt => external!(drop_intint),
            MapIntFloat => external!(drop_intfloat),
            MapIntStr => external!(drop_intstr),
            MapStrInt => external!(drop_strint),
            MapStrFloat => external!(drop_strfloat),
            MapStrStr => external!(drop_strstr),
            Str => external!(drop_str),
            Null | Int | Float | IterInt | IterStr => return,
        };
        self.call_external_void(func, &[v]);
    }

    /// Call and external function that returns a value.
    ///
    /// Panics if `func` has not been registered as an external function, or if it was not
    /// registered as returning a single value.
    fn call_external(&mut self, func: *const u8, args: &[Value]) -> Value {
        let inst = self.call_inst(func, args);
        let mut iter = self.builder.inst_results(inst).iter().cloned();
        let ret = iter.next().expect("expected return value");
        // For now, we expect all functions to have a single return value.
        debug_assert!(iter.next().is_none());
        ret
    }

    /// Call and external function that does not return a value.
    ///
    /// Panics if `func` has not been registered as an external function, or if it was not
    /// registered as returning a single value.
    fn call_external_void(&mut self, func: *const u8, args: &[Value]) {
        let _inst = self.call_inst(func, args);
        debug_assert!(self.builder.inst_results(_inst).iter().next().is_none());
    }

    fn call_inst(&mut self, func: *const u8, args: &[Value]) -> cranelift_codegen::ir::Inst {
        let id = self.shared.external_funcs[&func];
        let fref = self
            .shared
            .module
            .declare_func_in_func(id, self.builder.func);
        self.builder.ins().call(fref, args)
    }

    /// frawk does not have booleans, so for now we always convert the results of comparison
    /// operations back to integers.
    ///
    /// NB: It would be interesting and likely useful to add a "bool" type (with consequent
    /// coercions).
    fn bool_to_int(&mut self, b: Value) -> Value {
        let int_ty = self.get_ty(compile::Ty::Int);
        self.builder.ins().bint(int_ty, b)
    }

    /// Generate a new value according to the comparison instruction, applied to `l` and `r`, which
    /// are assumed to be floating point values if `is_float` and (signed, as is the case in frawk)
    /// integer values otherwise.
    ///
    /// As with the LLVM, we use the "ordered" variants on comparsion: the ones that return false
    /// if either operand is NaN.
    fn cmp(&mut self, op: crate::codegen::Cmp, is_float: bool, l: Value, r: Value) -> Value {
        use crate::codegen::Cmp::*;
        let res = if is_float {
            match op {
                EQ => self.builder.ins().fcmp(FloatCC::Equal, l, r),
                LTE => self.builder.ins().fcmp(FloatCC::LessThanOrEqual, l, r),
                LT => self.builder.ins().fcmp(FloatCC::LessThan, l, r),
                GTE => self.builder.ins().fcmp(FloatCC::GreaterThanOrEqual, l, r),
                GT => self.builder.ins().fcmp(FloatCC::GreaterThan, l, r),
            }
        } else {
            match op {
                EQ => self.builder.ins().icmp(IntCC::Equal, l, r),
                LTE => self.builder.ins().icmp(IntCC::SignedLessThanOrEqual, l, r),
                LT => self.builder.ins().icmp(IntCC::SignedLessThan, l, r),
                GTE => self
                    .builder
                    .ins()
                    .icmp(IntCC::SignedGreaterThanOrEqual, l, r),
                GT => self.builder.ins().icmp(IntCC::SignedGreaterThan, l, r),
            }
        };
        self.bool_to_int(res)
    }

    /// Generate a new value according to the operation specified in `op`.
    ///
    /// We assume that `args` contains floating point or signed integer values depending on the
    /// value of `is_float`. Panics if args has the wrong arity.
    fn arith(&mut self, op: crate::codegen::Arith, is_float: bool, args: &[Value]) -> Value {
        use crate::codegen::Arith::*;
        if is_float {
            match op {
                Mul => self.builder.ins().fmul(args[0], args[1]),
                Minus => self.builder.ins().fsub(args[0], args[1]),
                Add => self.builder.ins().fadd(args[0], args[1]),
                // No floating-point modulo in cranelift?
                Mod => self.call_external(external!(_frawk_fprem), args),
                Neg => self.builder.ins().fneg(args[0]),
            }
        } else {
            match op {
                Mul => self.builder.ins().imul(args[0], args[1]),
                Minus => self.builder.ins().isub(args[0], args[1]),
                Add => self.builder.ins().iadd(args[0], args[1]),
                Mod => self.builder.ins().srem(args[0], args[1]),
                Neg => self.builder.ins().ineg(args[0]),
            }
        }
    }

    /// Apply the bitwise operation specified in `op` to `args`.
    ///
    /// Panics if args has the wrong arity (2 for all bitwise operations except for `Complement`).
    /// All of the entries in `args` should be integer values.
    fn bitwise(&mut self, op: builtins::Bitwise, args: &[Value]) -> Value {
        use builtins::Bitwise::*;
        match op {
            Complement => self.builder.ins().bnot(args[0]),
            And => self.builder.ins().band(args[0], args[1]),
            Or => self.builder.ins().bor(args[0], args[1]),
            LogicalRightShift => self.builder.ins().ushr(args[0], args[1]),
            ArithmeticRightShift => self.builder.ins().ushr(args[0], args[1]),
            LeftShift => self.builder.ins().ishl(args[0], args[1]),
            Xor => self.builder.ins().bxor(args[0], args[1]),
        }
    }

    /// Apply the [`FloatFunc`] operation specified in `op` to `args`.
    ///
    /// Panics if args has the wrong arity. Unlike LLVM, most of these functions do not have direct
    /// instructions (or intrinsics), so they are implemented as function calls to rust functions
    /// which in turn call into the standard library.
    fn floatfunc(&mut self, op: builtins::FloatFunc, args: &[Value]) -> Value {
        use builtins::FloatFunc::*;
        match op {
            Cos => self.call_external(external!(_frawk_cos), args),
            Sin => self.call_external(external!(_frawk_sin), args),
            Atan => self.call_external(external!(_frawk_atan), args),
            Atan2 => self.call_external(external!(_frawk_atan2), args),
            Log => self.call_external(external!(_frawk_log), args),
            Log2 => self.call_external(external!(_frawk_log2), args),
            Log10 => self.call_external(external!(_frawk_log10), args),
            Sqrt => self.builder.ins().sqrt(args[0]),
            Exp => self.call_external(external!(_frawk_exp), args),
        }
    }

    fn get_iter(&mut self, iter: Ref) -> Result<IterState> {
        if let Some(x) = self.f.iters.get(&iter).cloned() {
            Ok(x)
        } else {
            let next = self.declare_iterator(iter.1)?;
            self.f.iters.insert(iter, next.clone());
            Ok(next)
        }
    }

    /// Process arguments to printf and sprintf into the form accepted by the corresponding
    /// functions in the `intrinsics` module. We allocate space on the stack for the arguments as
    /// well as information about the arguments' types and then store into those slots.
    fn bundle_printf_args(
        &mut self,
        args: &Vec<Ref>,
    ) -> Result<(
        /* args */ StackSlot,
        /* types */ StackSlot,
        /* len */ Value,
    )> {
        let len = i32::try_from(args.len()).expect("too many arguments to print_all") as u32;
        let slot_size = mem::size_of::<usize>() as i32;
        // allocate an array for arguments on the stack
        let arg_slot = self.stack_slot_bytes(
            len.checked_mul(slot_size as u32)
                .expect("too many arguments to print_all"),
        );
        // and for argument types
        let type_slot = self.stack_slot_bytes(mem::size_of::<u32>() as u32 * len);

        // Store arguments and types into the corresponding stack slot.
        for (ix, rf) in args.iter().cloned().enumerate() {
            let ix = ix as i32;
            let arg = self.get_val(rf)?;
            let ty = self.builder.ins().iconst(types::I32, rf.1 as i64);
            self.builder
                .ins()
                .stack_store(arg, arg_slot, ix * slot_size);
            self.builder
                .ins()
                .stack_store(ty, type_slot, ix * mem::size_of::<u32>() as i32);
        }
        let num_args = self.const_int(len as _);
        Ok((arg_slot, type_slot, num_args))
    }

    /// Get the `VarRef` bound to `r` in the current frame, otherwise allocate a fresh local
    /// variable and insert it into the frame's bindings.
    fn get_var_default_local(&mut self, r: Ref) -> Result<VarRef> {
        if let Some(v) = self.f.vars.get(&r) {
            Ok(v.clone())
        } else {
            let var = self.declare_local(r.1)?;
            let vref = VarRef {
                var,
                is_global: false,
                skip_drop: false,
            };
            self.f.vars.insert(r, vref.clone());
            Ok(vref)
        }
    }
}

// For Cranelift, we need to register function names in a lookup table before constructing a
// module, so we actually implement `Backend` twice for each registration step.

struct RegistrationState {
    builder: JITBuilder,
}

impl Backend for RegistrationState {
    type Ty = ();
    fn void_ptr_ty(&self) -> () {
        ()
    }
    fn ptr_to(&self, (): ()) -> () {
        ()
    }
    fn usize_ty(&self) -> () {
        ()
    }
    fn u32_ty(&self) -> () {
        ()
    }
    fn get_ty(&self, _ty: compile::Ty) -> () {
        ()
    }

    fn register_external_fn(
        &mut self,
        name: &'static str,
        _name_c: *const u8,
        addr: *const u8,
        _sig: Sig<Self>,
    ) -> Result<()> {
        self.builder.symbol(name, addr);
        Ok(())
    }
}

impl<'a> Backend for View<'a> {
    type Ty = Type;
    // mappings from compile::Ty to Self::Ty
    fn void_ptr_ty(&self) -> Self::Ty {
        self.shared.module.target_config().pointer_type()
    }
    fn ptr_to(&self, _ty: Self::Ty) -> Self::Ty {
        // Cranelift pointers are all a single type, though we may eventually need to care more
        // about "references", which cranelift uses to compute stack maps.
        self.void_ptr_ty()
    }
    fn usize_ty(&self) -> Self::Ty {
        // assume pointers are 64 bits
        types::I64
    }
    fn u32_ty(&self) -> Self::Ty {
        types::I32
    }
    fn get_ty(&self, ty: compile::Ty) -> Self::Ty {
        use compile::Ty::*;
        match ty {
            Null | Int => types::I64,
            Float => types::F64,
            Str => types::I128,
            MapIntInt | MapIntFloat | MapIntStr => self.void_ptr_ty(),
            MapStrInt | MapStrFloat | MapStrStr => self.void_ptr_ty(),
            IterInt | IterStr => panic!("taking the type of an iterator"),
        }
    }

    fn register_external_fn(
        &mut self,
        name: &'static str,
        _name_c: *const u8,
        addr: *const u8,
        sig: Sig<Self>,
    ) -> Result<()> {
        let cl_sig = &mut self.shared.sig;
        cl_sig.params.clear();
        cl_sig.returns.clear();
        cl_sig
            .params
            .extend(sig.args.iter().cloned().map(AbiParam::new));
        cl_sig
            .returns
            .extend(sig.ret.as_ref().into_iter().cloned().map(AbiParam::new));
        let id = self
            .shared
            .module
            .declare_function(name, Linkage::Import, cl_sig)
            .map_err(|e| {
                CompileError(format!(
                    "error declaring {} in module: {}",
                    name,
                    e.to_string()
                ))
            })?;
        self.shared.external_funcs.insert(addr, id);
        Ok(())
    }
}

impl<'a> CodeGenerator for View<'a> {
    type Val = Value;

    // mappings to and from bytecode-level registers to IR-level values
    fn bind_val(&mut self, r: Ref, v: Self::Val) -> Result<()> {
        use compile::Ty::*;
        let VarRef { var, is_global, .. } = self.get_var_default_local(r)?;
        match r.1 {
            Int | Float => {
                if is_global {
                    let p = self.builder.use_var(var);
                    self.builder.ins().store(MemFlags::trusted(), v, p, 0);
                } else {
                    self.builder.def_var(var, v);
                }
            }
            Str => {
                // NB: we assume that `v` is a string, not a pointer to a string.

                // For now, we treat globals and locals the same for strings.
                // TODO: Hopefully the stack slot mechanics don't ruin all of that...

                // first, drop the value currently in the pointer
                let p = self.builder.use_var(var);
                self.drop_val(Str, p);
                self.builder.ins().store(MemFlags::trusted(), v, p, 0);
            }
            MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat | MapStrStr => {
                // first, ref the new value
                // TODO: can we skip the ref here?
                self.ref_val(r.1, v);
                if is_global {
                    // then, drop the value currently in the pointer
                    let p = self.builder.use_var(var);
                    let pointee = self
                        .builder
                        .ins()
                        // TODO: should this be a pointer type?
                        .load(types::I64, MemFlags::trusted(), p, 0);
                    self.drop_val(r.1, pointee);

                    // And slot the new value in
                    self.builder.ins().store(MemFlags::trusted(), v, p, 0);
                } else {
                    let cur = self.builder.use_var(var);
                    self.drop_val(r.1, cur);
                    self.builder.def_var(var, v);
                }
            }
            Null => {}
            IterInt | IterStr => return err!("attempting to store an iterator value"),
        }
        Ok(())
    }
    fn get_val(&mut self, r: Ref) -> Result<Self::Val> {
        use compile::Ty::*;
        if let Null = r.1 {
            return Ok(self.const_int(0));
        }

        let VarRef { var, is_global, .. } = self.get_var_default_local(r)?;
        let val = self.builder.use_var(var);

        match r.1 {
            MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat | MapStrStr | Int
            | Float => {
                if is_global {
                    let ty = self.get_ty(r.1);
                    Ok(self.builder.ins().load(ty, MemFlags::trusted(), val, 0))
                } else {
                    Ok(val)
                }
            }
            Str => Ok(val),
            IterInt | IterStr => err!("attempting to load an iterator pointer"),
            Null => Ok(self.const_int(0)),
        }
    }

    // backend-specific handling of constants and low-level operations.
    fn runtime_val(&mut self) -> Self::Val {
        self.builder.use_var(self.f.runtime)
    }
    fn const_int(&mut self, i: i64) -> Self::Val {
        self.builder.ins().iconst(types::I64, i)
    }
    fn const_float(&mut self, f: f64) -> Self::Val {
        self.builder.ins().f64const(f)
    }
    fn const_str<'b>(&mut self, s: &UniqueStr<'b>) -> Self::Val {
        // iconst does not support I128, so we concatenate two I64 constants.
        let bits: u128 = s.clone_str().into_bits();
        let low = bits as i64;
        let high = (bits >> 64) as i64;
        let low_v = self.builder.ins().iconst(types::I64, low);
        let high_v = self.builder.ins().iconst(types::I64, high);
        self.builder.ins().iconcat(low_v, high_v)
    }
    fn const_ptr<'b, T>(&'b mut self, c: &'b T) -> Self::Val {
        self.const_int(c as *const _ as i64)
    }

    fn call_void(&mut self, func: *const u8, args: &mut [Self::Val]) -> Result<()> {
        Ok(self.call_external_void(func, args))
    }

    // TODO if all goes well, remove the Result<..> wrapper and migrate the callers.
    fn call_intrinsic(&mut self, func: Op, args: &mut [Self::Val]) -> Result<Self::Val> {
        use Op::*;
        match func {
            Cmp { is_float, op } => Ok(self.cmp(op, is_float, args[0], args[1])),
            Arith { is_float, op } => Ok(self.arith(op, is_float, args)),
            Bitwise(bw) => Ok(self.bitwise(bw, args)),
            Math(ff) => Ok(self.floatfunc(ff, args)),
            Div => Ok(self.builder.ins().fdiv(args[0], args[1])),
            Pow => Ok(self.call_external(external!(_frawk_pow), args)),
            FloatToInt => {
                let ty = self.get_ty(compile::Ty::Int);
                Ok(self.builder.ins().fcvt_to_sint_sat(ty, args[0]))
            }
            IntToFloat => {
                let ty = self.get_ty(compile::Ty::Float);
                Ok(self.builder.ins().fcvt_from_sint(ty, args[0]))
            }
            Intrinsic(e) => Ok(self.call_external(e, args)),
        }
    }

    // var-arg printing functions. The arguments here directly parallel the instruction
    // definitions.

    fn printf(
        &mut self,
        output: &Option<(StrReg, FileSpec)>,
        fmt: &StrReg,
        args: &Vec<Ref>,
    ) -> Result<()> {
        let (arg_slot, type_slot, num_args) = self.bundle_printf_args(args)?;

        let rt = self.runtime_val();
        let ty = self.void_ptr_ty();
        let arg_addr = self.builder.ins().stack_addr(ty, arg_slot, 0);
        let ty_addr = self.builder.ins().stack_addr(ty, type_slot, 0);
        let fmt = self.get_val(fmt.reflect())?;

        if let Some((out, spec)) = output {
            let output = self.get_val(out.reflect())?;
            let fspec = self.const_int(*spec as _);
            self.call_external_void(
                external!(printf_impl_file),
                &[rt, fmt, arg_addr, ty_addr, num_args, output, fspec],
            )
        } else {
            self.call_external_void(
                external!(printf_impl_stdout),
                &[rt, fmt, arg_addr, ty_addr, num_args],
            )
        }
        Ok(())
    }

    fn sprintf(&mut self, dst: &StrReg, fmt: &StrReg, args: &Vec<Ref>) -> Result<()> {
        let (arg_slot, type_slot, num_args) = self.bundle_printf_args(args)?;

        let rt = self.runtime_val();
        let ty = self.void_ptr_ty();
        let arg_addr = self.builder.ins().stack_addr(ty, arg_slot, 0);
        let ty_addr = self.builder.ins().stack_addr(ty, type_slot, 0);
        let fmt = self.get_val(fmt.reflect())?;

        let res = self.call_external(
            external!(sprintf_impl),
            &[rt, fmt, arg_addr, ty_addr, num_args],
        );
        self.bind_val(dst.reflect(), res)
    }

    fn print_all(&mut self, output: &Option<(StrReg, FileSpec)>, args: &Vec<StrReg>) -> Result<()> {
        // NB: Unlike LLVM, we do not generate custom stub methods here, we just inline the the
        // "var args" implementation.

        // First, allocate an array for all of the arguments on the stack.
        let len = i32::try_from(args.len()).expect("too many arguments to print_all") as u32;
        let slot_size = mem::size_of::<usize>() as i32;
        let bytes = len
            .checked_mul(slot_size as u32)
            .expect("too many arguments to print_all");
        let slot = self.stack_slot_bytes(bytes);

        // Now, store pointers to each of the strings into the array.
        for (ix, reg) in args.iter().cloned().enumerate() {
            let arg = self.get_val(reg.reflect())?;
            self.builder
                .ins()
                .stack_store(arg, slot, ix as i32 * slot_size);
        }

        let rt = self.runtime_val();
        let ty = self.void_ptr_ty();
        let addr = self.builder.ins().stack_addr(ty, slot, 0);
        let num_args = self.const_int(len as _);

        if let Some((out, spec)) = output {
            let output = self.get_val(out.reflect())?;
            let fspec = self.const_int(*spec as _);
            self.call_external_void(
                external!(print_all_file),
                &[rt, addr, num_args, output, fspec],
            );
        } else {
            self.call_external_void(external!(print_all_stdout), &[rt, addr, num_args]);
        }
        Ok(())
    }

    fn mov(&mut self, ty: compile::Ty, dst: NumTy, src: NumTy) -> Result<()> {
        use compile::Ty::*;
        let src = self.get_val((src, ty))?;
        match ty {
            Int | Float => self.bind_val((dst, ty), src)?,
            Str => {
                self.call_external_void(external!(ref_str), &[src]);
                let str_ty = self.get_ty(Str);
                let loaded = self.builder.ins().load(str_ty, MemFlags::trusted(), src, 0);
                self.bind_val((dst, Str), loaded)?;
            }
            MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat | MapStrStr => {
                self.call_external_void(external!(ref_map), &[src]);
                self.bind_val((dst, ty), src)?;
            }
            IterInt | IterStr => return err!("attempting to apply `mov` to an iterator!"),
            Null => {
                let zero = self.const_int(0);
                self.bind_val((dst, ty), zero)?;
            }
        }
        Ok(())
    }

    fn iter_begin(&mut self, dst: Ref, map: Ref) -> Result<()> {
        use compile::Ty::*;
        let (len_fn, begin_fn) = match map.1 {
            MapIntInt => (external!(len_intint), external!(iter_intint)),
            MapIntStr => (external!(len_intstr), external!(iter_intstr)),
            MapIntFloat => (external!(len_intfloat), external!(iter_intfloat)),
            MapStrInt => (external!(len_strint), external!(iter_strint)),
            MapStrStr => (external!(len_strstr), external!(iter_strstr)),
            MapStrFloat => (external!(len_strfloat), external!(iter_strfloat)),
            IterInt | IterStr | Int | Float | Str | Null => {
                return err!("iterating over non-map type: {:?}", map.1)
            }
        };
        let map = self.get_val(map)?;
        let IterState { len, cur, base } = self.get_iter(dst)?;
        let ptr = self.call_external(begin_fn, &[map]);
        let map_len = self.call_external(len_fn, &[map]);
        let zero = self.const_int(0);
        self.builder.def_var(cur, zero);
        self.builder.def_var(len, map_len);
        self.builder.def_var(base, ptr);
        Ok(())
    }

    fn iter_hasnext(&mut self, dst: Ref, iter: Ref) -> Result<()> {
        let IterState { len, cur, .. } = self.get_iter(iter)?;
        let lenv = self.builder.use_var(len);
        let curv = self.builder.use_var(cur);
        let cmp = self.builder.ins().icmp(IntCC::UnsignedLessThan, curv, lenv);
        let cmp_int = self.bool_to_int(cmp);
        self.bind_val(dst, cmp_int)
    }

    fn iter_getnext(&mut self, dst: Ref, iter: Ref) -> Result<()> {
        // Compute base+cur and load it into a value
        let IterState { cur, base, .. } = self.get_iter(iter)?;
        let base = self.builder.use_var(base);
        let cur = self.builder.use_var(cur);
        let ptr = self.builder.ins().iadd(base, cur);
        let ty = self.get_ty(dst.1);
        let contents = self.builder.ins().load(ty, MemFlags::trusted(), ptr, 0);

        // Now bind it to `dst` and increment the refcount, if relevant.
        self.bind_val(dst, contents)?;
        let dst_ptr = self.get_val(dst)?;
        self.ref_val(dst.1, dst_ptr);
        Ok(())
    }

    fn var_loaded(&mut self, dst: Ref) -> Result<()> {
        // `bind_val` will ref any maps that we store. Maps that come as the result of function
        // calls (e.g. load_var_intmap) will not have been stored elsewhere, and will have a
        // refcount already incremented.
        if dst.1.is_array() {
            let v = self.get_val(dst)?;
            self.drop_val(dst.1, v);
        }
        Ok(())
    }
}
