//! Cranelift code generation for frawk programs.
use cranelift::prelude::*;
use cranelift_module::{FuncId, Module};
use cranelift_simplejit::SimpleJITModule;
use hashbrown::HashMap;

use crate::codegen::{CodeGenerator, Op, Ref, Sig, StrReg};
use crate::common::{FileSpec, NumTy, Result};
use crate::compile;
use crate::runtime::UniqueStr;

// TODO: IntrinsicMap equivalent (convert signatures)
// TODO: locals => Variable map
//       globals referenced as loads to a global variable.
//       stores of ref-counted type are refs
//          (except for eliminated phi nodes)
//      things should, on the whole, be simpler (knock on wood)
//
//
// so, concretely. Let's keep the same Generator/View structure, but this time encapsulate all the
// shit into a Shared struct
//
// * implement type maps. (or perhapss we don't need them?)
// * declare all functions.
// * fill in CodeGenerator trait impl for View

/// Function-independent data used in compilation
struct Shared {
    codegen_ctx: codegen::Context,
    module: SimpleJITModule,
    func_ids: Vec<FuncId>,
}

/// A cranelift [`Variable`] with frawk-specific metadata
struct VarRef {
    var: Variable,
    is_global: bool,
    skip_drop: bool,
}

/// Function-level state
#[derive(Default)]
struct Frame {
    globals: HashMap<Ref, VarRef>,
    locals: HashMap<Ref, VarRef>,
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
    ctx: &'a mut FunctionBuilderContext,
    shared: &'a mut Shared,
}

impl<'a> CodeGenerator for View<'a> {
    type Ty = Type;
    type Val = Value;

    /// Register a function with address `addr` and name `name` (/ `name_c`, the null-terminated
    /// variant) with signature `Sig` to be called.
    fn register_external_fn(
        &mut self,
        name: &'static str,
        name_c: *const u8,
        addr: *const u8,
        sig: Sig<Self>,
    ) -> Result<()> {
        unimplemented!()
    }

    // mappings from compile::Ty to Self::Ty
    fn void_ptr_ty(&self) -> Self::Ty {
        self.shared.module.target_config().pointer_type()
    }
    fn ptr_to(&self, _ty: Self::Ty) -> Self::Ty {
        // Cranelift pointers are all a single type.
        self.void_ptr_ty()
    }
    fn usize_ty(&self) -> Self::Ty {
        // assume pointerse are 64 bits
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

    // mappings to and from bytecode-level registers to IR-level values
    fn bind_val(&mut self, r: Ref, v: Self::Val) -> Result<()> {
        unimplemented!()
    }
    fn get_val(&mut self, r: Ref) -> Result<Self::Val> {
        // * check locals, if it's there, return the pointer directly.
        // * if it is not, then check globals and issue the load
        unimplemented!()
    }

    // backend-specific handling of constants and low-level operations.
    fn runtime_val(&self) -> Self::Val {
        unimplemented!()
    }
    fn const_int(&self, i: i64) -> Self::Val {
        unimplemented!()
    }
    fn const_float(&self, f: f64) -> Self::Val {
        unimplemented!()
    }
    fn const_str<'b>(&self, s: &UniqueStr<'b>) -> Self::Val {
        unimplemented!()
    }
    fn const_ptr<'b, T>(&'b self, c: &'b T) -> Self::Val {
        unimplemented!()
    }

    /// Call an intrinsic, given a pointer to the [`intrinsics`] module and a list of arguments.
    fn call_intrinsic(&mut self, func: Op, args: &mut [Self::Val]) -> Result<Self::Val> {
        unimplemented!()
    }

    // var-arg printing functions. The arguments here directly parallel the instruction
    // definitions.

    fn printf(
        &mut self,
        output: &Option<(StrReg, FileSpec)>,
        fmt: &StrReg,
        args: &Vec<Ref>,
    ) -> Result<()> {
        unimplemented!()
    }

    fn sprintf(&mut self, dst: &StrReg, fmt: &StrReg, args: &Vec<Ref>) -> Result<()> {
        unimplemented!()
    }

    fn print_all(&mut self, output: &Option<(StrReg, FileSpec)>, args: &Vec<StrReg>) -> Result<()> {
        unimplemented!()
    }

    /// Moves the contents of `src` into `dst`, taking refcounts into consideration if necessary.
    fn mov(&mut self, ty: compile::Ty, dst: NumTy, src: NumTy) -> Result<()> {
        unimplemented!()
    }

    /// Constructs an iterator over the keys of `map` and stores it in `dst`.
    fn iter_begin(&mut self, dst: Ref, map: Ref) -> Result<()> {
        unimplemented!()
    }

    /// Queries the iterator in `iter` as to whether any elements remain, stores the result in the
    /// `dst` register.
    fn iter_hasnext(&mut self, dst: Ref, iter: Ref) -> Result<()> {
        unimplemented!()
    }

    /// Advances the iterator in `iter` to the next element and stores the current element in `dst`
    fn iter_getnext(&mut self, dst: Ref, iter: Ref) -> Result<()> {
        unimplemented!()
    }

    // The plumbing for builtin variable manipulation is mostly pretty wrote ... anything we can do
    // here?

    /// Method called after loading a builtin variable into `dst`.
    ///
    /// This is included to help clean up ref-counts on string or map builtins, if necessary.
    fn var_loaded(&mut self, dst: Ref) -> Result<()> {
        unimplemented!()
    }
}
