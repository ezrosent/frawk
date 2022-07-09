//! Cranelift code generation for frawk programs.
//!
//! A few notes on how frawk typed IR is translated into CLIF (from which cranelift can JIT machine
//! code):
//! * Integers are I64s
//! * Floats are F64s
//! * Strings are I128s
//! * Maps are I64s (pointers, in actuality, but we have no need for cranelift's special handling
//! of reference types)
//! * Iterators are separate variables for the base pointer, the current offset, and the length of
//! the array of keys. We can get away without packaging these together into their own stack slots
//! because iterators are always local to the current function scope.
//!
//! Global variables are allocated on the entry function's stack and passed as extra function
//! parameters to the main function and UDFs. We include metadata in [`VarRef`] to ensure we can
//! emit separate code for assignments into global and local variables, as necessary.
//!
//! Strings are passed "by reference" to functions, so we explicitly allocate string variables on
//! the stack and then pass pointers to them.
use cranelift::prelude::*;
use cranelift_codegen::ir::StackSlot;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};
use hashbrown::HashMap;
use smallvec::{smallvec, SmallVec};

use crate::builtins;
use crate::bytecode::Accum;
use crate::codegen::{
    intrinsics, Backend, CodeGenerator, Config, Handles, Jit, Op, Ref, Sig, StrReg,
};
use crate::common::{traverse, CompileError, Either, FileSpec, NodeIx, NumTy, Result, Stage};
use crate::compile::{self, Typer};
use crate::runtime::{self, UniqueStr};

use std::convert::TryFrom;
use std::mem;

/// Information about a user-defined function needed by callers.
#[derive(Clone)]
struct FuncInfo {
    globals: SmallVec<[Ref; 2]>,
    func_id: FuncId,
}

const PLACEHOLDER: Ref = (compile::UNUSED, compile::Ty::Null);

// for debugging
const DUMP_IR: bool = false;

/// After a function is declared, some additional information is required to map parameteres to
/// variables. `Prelude` contains that information.
struct Prelude {
    /// The Cranelift-level signature for the function
    sig: Signature,
    /// The frawk-level Refs corresponding to each of the parameters enumerated in `sig`. Includes
    /// a placeholder for the final parameter containing the frawk runtime for more convenient
    /// iteration.
    refs: SmallVec<[Ref; 4]>,
    /// The number of parameters that contain "true function arguments". These are passed first.
    /// The rest of the parameters correspond to global variables and the runtime. Because the
    /// runtime is always the last parameter, a single offset is enough to identify all three
    /// classes of parameters.
    n_args: usize,
}
/// A cranelift [`Variable`] with frawk-specific metadata
#[derive(Clone)]
struct VarRef {
    var: Variable,
    kind: VarKind,
}

/// The different kinds of variables we track
#[derive(Copy, Clone)]
enum VarKind {
    /// Variables defined locally in the a function. Under some circumstances, we do not drop these
    /// variables (e.g. at join points, or when returning them from a function).
    Local { skip_drop: bool },
    /// frawk-level function parameters. These are treated very similarly to local variables with
    /// skip_drop set to true, the only difference is that parameters are reffed before being
    /// returned from a function, whereas we simply skip dropping.
    Param,
    /// Global variables. These are treated specially throughout, as even integers and floats are
    /// passed by reference. Like params, these are reffed before being returned.
    Global,
}

impl VarKind {
    fn skip_drop(&mut self) {
        if let VarKind::Local { skip_drop } = self {
            *skip_drop = true;
        }
    }
}

/// Iterator-specific variable state. This is treated differently from [`VarRef`] because iterators
/// function in a much more restricted context when compared with variables.
#[derive(Clone)]
struct IterState {
    // NB: a more compact representation would just be to store `start` and `end`, but we need to
    // hold onto the true `start` in order to free the memory.
    bytes: Variable, // int, the length of the map multiplied by the type size
    cur: Variable,   // int, the current byte offset of the iteration
    base: Variable,  // pointer
}

/// Function-level state
struct Frame {
    vars: HashMap<Ref, VarRef>,
    iters: HashMap<Ref, IterState>,
    header_actions: Vec<EntryDeclaration>,
    runtime: Variable,
    // The entry block is the entry to the function. It is filled in first and contains argument
    // initialization code. It jumps unconditionally to the header block.
    entry_block: Block,
    // The header block is filled in last. It contains any local variable initializations, we
    // "discover" which initializations are required during code generation.
    //
    // Neither the entry block nor the header block contain any explicit branches, but Cranelift
    // requires that basic blocks are in a more-or-less finished state before jumping away from
    // them (LLVM does not have this restriction, so the code for that backend is structured
    // somewhat differently).
    header_block: Block,
    n_params: usize,
    n_vars: usize,
}

/// Function-independent data used in compilation
struct Shared {
    module: JITModule,
    func_ids: Vec<Option<FuncInfo>>,
    external_funcs: HashMap<*const u8, FuncId>,
    // We need cranelift Signatures for declaring external functions. We put them here to reuse
    // them across calls to `register_external_fn`.
    sig: Signature,
    handles: Handles,
}

/// Toplevel information
pub(crate) struct Generator {
    shared: Shared,
    ctx: FunctionBuilderContext,
    cctx: codegen::Context,
    funcs: Vec<Option<Prelude>>,
    mains: Stage<FuncId>,
}

/// The state required for generating code for the function at `f`.
struct View<'a> {
    f: Frame,
    builder: FunctionBuilder<'a>,
    shared: &'a mut Shared,
}

/// A specification of a declaration to append to the top of the function.
///
/// We cannot simply `switch_to_block` back to the entry block as we do in LLVM and append the
/// instructions immediately as we discover they are needed because the Cranelift frontend requires
/// that the current block is fully terminated before we can switch to another one.
struct EntryDeclaration {
    var: Variable,
    ty: compile::Ty,
}

/// Map frawk-level type to its type when passed as a parameter to a cranelift function.
///
/// Iterator types are disallowed; they are never passed as function parameterse.
fn ty_to_param(ty: compile::Ty, ptr_ty: Type) -> Result<AbiParam> {
    use compile::Ty::*;
    let clif_ty = match ty {
        Null | Int => types::I64,
        Float => types::F64,
        MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat | MapStrStr | Str => ptr_ty,
        IterInt | IterStr => return err!("attempt to take iterator as parameter"),
        // We assume that null parameters are omitted from the argument list ahead of time
        // Null => return err!("attempt to take null as parameter"),
    };
    Ok(AbiParam::new(clif_ty))
}

/// Map frawk-level types to a cranelift-level type.
///
/// Iterator types are disallowed; they do not correspond to a single type in cranelift and must be
/// handled specially.
fn ty_to_clifty(ty: compile::Ty, ptr_ty: Type) -> Result<Type> {
    use compile::Ty::*;
    match ty {
        Null | Int => Ok(types::I64),
        Float => Ok(types::F64),
        Str => Ok(types::I128),
        MapIntInt | MapIntFloat | MapIntStr => Ok(ptr_ty),
        MapStrInt | MapStrFloat | MapStrStr => Ok(ptr_ty),
        IterInt | IterStr => err!("taking type of an iterator"),
    }
}

impl Jit for Generator {
    fn main_pointers(&mut self) -> Result<Stage<*const u8>> {
        Ok(self
            .mains
            .map_ref(|id| self.shared.module.get_finalized_function(*id)))
    }
}

fn jit_builder() -> Result<JITBuilder> {
    // Adapted from the cranelift source.
    let mut flag_builder = settings::builder();
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "aarch64")] {
            // See https://github.com/bytecodealliance/wasmtime/issues/2735
            flag_builder.set("is_pic", "false").unwrap();
            // Notes from cranelift source: "On at least AArch64, 'colocated' calls use
            // shorter-range relocations, which might not reach all definitions; we
            // can't handle that here, so we require long-range relocation types."
            flag_builder.set("use_colocated_libcalls", "false").unwrap();
        } else {
            flag_builder.set("is_pic", "true").unwrap();
        }
    }
    let isa_builder = cranelift_native::builder()
        .map_err(|msg| err_raw!("host machine is not supported by cranelift: {}", msg))?;
    flag_builder.enable("enable_llvm_abi_extensions").unwrap();
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| err_raw!("failed to initialize cranelift isa: {:?}", e))?;
    Ok(JITBuilder::with_isa(isa, default_libcall_names()))
}

impl Generator {
    pub(crate) fn init(typer: &mut Typer, _config: Config) -> Result<Generator> {
        let builder = jit_builder()?;
        let mut regstate = RegistrationState { builder };
        intrinsics::register_all(&mut regstate)?;
        let module = JITModule::new(regstate.builder);
        let cctx = module.make_context();
        let shared = Shared {
            module,
            func_ids: Default::default(),
            external_funcs: Default::default(),
            sig: cctx.func.signature.clone(),
            handles: Default::default(),
        };
        let mut global = Generator {
            shared,
            ctx: FunctionBuilderContext::new(),
            cctx,
            funcs: Default::default(),
            // placeholder
            mains: Stage::Main(FuncId::from_u32(0)),
        };
        global.define_functions(typer)?;
        let stage = match typer.stage() {
            Stage::Main(main) => Stage::Main(global.define_main_function("__frawk_main", main)?),
            Stage::Par {
                begin,
                main_loop,
                end,
            } => Stage::Par {
                begin: traverse(
                    begin.map(|off| global.define_main_function("__frawk_begin", off)),
                )?,
                main_loop: traverse(
                    main_loop.map(|off| global.define_main_function("__frawk_main_loop", off)),
                )?,
                end: traverse(end.map(|off| global.define_main_function("__frawk_end", off)))?,
            },
        };
        global.mains = stage;
        global.shared.module.finalize_definitions();
        Ok(global)
    }

    /// We get a set of UDFs that are idenfitied as "toplevel" or "main", but these will probably
    /// reference global variables, which we have compiled to take as function parameters. This
    /// method allocates those globals on the stack (while still taking a pointer to the runtime as
    /// a parameter) and then calls into that function.
    fn define_main_function(&mut self, name: &str, udf: usize) -> Result<FuncId> {
        // first, synthesize a `Prelude` matching a main function. This is pretty simple.
        let mut sig = Signature::new(isa::CallConv::SystemV);
        let ptr_ty = self.shared.module.target_config().pointer_type();
        sig.params.push(AbiParam::new(ptr_ty));
        let res = self
            .shared
            .module
            .declare_function(name, Linkage::Export, &sig)
            .map_err(|e| CompileError(format!("failed to declare main function: {}", e)))?;
        let prelude = Prelude {
            sig,
            refs: smallvec![PLACEHOLDER],
            n_args: 0,
        };

        // And now we allocate global variables. First, grab the globals we need from the FuncInfo
        // stored for `udf`.
        let globals = self.shared.func_ids[udf as usize]
            .as_ref()
            .unwrap()
            .globals
            .clone();

        // We'll keep track of these variables and their types so we can drop them at the end.
        let mut vars = Vec::with_capacity(globals.len());

        // We'll reuse the existing function building machinery.
        let mut view = self.create_view(prelude);
        view.builder.switch_to_block(view.f.header_block);
        view.builder.seal_block(view.f.header_block);

        // Now, build each global variable "by hand". Allocate a variable for it, assign it to the
        // address of a default value of the type in question on the stack.
        for (reg, ty) in globals {
            let var = Variable::new(view.f.n_vars);
            vars.push((var, ty));
            view.f.n_vars += 1;
            let cl_ty = view.get_ty(ty);
            let ptr_ty = view.ptr_to(cl_ty);
            view.builder.declare_var(var, ptr_ty);

            let slot = view.stack_slot_bytes(cl_ty.lane_bits() as u32 / 8);
            let default = view.default_value(ty)?;
            if let compile::Ty::Str = ty {
                view.store_string(slot, default);
            } else {
                view.builder.ins().stack_store(default, slot, 0);
            }

            let addr = view.builder.ins().stack_addr(ptr_ty, slot, 0);
            view.builder.def_var(var, addr);
            view.f.vars.insert(
                (reg, ty),
                VarRef {
                    var,
                    kind: VarKind::Global,
                },
            );
        }

        view.call_udf(NumTy::try_from(udf).expect("function Id too large"), &[])?;
        for (var, ty) in vars {
            let val = view.builder.use_var(var);
            view.drop_val(ty, val);
        }
        view.builder.ins().return_(&[]);
        view.builder.finalize();
        mem::drop(view);
        self.define_cur_function(res)?;
        Ok(res)
    }

    fn define_cur_function(&mut self, id: FuncId) -> Result<()> {
        self.shared
            .module
            .define_function(id, &mut self.cctx)
            .map_err(|e| CompileError(e.to_string()))?;
        self.shared.module.clear_context(&mut self.cctx);
        Ok(())
    }

    fn define_functions(&mut self, typer: &mut Typer) -> Result<()> {
        self.declare_local_funcs(typer)?;
        for (i, frame) in typer.frames.iter().enumerate() {
            if let Some(prelude) = self.funcs[i].take() {
                let mut view = self.create_view(prelude);
                if i == 0 {
                    intrinsics::register_all(&mut view)?;
                }
                view.gen_function_body(frame)?;
                // func_id and prelude entries should be initialized in lockstep.
                let id = self.shared.func_ids[i].as_ref().unwrap().func_id;
                self.define_cur_function(id)?;
            }
        }
        Ok(())
    }

    /// Initialize a new user-defined function and prepare it for full code generation.
    fn create_view(&mut self, Prelude { sig, refs, n_args }: Prelude) -> View {
        // Initialize a frame for the function at the given offset, declare variables corresponding
        // to globals and params, return a View to proceed with the rest of code generation.
        let n_params = sig.params.len();
        let param_tys: SmallVec<[(Ref, Type); 5]> = refs
            .iter()
            .cloned()
            .zip(sig.params.iter().map(|p| p.value_type))
            .collect();
        self.cctx.func.signature = sig;
        let mut builder = FunctionBuilder::new(&mut self.cctx.func, &mut self.ctx);
        let entry_block = builder.create_block();
        let header_block = builder.create_block();
        let mut res = View {
            f: Frame {
                n_params,
                entry_block,
                header_block,
                // this will get overwritten in process_args
                runtime: Variable::new(0),
                n_vars: 0,
                vars: Default::default(),
                iters: Default::default(),
                header_actions: Default::default(),
            },
            builder,
            shared: &mut self.shared,
        };
        res.process_args(param_tys.into_iter(), n_args);
        res
    }

    /// Declare non-main functions and generate corresponding [`Prelude`]s, but do not generate
    /// their bodies.
    fn declare_local_funcs(&mut self, typer: &mut Typer) -> Result<()> {
        let globals = typer.get_global_refs();
        let cc = isa::CallConv::Fast;
        let ptr_ty = self.shared.module.target_config().pointer_type();
        for (i, (info, refs)) in typer.func_info.iter().zip(globals.iter()).enumerate() {
            if !typer.frames[i].is_called {
                self.funcs.push(None);
                self.shared.func_ids.push(None);
                continue;
            }
            let mut sig = Signature::new(cc);
            let total_args = info.arg_tys.len() + refs.len() + 1 /* runtime */;
            sig.params.reserve(total_args);
            // Used in FuncInfo to let callers know which values to pass.
            let mut globals = SmallVec::with_capacity(refs.len());
            let mut arg_refs = SmallVec::with_capacity(total_args);

            // Used in Prelude to provide enough information to initialize variables corresponding
            // to function parameters
            let n_args = info.arg_tys.len();

            let name = format!("udf{}", i);

            // Build up a signature; there are three parts to a user-defined function parameter
            // list (in order):
            // 1. Function-level parameters,
            // 2. Pointers to global variables required by the function,
            // 3. A pointer to the runtime variable
            //
            // All functions are typed to return a single value.
            for r in typer.frames[i]
                .arg_regs
                .iter()
                .cloned()
                .zip(info.arg_tys.iter().cloned())
            {
                let param = ty_to_param(r.1, ptr_ty)?;
                sig.params.push(param);
                arg_refs.push(r);
            }
            for r in refs.iter().cloned() {
                globals.push(r);
                // All globals are passed as pointers
                sig.params.push(AbiParam::new(ptr_ty));
                arg_refs.push(r);
            }
            // Put a placeholder in for the last argument
            arg_refs.push(PLACEHOLDER);

            sig.params.push(AbiParam::new(ptr_ty)); // runtime
            sig.returns
                .push(AbiParam::new(ty_to_clifty(info.ret_ty, ptr_ty)?));

            // Now, to create a function and prelude
            let func_id = self
                .shared
                .module
                .declare_function(name.as_str(), Linkage::Local, &sig)
                .map_err(|e| CompileError(format!("cranelift module error: {}", e)))?;

            self.funcs.push(Some(Prelude {
                sig,
                n_args,
                refs: arg_refs,
            }));
            self.shared
                .func_ids
                .push(Some(FuncInfo { globals, func_id }));
        }
        Ok(())
    }
}

macro_rules! external {
    ($name:ident) => {
        crate::codegen::intrinsics::$name as *const u8
    };
}

impl<'a> View<'a> {
    fn stack_slot_bytes(&mut self, bytes: u32) -> StackSlot {
        debug_assert!(bytes > 0); // This signals a bug; all frawk types have positive size.
        let data = StackSlotData::new(StackSlotKind::ExplicitSlot, bytes);
        self.builder.create_stack_slot(data)
    }

    fn gen_function_body(&mut self, insts: &compile::Frame) -> Result<()> {
        let nodes = insts.cfg.raw_nodes();
        let bbs: Vec<_> = (0..nodes.len())
            .map(|_| self.builder.create_block())
            .collect();
        let mut to_visit: Vec<_> = (0..nodes.len()).collect();
        let mut to_visit_next = Vec::with_capacity(1);

        for round in 0..2 {
            for i in to_visit.drain(..) {
                let node = &nodes[i];
                if node.weight.exit && round == 0 {
                    // We defer processing exit nodes to the end.
                    to_visit_next.push(i);
                    continue;
                }
                self.builder.switch_to_block(bbs[i]);
                for inst in &node.weight.insts {
                    match inst {
                        Either::Left(ll) => self.gen_ll_inst(ll)?,
                        Either::Right(hl) => self.gen_hl_inst(hl)?,
                    }
                }
                // branch-related metadata
                let mut tcase = None;
                let mut ecase = None;

                let mut walker = insts.cfg.neighbors(NodeIx::new(i)).detach();
                while let Some(e) = walker.next_edge(&insts.cfg) {
                    let (_, next) = insts.cfg.edge_endpoints(e).unwrap();
                    let bb = bbs[next.index()];
                    if let Some(e) = *insts.cfg.edge_weight(e).unwrap() {
                        tcase = Some(((e, compile::Ty::Int), bb));
                    } else {
                        ecase = Some(bb);
                    }

                    // Now scan any phi nodes in the successor block for references back to the current
                    // one. If we find one, we issue an assignment, though we skip the drop as it will
                    // be covered by predecessor blocks (alternatively, we could issue a "mov" here,
                    // but this does less work).
                    //
                    // NB We could avoid scanning duplicate Phis here by tracking dependencies a bit
                    // more carefully, but in practice the extra work done seems fairly low given the
                    // CFGs that we generate at time of writing.
                    for inst in &nodes[next.index()].weight.insts {
                        if let Either::Right(compile::HighLevel::Phi(dst_reg, ty, preds)) = inst {
                            for src_reg in preds.iter().filter_map(|(bb, reg)| {
                                if bb.index() == i {
                                    Some(*reg)
                                } else {
                                    None
                                }
                            }) {
                                self.mov_inner(*ty, *dst_reg, src_reg, /*skip_drop=*/ false)?;
                            }
                        } else {
                            // We can bail out once we see the first non-phi instruction. Those all go
                            // at the top
                            break;
                        }
                    }
                }

                if let Some(ecase) = ecase {
                    self.branch(tcase, ecase)?;
                }
            }
            mem::swap(&mut to_visit, &mut to_visit_next);
        }

        // Finally, fill in our "header block" containing variable initializations and jump to
        // bbs[0].
        self.builder.switch_to_block(self.f.header_block);
        self.builder.seal_block(self.f.header_block);
        self.execute_actions()?;
        self.builder.ins().jump(bbs[0], &[]);
        self.builder.seal_all_blocks();
        self.builder.finalize();
        if DUMP_IR {
            eprintln!("{}", self.builder.func);
        }
        Ok(())
    }

    /// If `tcase` is set, jump to the given block if the given value is non-zero. Regardless, jump
    /// unconditionally to `ecase`.
    fn branch(&mut self, tcase: Option<(Ref, Block)>, ecase: Block) -> Result<()> {
        if let Some((cond, b)) = tcase {
            let cv = self.get_val(cond)?;
            self.builder.ins().brnz(cv, b, &[]);
        }
        self.builder.ins().jump(ecase, &[]);
        Ok(())
    }

    /// Assign each incoming parameter to a Variable and an appropriate binding in the `vars` map.
    ///
    /// n_args signifies the length of the prefix of `param_tys` corresponding to
    /// frawk-function-level parameters, where the remaining arguments contain global variables and
    /// a pointer to the runtime.
    fn process_args(&mut self, param_tys: impl Iterator<Item = (Ref, Type)>, n_args: usize) {
        self.builder
            .append_block_params_for_function_params(self.f.entry_block);
        self.builder.switch_to_block(self.f.entry_block);
        self.builder.seal_block(self.f.entry_block);

        // need to copy params because we borrow builder mutably in the loop body.
        let params: SmallVec<[Value; 5]> = self
            .builder
            .block_params(self.f.entry_block)
            .iter()
            .cloned()
            .collect();
        for (i, (val, (rf, ty))) in params.into_iter().zip(param_tys).enumerate() {
            let var = Variable::new(self.f.n_vars);
            self.f.n_vars += 1;
            self.builder.declare_var(var, ty);
            self.builder.def_var(var, val);
            if i == self.f.n_params - 1 {
                // runtime
                self.f.runtime = var;
            } else if i >= n_args {
                // global
                self.f.vars.insert(
                    rf,
                    VarRef {
                        var,
                        kind: VarKind::Global,
                    },
                );
            } else {
                // normal arg. These behave like normal variables, except we do not drop them (they
                // are, in effect, borrowed).
                self.f.vars.insert(
                    rf,
                    VarRef {
                        var,
                        kind: VarKind::Param,
                    },
                );
            }
        }
        self.builder.ins().jump(self.f.header_block, &[]);
    }

    /// Issue end-of-function drop instructions to all local variables that have (a) a non-trivial
    /// drop procedure and (b) have not been marked `skip_drop`.
    fn drop_all(&mut self) {
        let mut drops = Vec::new();
        for ((_, ty), VarRef { var, kind }) in self.f.vars.iter() {
            if let VarKind::Local { skip_drop: false } = kind {
                use compile::Ty::*;
                let drop_fn = match ty {
                    MapIntInt => external!(drop_intint),
                    MapIntFloat => external!(drop_intfloat),
                    MapIntStr => external!(drop_intstr),
                    MapStrInt => external!(drop_strint),
                    MapStrFloat => external!(drop_strfloat),
                    MapStrStr => external!(drop_strstr),
                    Str => external!(drop_str),
                    _ => continue,
                };
                let val = self.builder.use_var(*var);
                drops.push((drop_fn, val));
            }
        }
        for (drop_fn, val) in drops {
            // NB: We could probably refactor call_external_void to only borrow non-f.vars fields
            // and then avoid the auxiliary vector, but life is short, and we will only call this
            // function once per live UDF.
            self.call_external_void(drop_fn, &[val]);
        }
    }

    /// Call a frawk-level (as opposed to builtin/external) function.
    fn call_udf(&mut self, id: NumTy, args: &[Ref]) -> Result<Value> {
        let mut to_pass = SmallVec::<[Value; 6]>::with_capacity(args.len() + 1);
        for arg in args.iter().cloned() {
            let v = self.get_val(arg)?;
            to_pass.push(v);
        }
        let FuncInfo { globals, func_id } = self.shared.func_ids[id as usize]
            .as_ref()
            .expect("all referenced functions must be declared");
        for global in globals {
            // We don't use get_val here because we want to pass the pointer to the global, and
            // get_val will issue a load.
            match self.f.vars.get(global) {
                Some(VarRef {
                    var,
                    kind: VarKind::Global,
                    ..
                }) => {
                    to_pass.push(self.builder.use_var(*var));
                }
                _ => return err!("internal error, functions disagree on if reference is global"),
            }
        }
        to_pass.push(self.builder.use_var(self.f.runtime));
        let fref = self
            .shared
            .module
            .declare_func_in_func(*func_id, self.builder.func);
        let call_inst = self.builder.ins().call(fref, &to_pass[..]);
        Ok(self
            .builder
            .inst_results(call_inst)
            .iter()
            .cloned()
            .next()
            .expect("all UDFs must return a value"))
    }

    /// Translate a high-level instruction. If the instruction is a `Ret`, we return the returned
    /// value for further processing. We want to process returns last because we can only be sure
    /// that the `drop_all` method will catch all relevant local variables once we have processed
    /// all of the other instructions.
    fn gen_hl_inst(&mut self, inst: &compile::HighLevel) -> Result<()> {
        use compile::HighLevel::*;
        match inst {
            Call {
                func_id,
                dst_reg,
                dst_ty,
                args,
            } => {
                let res = self.call_udf(*func_id, args.as_slice())?;
                self.bind_val((*dst_reg, *dst_ty), res)?;
                Ok(())
            }
            Ret(reg, ty) => {
                let mut v = self.get_val((*reg, *ty))?;
                let test = self.f.vars.get_mut(&(*reg, *ty)).map(|x| {
                    // if this is a local, we want to avoid dropping it.
                    x.kind.skip_drop();
                    &x.kind
                });
                if let Some(VarKind::Param) | Some(VarKind::Global) = test {
                    // This was passed in as a parameter, so us returning it will introduce a new
                    // reference.
                    self.ref_val(*ty, v);
                }
                if let compile::Ty::Str = ty {
                    let str_ty = self.get_ty(*ty);
                    v = self.builder.ins().load(str_ty, MemFlags::trusted(), v, 0);
                }
                self.drop_all();
                self.builder.ins().return_(&[v]);
                Ok(())
            }
            DropIter(reg, ty) => {
                use compile::Ty::*;
                let drop_fn = match ty {
                    IterInt => external!(drop_iter_int),
                    IterStr => external!(drop_iter_str),
                    _ => return err!("can only drop iterators, got {:?}", ty),
                };
                let IterState { base, bytes, .. } = self.get_iter((*reg, *ty))?;
                let base = self.builder.use_var(base);
                let bytes = self.builder.use_var(bytes);
                let key_ty = self.get_ty(ty.iter()?);
                let len = self.div_by_type_size(key_ty, bytes)?;
                self.call_external_void(drop_fn, &[base, len]);
                Ok(())
            }
            // Phis are handled in predecessor blocks
            Phi(..) => Ok(()),
        }
    }

    fn default_value(&mut self, ty: compile::Ty) -> Result<Value> {
        use compile::Ty::*;
        match ty {
            Null | Int => Ok(self.const_int(0)),
            Float => Ok(self.builder.ins().f64const(0.0)),
            Str => {
                // cranelift does not currently support iconst for I128
                let zero64 = self.builder.ins().iconst(types::I64, 0);
                Ok(self.builder.ins().iconcat(zero64, zero64))
            }
            MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat | MapStrStr => {
                let alloc_fn = match ty {
                    MapIntInt => external!(alloc_intint),
                    MapIntFloat => external!(alloc_intfloat),
                    MapIntStr => external!(alloc_intstr),
                    MapStrInt => external!(alloc_strint),
                    MapStrFloat => external!(alloc_strfloat),
                    MapStrStr => external!(alloc_strstr),
                    _ => unreachable!(),
                };
                Ok(self.call_external(alloc_fn, &[]))
            }
            IterInt | IterStr => err!("iterators do not have default values"),
        }
    }

    fn store_string(&mut self, ss: StackSlot, v: Value) {
        let str_ty = self.get_ty(compile::Ty::Str);
        let ptr_ty = self.ptr_to(str_ty);
        // We get an error if we do a direct stack_store here
        let addr = self.builder.ins().stack_addr(ptr_ty, ss, 0);
        self.builder.ins().store(MemFlags::trusted(), v, addr, 0);
    }

    fn execute_actions(&mut self) -> Result<()> {
        let header_actions = mem::take(&mut self.f.header_actions);
        for EntryDeclaration { var, ty } in header_actions {
            use compile::Ty::*;
            let cl_ty = self.get_ty(ty);
            let default_v = self.default_value(ty)?;
            match ty {
                Null | Int | Float => {
                    self.builder.def_var(var, default_v);
                }
                Str => {
                    // allocate a stack slot for the string, then assign var to point to that
                    // slot.
                    let ptr_ty = self.ptr_to(cl_ty);
                    let slot = self.stack_slot_bytes(mem::size_of::<runtime::Str>() as u32);
                    let addr = self.builder.ins().stack_addr(ptr_ty, slot, 0);
                    self.builder.def_var(var, addr);
                    self.store_string(slot, default_v);
                }
                MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat | MapStrStr => {
                    self.builder.def_var(var, default_v);
                }
                IterInt | IterStr => return err!("attempting to default-initialize iterator type"),
            }
        }
        Ok(())
    }

    /// Initialize the `Variable` associated with a non-iterator local variable of type `ty`.
    fn declare_local(&mut self, ty: compile::Ty) -> Result<Variable> {
        use compile::Ty::*;
        let next_var = Variable::new(self.f.n_vars);
        self.f.n_vars += 1;
        let cl_ty = self.get_ty(ty);
        // Remember to allocate/initialize this variable in the header
        self.f
            .header_actions
            .push(EntryDeclaration { ty, var: next_var });
        match ty {
            Null | Int | Float => {
                self.builder.declare_var(next_var, cl_ty);
            }
            Str => {
                let ptr_ty = self.ptr_to(cl_ty);
                self.builder.declare_var(next_var, ptr_ty);
            }
            MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat | MapStrStr => {
                self.builder.declare_var(next_var, cl_ty);
            }
            IterInt | IterStr => return err!("iterators cannot be declared"),
        }
        Ok(next_var)
    }

    /// Construct an [`IterState`] corresponding to an iterator of type `ty`.
    fn declare_iterator(&mut self, ty: compile::Ty) -> Result<IterState> {
        use compile::Ty::*;
        match ty {
            IterStr | IterInt => {
                let bytes = Variable::new(self.f.n_vars);
                let cur = Variable::new(self.f.n_vars + 1);
                let base = Variable::new(self.f.n_vars + 2);
                self.f.n_vars += 3;
                self.builder.declare_var(bytes, types::I64);
                self.builder.declare_var(cur, types::I64);
                self.builder.declare_var(base, self.void_ptr_ty());
                Ok(IterState { bytes, cur, base })
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
                Eq => self.builder.ins().fcmp(FloatCC::Equal, l, r),
                Lte => self.builder.ins().fcmp(FloatCC::LessThanOrEqual, l, r),
                Lt => self.builder.ins().fcmp(FloatCC::LessThan, l, r),
                Gte => self.builder.ins().fcmp(FloatCC::GreaterThanOrEqual, l, r),
                Gt => self.builder.ins().fcmp(FloatCC::GreaterThan, l, r),
            }
        } else {
            match op {
                Eq => self.builder.ins().icmp(IntCC::Equal, l, r),
                Lte => self.builder.ins().icmp(IntCC::SignedLessThanOrEqual, l, r),
                Lt => self.builder.ins().icmp(IntCC::SignedLessThan, l, r),
                Gte => self
                    .builder
                    .ins()
                    .icmp(IntCC::SignedGreaterThanOrEqual, l, r),
                Gt => self.builder.ins().icmp(IntCC::SignedGreaterThan, l, r),
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
            ArithmeticRightShift => self.builder.ins().sshr(args[0], args[1]),
            LeftShift => self.builder.ins().ishl(args[0], args[1]),
            Xor => self.builder.ins().bxor(args[0], args[1]),
        }
    }

    /// Apply the [`FloatFunc`] operation specified in `op` to `args`.
    ///
    /// Panics if args has the wrong arity. Unlike LLVM, most of these functions do not have direct
    /// instructions (or intrinsics), so they are implemented as function calls to rust functions
    /// which in turn call into the standard library.
    ///
    /// [`FloatFunc`]: [crate::builtins::FloatFunc]
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
        args: &[Ref],
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
    fn get_var_default_local(&mut self, r: Ref, skip_drop: bool) -> Result<VarRef> {
        if let Some(v) = self.f.vars.get(&r) {
            Ok(v.clone())
        } else {
            let var = self.declare_local(r.1)?;
            let vref = VarRef {
                var,
                kind: VarKind::Local { skip_drop },
            };
            self.f.vars.insert(r, vref.clone());
            Ok(vref)
        }
    }

    fn bind_val_inner(&mut self, r: Ref, v: Value, skip_drop: bool) -> Result<()> {
        use compile::Ty::*;
        let VarRef { var, kind } = self.get_var_default_local(r, skip_drop)?;
        match r.1 {
            Int | Float => {
                if let VarKind::Global = kind {
                    let p = self.builder.use_var(var);
                    self.builder.ins().store(MemFlags::trusted(), v, p, 0);
                } else {
                    self.builder.def_var(var, v);
                }
            }
            Str => {
                // NB: we assume that `v` is a string, not a pointer to a string.

                // first, drop the value currently in the pointer
                let p = self.builder.use_var(var);
                self.drop_val(Str, p);
                self.builder.ins().store(MemFlags::trusted(), v, p, 0);
            }
            MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat | MapStrStr => {
                if let VarKind::Global = kind {
                    // Drop the value currently in the pointer
                    let p = self.builder.use_var(var);
                    let pointee = self
                        .builder
                        .ins()
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

    fn mov_inner(
        &mut self,
        ty: compile::Ty,
        dst: NumTy,
        src: NumTy,
        skip_drop: bool,
    ) -> Result<()> {
        use compile::Ty::*;
        let src = self.get_val((src, ty))?;
        match ty {
            Int | Float => self.bind_val_inner((dst, ty), src, skip_drop)?,
            Str => {
                self.call_external_void(external!(ref_str), &[src]);
                let str_ty = self.get_ty(Str);
                let loaded = self.builder.ins().load(str_ty, MemFlags::trusted(), src, 0);
                self.bind_val_inner((dst, Str), loaded, skip_drop)?;
            }
            MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat | MapStrStr => {
                self.call_external_void(external!(ref_map), &[src]);
                self.bind_val_inner((dst, ty), src, skip_drop)?;
            }
            IterInt | IterStr => return err!("attempting to apply `mov` to an iterator!"),
            Null => {
                let zero = self.const_int(0);
                self.bind_val_inner((dst, ty), zero, skip_drop)?;
            }
        }
        Ok(())
    }

    /// For a type whose size is a power of two, divide the multiply the integer Value v by that
    /// size
    fn mul_by_type_size(&mut self, ty: Type, v: Value) -> Result<Value> {
        let ty_bytes = ty.lane_bits() / 8;
        if !ty_bytes.is_power_of_two() {
            return err!("unsupported type size");
        }
        let shift = self.const_int(ty_bytes.trailing_zeros() as i64);
        Ok(self.builder.ins().ishl(v, shift))
    }

    /// For a type whose size is a power of two, divide the divide the integer Value v by that
    /// size
    fn div_by_type_size(&mut self, ty: Type, v: Value) -> Result<Value> {
        let ty_bytes = ty.lane_bits() / 8;
        if !ty_bytes.is_power_of_two() {
            return err!("unsupported type size");
        }
        let shift = self.const_int(ty_bytes.trailing_zeros() as i64);
        Ok(self.builder.ins().ushr(v, shift))
    }
}

// For Cranelift, we need to register function names in a lookup table before constructing a
// module, so we actually implement `Backend` twice for each registration step.

struct RegistrationState {
    builder: JITBuilder,
}

impl Backend for RegistrationState {
    type Ty = ();
    fn void_ptr_ty(&self) {}
    fn ptr_to(&self, (): ()) {}
    fn usize_ty(&self) {}
    fn u32_ty(&self) {}
    fn get_ty(&self, _ty: compile::Ty) {}

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
        let ptr_ty = self.void_ptr_ty();
        ty_to_clifty(ty, ptr_ty).expect("invalid type argument")
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
            .map_err(|e| CompileError(format!("error declaring {} in module: {}", name, e,)))?;
        self.shared.external_funcs.insert(addr, id);
        Ok(())
    }
}

impl<'a> CodeGenerator for View<'a> {
    type Val = Value;

    // mappings to and from bytecode-level registers to IR-level values
    fn bind_val(&mut self, r: Ref, v: Self::Val) -> Result<()> {
        self.bind_val_inner(r, v, /*skip_drop=*/ false)
    }

    fn get_val(&mut self, r: Ref) -> Result<Self::Val> {
        use compile::Ty::*;
        if let Null = r.1 {
            return Ok(self.const_int(0));
        }

        let VarRef { var, kind, .. } = self.get_var_default_local(r, /*skip_drop=*/ false)?;
        let is_global = matches!(kind, VarKind::Global);
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
    fn const_ptr<T>(&mut self, c: *const T) -> Self::Val {
        self.const_int(c as *const _ as i64)
    }
    fn handles(&mut self) -> &mut Handles {
        &mut self.shared.handles
    }

    fn call_void(&mut self, func: *const u8, args: &mut [Self::Val]) -> Result<()> {
        self.call_external_void(func, args);
        Ok(())
    }

    // TODO We may eventually want to remove the `Result` return value here
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
        args: &[Ref],
    ) -> Result<()> {
        // For empty args, just delegate to print_all
        if args.is_empty() {
            return self.print_all(output, &[*fmt]);
        }
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

    fn sprintf(&mut self, dst: &StrReg, fmt: &StrReg, args: &[Ref]) -> Result<()> {
        // For empty args, just move fmt into dst.
        if args.is_empty() {
            return self.mov(compile::Ty::Str, dst.reflect().0, fmt.reflect().0);
        }

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

    fn print_all(&mut self, output: &Option<(StrReg, FileSpec)>, args: &[StrReg]) -> Result<()> {
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
        self.mov_inner(ty, dst, src, /*skip_drop=*/ false)
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
        let key_ty = self.get_ty(dst.1.iter()?);
        let map = self.get_val(map)?;
        let IterState { bytes, cur, base } = self.get_iter(dst)?;
        let ptr = self.call_external(begin_fn, &[map]);
        let map_len = self.call_external(len_fn, &[map]);
        let total_bytes = self.mul_by_type_size(key_ty, map_len)?;
        let zero = self.const_int(0);
        self.builder.def_var(cur, zero);
        self.builder.def_var(bytes, total_bytes);
        self.builder.def_var(base, ptr);
        Ok(())
    }

    fn iter_hasnext(&mut self, dst: Ref, iter: Ref) -> Result<()> {
        let IterState { bytes, cur, .. } = self.get_iter(iter)?;
        let lenv = self.builder.use_var(bytes);
        let curv = self.builder.use_var(cur);
        let cmp = self.builder.ins().icmp(IntCC::UnsignedLessThan, curv, lenv);
        let cmp_int = self.bool_to_int(cmp);
        self.bind_val(dst, cmp_int)
    }

    fn iter_getnext(&mut self, dst: Ref, iter: Ref) -> Result<()> {
        // Compute base+cur and load it into a value
        let IterState { cur, base, .. } = self.get_iter(iter)?;
        let base = self.builder.use_var(base);
        let cur_val = self.builder.use_var(cur);
        let ptr = self.builder.ins().iadd(base, cur_val);
        let ty = self.get_ty(dst.1);
        let contents = self.builder.ins().load(ty, MemFlags::trusted(), ptr, 0);

        // Increment cur
        let type_size = self.get_ty(dst.1).lane_bits() / 8;
        let inc = self.const_int(type_size as i64);
        let inc_cur = self.builder.ins().iadd(cur_val, inc);
        self.builder.def_var(cur, inc_cur);

        // bind the result to `dst` and increment the refcount, if relevant.
        self.bind_val(dst, contents)?;
        let dst_ptr = self.get_val(dst)?;
        self.ref_val(dst.1, dst_ptr);
        Ok(())
    }
}
