use hashbrown::{hash_map::Entry, HashMap};
use smallvec::smallvec;

use crate::builtins::{self, Variable};
use crate::bytecode::{self, Instr, Interp};
use crate::cfg::{self, is_unused, Ident, PrimExpr, PrimStmt, PrimVal};
use crate::cfg2::{Function, ProgramContext};
use crate::common::{NodeIx, NumTy, Result};
use crate::types::{self, SmallVec};

// TODO: implement basic "optimizations"
//    * avoid excessive moves (unless those come from the cfg?)
//    * don't emit jumps that just point to the next instruction.

const UNUSED: u32 = u32::max_value();

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub(crate) enum Ty {
    Int = 0,
    Float = 1,
    Str = 2,
    MapIntInt = 3,
    MapIntFloat = 4,
    MapIntStr = 5,
    MapStrInt = 6,
    MapStrFloat = 7,
    MapStrStr = 8,
    IterInt = 9,
    IterStr = 10,
}

impl Default for Ty {
    fn default() -> Ty { Ty::Str }
}

impl Ty {
    fn of_var(v: Variable) -> Ty {
        use Variable::*;
        match v {
            ARGC | FS | OFS | RS | FILENAME => Ty::Str,
            NF | NR => Ty::Int,
            ARGV => Ty::MapIntStr,
        }
    }

    fn iter(self) -> Result<Ty> {
        use Ty::*;
        match self {
            IterInt => Ok(Int),
            IterStr => Ok(Str),
            Int | Float | Str | MapIntInt | MapIntFloat | MapIntStr | MapStrInt | MapStrFloat
            | MapStrStr => err!("attempt to get element of non-iterator type: {:?}", self),
        }
    }

    fn key(self) -> Result<Ty> {
        use Ty::*;
        match self {
            MapIntInt | MapIntFloat | MapIntStr => Ok(Int),
            MapStrInt | MapStrFloat | MapStrStr => Ok(Str),
            Int | Float | Str | IterInt | IterStr => {
                err!("attempt to get key of non-map type: {:?}", self)
            }
        }
    }
    fn val(self) -> Result<Ty> {
        use Ty::*;
        match self {
            MapStrInt | MapIntInt => Ok(Int),
            MapStrFloat | MapIntFloat => Ok(Float),
            MapStrStr | MapIntStr => Ok(Str),
            Int | Float | Str | IterInt | IterStr => {
                err!("attempt to get val of non-map type: {:?}", self)
            }
        }
    }
}

const NUM_TYPES: usize = Ty::IterStr as usize + 1;
// TODO:
//  * stub out new `full_bytecode` function taking a cfg2::ProgramContext
//  * Build a new ProgramGenerator that consists of Registers and Map<(NumTy /*old function Id */,
//  SmallVec<Ty>), NumTy /*new function Id */>, and Vec<Frame> (or Frame,Instrs, depending on how
//  it all works out)
//      * Get main to go first
//  * Build a View with mut refs to Registers and Frame
//  * Move all code to new ProgramGenerator (c/p is fine;we'll remove it all, but it could be
//  doable to just port everything over.)
//  Get all code compiling and tests passing before adding in full function support.
//
//  * On compile side, we'll need a runtime stack for continuations, a Vec<Vec<Instr>> for all
//  functions, and some return mechanism.

// This is a macro to defeat the borrow checker when used inside methods for `Generator`.
macro_rules! reg_of_ty {
    ($slf:expr, $ty:expr) => {{
        let cnt = &mut $slf.reg_counts[$ty as usize];
        let res = *cnt;
        *cnt += 1;
        res
    }};
}

#[derive(Default)]
struct ProgramGenerator<'a> {
    regs: Registers,

    // TODO: can this be a Vec<NumTy>?
    arity_map: HashMap<NumTy /* cfg-level func id */, NumTy /* arity */>,
    id_map: HashMap<
        // TODO: make newtypes for these different Ids?
        (
            NumTy,        /* cfg-level func id */
            SmallVec<Ty>, /* arg types */
        ),
        NumTy, /* bytecode-level func id */
    >,
    // TODO store return type and register in a separate vector, store offset into the vector in
    // the frame, pass a reference to the vector in View.
    func_rets:
        HashMap<(NumTy /* func id */, SmallVec<Ty> /* arg types */), Ty /* return type */>,
    frames: Vec<Frame<'a>>,
}

impl<'a> ProgramGenerator<'a> {
    fn init_from_ctx(pc: &ProgramContext<'a, &'a str>) -> Result<ProgramGenerator<'a>> {
        // Type-check the code, then initialize a ProgramGenerator, assigning registers to local
        // and global variables.
        let mut n_funcs = 0;
        let mut gen = ProgramGenerator::default();
        let types::TypeInfo { var_tys, func_tys } = types::get_types_program(pc)?;
        gen.func_rets = func_tys;
        for ((id, func_id, args), ty) in var_tys.iter() {
            let map = if id.global {
                &mut gen.regs.globals
            } else {
                let mapped_func = match gen.id_map.entry((*func_id, args.clone())) {
                    Entry::Occupied(o) => &mut gen.frames[*o.get() as usize],
                    Entry::Vacant(v) => {
                        let res = n_funcs;
                        n_funcs += 1;
                        v.insert(res);
                        let ret_ty = gen.func_rets[&(*func_id, args.clone())];
                        let ret_reg = reg_of_ty!(gen.regs, ret_ty);
                        let mut f = Frame::default();
                        f.ret_reg=ret_reg;
                        f.ret_ty=ret_ty;
                        f.src_function = *func_id;
                        let arity = pc.funcs[*func_id as usize].args.len() as u32;
                        gen.arity_map.insert(*func_id, arity);
                        gen.frames.push(f);
                        &mut gen.frames[res as usize]
                    }
                };
                &mut mapped_func.locals
            };
            let reg = reg_of_ty!(gen.regs, *ty);
            if let Some(old) = map.insert(*id, (reg, *ty)) {
                return err!(
                    "internal error: duplicate entries for same local in types {:?} vs {:?}",
                    old,
                    (reg, *ty)
                );
            }
        }
        for frame in gen.frames.iter_mut() {
            let src_func = frame.src_function as usize;
            View {
                frame,
                regs: &mut gen.regs,
                id_map: &gen.id_map,
                arity_map: &gen.arity_map,
            }
            .process_function(&pc.funcs[src_func])?;
        }
        Ok(gen)
    }
}

#[derive(Default)]
struct Registers {
    reg_counts: [u32; NUM_TYPES],
    globals: HashMap<Ident, (u32, Ty)>,
}

#[derive(Default)]
struct Frame<'a> {
    src_function: NumTy,
    locals: HashMap<Ident, (u32, Ty)>,
    jmps: Vec<usize>,
    bb_to_instr: Vec<usize>,
    instrs: Vec<Instr<'a>>,
    // TODO add these fields, but first we need to propagate return types in the `types` module
    ret_reg: u32,
    ret_ty: Ty,
}

struct View<'a, 'b> {
    frame: &'b mut Frame<'a>,
    regs: &'b mut Registers,
    id_map: &'b HashMap<(NumTy, SmallVec<Ty>), NumTy>,
    arity_map: &'b HashMap<NumTy, NumTy>,
}

struct Generator {
    // local*
    registers: HashMap<Ident, (u32, Ty)>,
    // global
    reg_counts: [u32; NUM_TYPES],
    // local
    jmps: Vec<usize>,
    // local
    bb_to_instr: Vec<usize>,
    // local*; but do we need it at all?
    ts: HashMap<Ident, Ty>,
    // *Could be done as (Ident, SmallVec<Ty>)
}

impl<'a, 'b> View<'a, 'b> {
    fn process_function(&mut self, func: &Function<'a, &'a str>) -> Result<()> {
        // No logic in this method relies on this directly, but it is an
        // invariant of the broader module.
        assert_eq!(self.frame.src_function, func.ident);

        // Pop any arguments off the stack.
        for arg in func.args.iter() {
            use Ty::*;
            let (reg, ty) = self.reg_of_ident(&arg.id);
            self.push(match ty {
                Int => Instr::PopInt(reg.into()),
                Float => Instr::PopFloat(reg.into()),
                Str => Instr::PopStr(reg.into()),
                MapIntInt => Instr::PopIntInt(reg.into()),
                MapIntFloat => Instr::PopIntFloat(reg.into()),
                MapIntStr => Instr::PopIntStr(reg.into()),
                MapStrInt => Instr::PopStrInt(reg.into()),
                MapStrFloat => Instr::PopStrFloat(reg.into()),
                MapStrStr => Instr::PopStrStr(reg.into()),
                IterInt | IterStr => {
                    return err!(
                        "unsupported argument type for function {}: {:?}",
                        if let Some(name) = func.name {
                            name
                        } else {
                            "<main>"
                        },
                        ty
                    )
                }
            });
        }
        self.frame.bb_to_instr = vec![0; func.cfg.node_count()];

        for (i, n) in func.cfg.raw_nodes().iter().enumerate() {
            self.frame.bb_to_instr[i] = self.frame.instrs.len();
            for stmt in n.weight.q.iter() {
                self.stmt(stmt)?;
            }

            let ix = NodeIx::new(i);
            let mut branches: cfg::SmallVec<petgraph::graph::EdgeIndex> = Default::default();
            let mut walker = func.cfg.neighbors(ix).detach();
            while let Some(e) = walker.next_edge(&func.cfg) {
                branches.push(e)
            }

            // Petgraph gives us edges back in reverse order.
            branches.reverse();

            // Replace Phi functions in successors with assignments at this point in the stream.
            //
            // NB Why is it sufficient to do all assignments here? Shouldn't we limit code gen to  the
            // branch we are actually going to take? No, because one must first go through another
            // block that assigns to the node again before actually reading the variable.
            for n in func.cfg.neighbors(NodeIx::new(i)) {
                let weight = func.cfg.node_weight(n).unwrap();
                for stmt in weight.q.iter() {
                    if let PrimStmt::AsgnVar(id, PrimExpr::Phi(preds)) = stmt {
                        let mut found = false;
                        for (pred, src) in preds.iter() {
                            if pred.index() == i {
                                found = true;
                                // now do the assignment
                                let (dst_reg, dst_ty) = self.reg_of_ident(id);
                                let (src_reg, src_ty) = self.reg_of_ident(src);
                                self.convert(dst_reg, dst_ty, src_reg, src_ty)?;
                            }
                        }
                        if !found {
                            return err!("malformed phi node: preds={:?} cur={:?}", &preds[..], i);
                        }
                    } else {
                        break;
                    }
                }
            }

            // Insert code for the branches at the end of this basic block. At first, labels just
            // indicate the basic block in the CFG. They are re-mapped at the end of execution.
            // NB Don't insert a halt if this block ends with a return.
            let mut is_end = i != func.exit.index();
            for b in branches.iter().cloned() {
                let next = func.cfg.edge_endpoints(b).unwrap().1;
                match &func.cfg.edge_weight(b).unwrap().0 {
                    Some(v) => {
                        let (mut reg, ty) = self.get_reg(v)?;
                        match ty {
                            Ty::Int => {}
                            Ty::Float => {
                                let dst = reg_of_ty!(self.regs, Ty::Int);
                                self.convert(dst, Ty::Int, reg, Ty::Float)?;
                                reg = dst;
                            }
                            Ty::Str => {
                                let dst = reg_of_ty!(self.regs, Ty::Int);
                                self.push(Instr::LenStr(dst.into(), reg.into()));
                                reg = dst;
                            }
                            _ => return err!("invalid type for branch: {:?}: {:?}", v, ty),
                        }

                        let end_pos = self.frame.instrs.len();
                        self.frame.jmps.push(end_pos);
                        self.push(Instr::JmpIf(reg.into(), next.index().into()));
                    }
                    None => {
                        is_end = false;
                        // There's no point issuing an unconditional jump to the next basic block
                        // because it comes next in the instruction stream.
                        if next.index() != i + 1 {
                            self.frame.jmps.push(self.frame.instrs.len());
                            self.push(Instr::Jmp(next.index().into()))
                        }
                    }
                }
            }
            if is_end {
                // TODO This should only happen in `main`, we should just allow main to have return
                // statements and not use Halt here at all.
                // block.
                self.push(Instr::Halt);
            }
        }

        // Rewrite jumps to point to proper offsets.
        for jmp in self.frame.jmps.iter().cloned() {
            match &mut self.frame.instrs[jmp] {
                Instr::Jmp(bytecode::Label(l)) => *l = self.frame.bb_to_instr[*l as usize] as u32,
                Instr::JmpIf(_, bytecode::Label(l)) => {
                    *l = self.frame.bb_to_instr[*l as usize] as u32
                }
                _ => unreachable!(),
            }
        }
        Ok(())
    }

    // Get the register associated with a given identifier and assign a new one if it does not yet
    // have one.
    fn reg_of_ident(&mut self, id: &Ident) -> (u32, Ty) {
        if is_unused(*id) {
            // We should not actually store into the "unused" identifier.
            // TODO: remove this once there's better test coverage and we use more unsafe code.
            return (UNUSED, Ty::Int);
        }
        if id.global {
            self.regs.globals[id]
        } else {
            self.frame.locals[id]
        }
    }

    // Move src into dst at type Ty.
    fn mov(&mut self, dst_reg: u32, src_reg: u32, ty: Ty) -> Result<()> {
        use Ty::*;
        if dst_reg == UNUSED || src_reg == UNUSED {
            return Ok(());
        }
        let res = match ty {
            Int => Instr::MovInt(dst_reg.into(), src_reg.into()),
            Float => Instr::MovFloat(dst_reg.into(), src_reg.into()),
            Str => Instr::MovStr(dst_reg.into(), src_reg.into()),

            MapIntInt => Instr::MovMapIntInt(dst_reg.into(), src_reg.into()),
            MapIntFloat => Instr::MovMapIntFloat(dst_reg.into(), src_reg.into()),
            MapIntStr => Instr::MovMapIntStr(dst_reg.into(), src_reg.into()),

            MapStrInt => Instr::MovMapStrInt(dst_reg.into(), src_reg.into()),
            MapStrFloat => Instr::MovMapStrFloat(dst_reg.into(), src_reg.into()),
            MapStrStr => Instr::MovMapStrStr(dst_reg.into(), src_reg.into()),

            IterInt | IterStr => return err!("attempt to move values of type {:?}", ty),
        };
        Ok(self.push(res))
    }

    // Get the register associated with a value. For identifiers this has the same semantics as
    // reg_of_ident. For literals, a new register of the appropriate type is allocated.
    fn get_reg(&mut self, v: &PrimVal<'a>) -> Result<(u32, Ty)> {
        match v {
            PrimVal::ILit(i) => {
                let nreg = reg_of_ty!(self.regs, Ty::Int);
                self.push(Instr::StoreConstInt(nreg.into(), *i));
                Ok((nreg, Ty::Int))
            }
            PrimVal::FLit(f) => {
                let nreg = reg_of_ty!(self.regs, Ty::Float);
                self.push(Instr::StoreConstFloat(nreg.into(), *f));
                Ok((nreg, Ty::Float))
            }
            PrimVal::StrLit(s) => {
                let nreg = reg_of_ty!(self.regs, Ty::Str);
                self.push(Instr::StoreConstStr(nreg.into(), (*s).into()));
                Ok((nreg, Ty::Str))
            }
            PrimVal::Var(v) => Ok(self.reg_of_ident(v)),
        }
    }

    // Convert src into dst. If the types match this is just a move. If both the types and the
    // registers match this is a noop.
    fn convert(&mut self, dst_reg: u32, dst_ty: Ty, src_reg: u32, src_ty: Ty) -> Result<()> {
        use Ty::*;
        if dst_reg == UNUSED {
            return Ok(());
        }
        if dst_reg == src_reg && dst_ty == src_ty {
            return Ok(());
        }
        let res = match (dst_ty, src_ty) {
            (Float, Int) => Instr::IntToFloat(dst_reg.into(), src_reg.into()),
            (Str, Int) => Instr::IntToStr(dst_reg.into(), src_reg.into()),

            (Int, Float) => Instr::FloatToInt(dst_reg.into(), src_reg.into()),
            (Str, Float) => Instr::FloatToStr(dst_reg.into(), src_reg.into()),

            (Int, Str) => Instr::StrToInt(dst_reg.into(), src_reg.into()),
            (Float, Str) => Instr::StrToFloat(dst_reg.into(), src_reg.into()),

            (MapIntFloat, MapIntFloat) => Instr::MovMapIntFloat(dst_reg.into(), src_reg.into()),
            (MapIntStr, MapIntStr) => Instr::MovMapIntStr(dst_reg.into(), src_reg.into()),

            (MapStrFloat, MapStrFloat) => Instr::MovMapStrFloat(dst_reg.into(), src_reg.into()),
            (MapStrStr, MapStrStr) => Instr::MovMapStrStr(dst_reg.into(), src_reg.into()),
            (dst, src) => {
                if dst == src {
                    return self.mov(dst_reg, src_reg, dst);
                }
                return err!(
                    "invalid attempt to convert or move dst={:?} src={:?}",
                    dst,
                    src
                );
            }
        };

        Ok(self.push(res))
    }

    // Store values into a register at a given type, converting if necessary.
    fn store(&mut self, dst_reg: u32, dst_ty: Ty, src: &PrimVal<'a>) -> Result<()> {
        match src {
            PrimVal::Var(id2) => {
                let (src_reg, src_ty) = self.reg_of_ident(id2);
                self.convert(dst_reg, dst_ty, src_reg, src_ty)?;
            }
            PrimVal::ILit(i) => {
                if dst_ty == Ty::Int {
                    self.push(Instr::StoreConstInt(dst_reg.into(), *i));
                } else {
                    let ir = reg_of_ty!(self.regs, Ty::Int);
                    self.push(Instr::StoreConstInt(ir.into(), *i));
                    self.convert(dst_reg, dst_ty, ir, Ty::Int)?;
                }
            }
            PrimVal::FLit(f) => {
                if dst_ty == Ty::Float {
                    self.push(Instr::StoreConstFloat(dst_reg.into(), *f));
                } else {
                    let ir = reg_of_ty!(self.regs, Ty::Float);
                    self.push(Instr::StoreConstFloat(ir.into(), *f));
                    self.convert(dst_reg, dst_ty, ir, Ty::Float)?;
                }
            }
            PrimVal::StrLit(s) => {
                if dst_ty == Ty::Str {
                    self.push(Instr::StoreConstStr(dst_reg.into(), (*s).into()));
                } else {
                    let ir = reg_of_ty!(self.regs, Ty::Str);
                    self.push(Instr::StoreConstStr(ir.into(), (*s).into()));
                    self.convert(dst_reg, dst_ty, ir, Ty::Str)?;
                }
            }
        };
        Ok(())
    }

    // Generate bytecode for map lookups.
    fn load_map(
        &mut self,
        dst_reg: u32,
        dst_ty: Ty,
        arr_reg: u32,
        arr_ty: Ty,
        key: &PrimVal<'a>,
    ) -> Result<()> {
        if dst_reg == UNUSED {
            return Ok(());
        }
        // Convert `key` if necessary.
        let target_ty = arr_ty.key()?;
        let (mut key_reg, key_ty) = self.get_reg(key)?;
        if target_ty != key_ty {
            let inter = reg_of_ty!(self.regs, target_ty);
            self.convert(inter, target_ty, key_reg, key_ty)?;
            key_reg = inter;
        }

        // Determine if we will need to convert the result when storing the variable.
        let arr_val_ty = arr_ty.val()?;
        let load_reg = if dst_ty == arr_val_ty {
            dst_reg
        } else {
            reg_of_ty!(self.regs, arr_val_ty)
        };

        // Emit the corresponding instruction.
        use Ty::*;
        self.push(match arr_ty {
            MapIntInt => Instr::LookupIntInt(load_reg.into(), arr_reg.into(), key_reg.into()),
            MapIntFloat => Instr::LookupIntFloat(load_reg.into(), arr_reg.into(), key_reg.into()),
            MapIntStr => Instr::LookupIntStr(load_reg.into(), arr_reg.into(), key_reg.into()),
            MapStrInt => Instr::LookupStrInt(load_reg.into(), arr_reg.into(), key_reg.into()),
            MapStrFloat => Instr::LookupStrFloat(load_reg.into(), arr_reg.into(), key_reg.into()),
            MapStrStr => Instr::LookupStrStr(load_reg.into(), arr_reg.into(), key_reg.into()),
            Int | Float | Str | IterInt | IterStr => {
                return err!("[load_map] expected map type, found {:?}", arr_ty)
            }
        });
        // Convert the result: note that if we had load_reg == dst_reg, then this is a noop.
        self.convert(dst_reg, dst_ty, load_reg, arr_val_ty)
    }
    fn builtin(
        &mut self,
        dst_reg: u32,
        dst_ty: Ty,
        bf: &builtins::Function,
        args: &cfg::SmallVec<PrimVal<'a>>,
    ) -> Result<()> {
        use crate::ast::{Binop::*, Unop::*};
        use builtins::Function::*;

        // Compile the argument values
        let mut args_regs = cfg::SmallVec::with_capacity(args.len());
        let mut args_tys = cfg::SmallVec::with_capacity(args.len());
        for arg in args.iter() {
            let (reg, ty) = self.get_reg(arg)?;
            args_regs.push(reg);
            args_tys.push(ty);
        }

        // Now, perform any necessary conversions if input types do not match the argument types.
        let mut conv_regs: cfg::SmallVec<_> = smallvec![!0u32; args.len()];
        let (conv_tys, res_ty) = bf.type_sig(&args_tys[..])?;

        for (areg, (aty, (creg, cty))) in args_regs.iter().cloned().zip(
            args_tys
                .iter()
                .cloned()
                .zip(conv_regs.iter_mut().zip(conv_tys.iter().cloned())),
        ) {
            if aty == cty {
                *creg = areg;
            } else {
                let reg = reg_of_ty!(self.regs, cty);
                self.convert(reg, cty, areg, aty)?;
                *creg = reg;
            }
        }

        let res_reg = if dst_ty == res_ty {
            reg_of_ty!(self.regs, res_ty)
        } else {
            dst_reg
        };

        // Helper macro for generating code for binary operators
        macro_rules! gen_op {
            ($op:tt, $( [$ty:tt, $inst:tt]),* ) => {
                match conv_tys[0] {
                    $( Ty::$ty => self.push(Instr::$inst(res_reg.into(),
                                        conv_regs[0].into(),
                                        conv_regs[1].into())), )*
                    _ => return err!("unexpected operands for {}", stringify!($op)),
                }
            }
        };

        match bf {
            Unop(Column) => self.push(Instr::GetColumn(res_reg.into(), conv_regs[0].into())),
            Unop(Not) => self.push(if conv_tys[0] == Ty::Str {
                Instr::NotStr(res_reg.into(), conv_regs[0].into())
            } else {
                debug_assert_eq!(conv_tys[0], Ty::Int);
                Instr::Not(res_reg.into(), conv_regs[0].into())
            }),
            Unop(Neg) => self.push(if conv_tys[0] == Ty::Float {
                Instr::NegFloat(res_reg.into(), conv_regs[0].into())
            } else {
                Instr::NegInt(res_reg.into(), conv_regs[0].into())
            }),
            Unop(Pos) => self.mov(res_reg, conv_regs[0], conv_tys[0])?,
            Binop(Plus) => gen_op!(Plus, [Float, AddFloat], [Int, AddInt]),
            Binop(Minus) => gen_op!(Minus, [Float, MinusFloat], [Int, MinusInt]),
            Binop(Mult) => gen_op!(Minus, [Float, MulFloat], [Int, MulInt]),
            Binop(Div) => gen_op!(Div, [Float, Div]),
            Binop(Mod) => gen_op!(Mod, [Float, ModFloat], [Int, ModInt]),
            Binop(Concat) => gen_op!(Concat, [Str, Concat]),
            Binop(Match) => gen_op!(Match, [Str, Match]),
            Binop(LT) => gen_op!(LT, [Float, LTFloat], [Int, LTInt], [Str, LTStr]),
            Binop(GT) => gen_op!(GT, [Float, GTFloat], [Int, GTInt], [Str, GTStr]),
            Binop(LTE) => gen_op!(LTE, [Float, LTEFloat], [Int, LTEInt], [Str, LTEStr]),
            Binop(GTE) => gen_op!(GTE, [Float, GTEFloat], [Int, GTEInt], [Str, GTEStr]),
            Binop(EQ) => gen_op!(EQ, [Float, EQFloat], [Int, EQInt], [Str, EQStr]),
            Contains => gen_op!(
                Contains,
                [MapIntInt, ContainsIntInt],
                [MapIntStr, ContainsIntStr],
                [MapIntFloat, ContainsIntFloat],
                [MapStrInt, ContainsStrInt],
                [MapStrStr, ContainsStrStr],
                [MapStrFloat, ContainsStrFloat]
            ),
            ReadErr => self.push(Instr::ReadErr(res_reg.into(), conv_regs[0].into())),
            Nextline => self.push(Instr::NextLine(res_reg.into(), conv_regs[0].into())),
            ReadErrStdin => self.push(Instr::ReadErrStdin(res_reg.into())),
            NextlineStdin => self.push(Instr::NextLineStdin(res_reg.into())),
            Setcol => self.push(Instr::SetColumn(conv_regs[0].into(), conv_regs[1].into())),
            Split => self.push(if conv_tys[1] == Ty::MapIntStr {
                Instr::SplitInt(
                    res_reg.into(),
                    conv_regs[0].into(),
                    conv_regs[1].into(),
                    conv_regs[2].into(),
                )
            } else if conv_tys[1] == Ty::MapStrStr {
                Instr::SplitStr(
                    res_reg.into(),
                    conv_regs[0].into(),
                    conv_regs[1].into(),
                    conv_regs[2].into(),
                )
            } else {
                return err!("invalid input types to split: {:?}", &conv_tys[..]);
            }),
            Length => self.push(match conv_tys[0] {
                Ty::MapIntInt => Instr::LenIntInt(res_reg.into(), conv_regs[0].into()),
                Ty::MapIntStr => Instr::LenIntStr(res_reg.into(), conv_regs[0].into()),
                Ty::MapIntFloat => Instr::LenIntFloat(res_reg.into(), conv_regs[0].into()),
                Ty::MapStrInt => Instr::LenStrInt(res_reg.into(), conv_regs[0].into()),
                Ty::MapStrStr => Instr::LenStrStr(res_reg.into(), conv_regs[0].into()),
                Ty::MapStrFloat => Instr::LenStrFloat(res_reg.into(), conv_regs[0].into()),
                Ty::Str => Instr::LenStr(res_reg.into(), conv_regs[0].into()),
                _ => return err!("invalid input type for length: {:?}", &conv_tys[..]),
            }),
            Print => {
                // XXX this imports a specific assumption on how the PrimStmt is generated, we may
                // want to make the bool parameter to Print dynamic.
                if let cfg::PrimVal::ILit(i) = &args[2] {
                    self.push(Instr::Print(
                        conv_regs[0].into(),
                        conv_regs[1].into(),
                        *i != 0,
                    ));
                    return Ok(());
                } else {
                    return err!("must pass constant append parameter to print");
                }
            }
            PrintStdout => {
                self.push(Instr::PrintStdout(conv_regs[0].into()));
                return Ok(());
            }
            Delete => match &conv_tys[0] {
                Ty::MapIntInt => self.push(Instr::DeleteIntInt(
                    conv_regs[0].into(),
                    conv_regs[1].into(),
                )),
                Ty::MapIntStr => self.push(Instr::DeleteIntStr(
                    conv_regs[0].into(),
                    conv_regs[1].into(),
                )),
                Ty::MapIntFloat => self.push(Instr::DeleteIntFloat(
                    conv_regs[0].into(),
                    conv_regs[1].into(),
                )),
                Ty::MapStrInt => self.push(Instr::DeleteStrInt(
                    conv_regs[0].into(),
                    conv_regs[1].into(),
                )),
                Ty::MapStrStr => self.push(Instr::DeleteStrStr(
                    conv_regs[0].into(),
                    conv_regs[1].into(),
                )),
                Ty::MapStrFloat => self.push(Instr::DeleteStrFloat(
                    conv_regs[0].into(),
                    conv_regs[1].into(),
                )),
                _ => return err!("incorrect parameter types for Delete: {:?}", &conv_tys[..]),
            },
        };
        self.convert(dst_reg, dst_ty, res_reg, res_ty)
    }

    fn expr(&mut self, dst_reg: u32, dst_ty: Ty, exp: &cfg::PrimExpr<'a>) -> Result<()> {
        match exp {
            PrimExpr::Val(v) => self.store(dst_reg, dst_ty, v)?,
            // Phi functions are handled elsewhere
            PrimExpr::Phi(_) => {}
            PrimExpr::CallBuiltin(bf, vs) => self.builtin(dst_reg, dst_ty, bf, vs)?,
            // TODO return register into dst_reg
            PrimExpr::CallUDF(func_id, vs) => {
                let mut arg_regs: SmallVec<_> = SmallVec::with_capacity(vs.len());
                let mut arg_tys: SmallVec<_> = SmallVec::with_capacity(vs.len());
                for v in vs.iter() {
                    let (reg, ty) = self.get_reg(v)?;
                    arg_regs.push(reg);
                    arg_tys.push(ty);
                }
                // Normalize the call
                let true_arity = self.arity_map[func_id] as usize;
                if arg_regs.len() < true_arity {
                    // Not enough arguments; fill in the rest with nulls.
                    // TODO reuse registers here. This is wasteful for functions with a lot of
                    // arguments.
                    let null_ty = types::null_ty();
                    let null_reg = reg_of_ty!(self.regs, null_ty);
                    for _ in 0..(true_arity-arg_regs.len()) {
                        arg_regs.push(null_reg);
                        arg_tys.push(null_ty);
                    }
                }
                if arg_regs.len() > true_arity {
                    // Too many arguments; don't pass the extra.
                    arg_regs.truncate(true_arity);
                    arg_tys.truncate(true_arity);
                }

                let monomorphized = self.id_map[&(*func_id, arg_tys.clone())];
                // Push onto the stack in reverse order.
                arg_regs.reverse();
                arg_tys.reverse();
                for (reg, ty) in arg_regs.iter().cloned().zip(arg_tys.iter().cloned()) {
                    use Ty::*;
                    self.push(match ty {
                        Int => Instr::PushInt(reg.into()),
                        Float => Instr::PushFloat(reg.into()),
                        Str => Instr::PushStr(reg.into()),
                        MapIntInt   => Instr::PushIntInt(reg.into()),
                        MapIntFloat => Instr::PushIntFloat(reg.into()),
                        MapIntStr   => Instr::PushIntStr(reg.into()),
                        MapStrInt   => Instr::PushStrInt(reg.into()),
                        MapStrFloat => Instr::PushStrFloat(reg.into()),
                        MapStrStr   => Instr::PushStrStr(reg.into()),
                        IterInt | IterStr => return err!("invalid argument type: {:?}", ty),
                    });
                }
                self.push(Instr::Call(monomorphized as usize));
                // TODO move the ret_reg information around so we can use it here.
                unimplemented!()
            },
            PrimExpr::Index(arr, k) => {
                let (arr_reg, arr_ty) = if let PrimVal::Var(arr_id) = arr {
                    self.reg_of_ident(arr_id)
                } else {
                    return err!("attempt to index into scalar literal: {}", arr);
                };
                self.load_map(dst_reg, dst_ty, arr_reg, arr_ty, k)?;
            }
            PrimExpr::IterBegin(pv) => {
                let (arr_reg, arr_ty) = self.get_reg(pv)?;
                if dst_ty.iter()? != arr_ty.key()? {
                    return err!(
                        "illegal iterator assignment {:?} = begin({:?})",
                        dst_ty,
                        arr_ty
                    );
                }
                self.push(match arr_ty {
                    Ty::MapIntInt => Instr::IterBeginIntInt(dst_reg.into(), arr_reg.into()),
                    Ty::MapIntFloat => Instr::IterBeginIntFloat(dst_reg.into(), arr_reg.into()),
                    Ty::MapIntStr => Instr::IterBeginIntStr(dst_reg.into(), arr_reg.into()),
                    Ty::MapStrInt => Instr::IterBeginStrInt(dst_reg.into(), arr_reg.into()),
                    Ty::MapStrFloat => Instr::IterBeginStrFloat(dst_reg.into(), arr_reg.into()),
                    Ty::MapStrStr => Instr::IterBeginStrStr(dst_reg.into(), arr_reg.into()),
                    Ty::Int | Ty::Float | Ty::Str | Ty::IterInt | Ty::IterStr => {
                        // covered by the error check above
                        unreachable!()
                    }
                });
            }
            PrimExpr::HasNext(pv) => {
                let target_reg = if dst_ty == Ty::Int {
                    dst_reg
                } else {
                    reg_of_ty!(self.regs, Ty::Int)
                };
                let (iter_reg, iter_ty) = self.get_reg(pv)?;
                self.push(match iter_ty.iter()? {
                    Ty::Int => Instr::IterHasNextInt(target_reg.into(), iter_reg.into()),
                    Ty::Str => Instr::IterHasNextStr(target_reg.into(), iter_reg.into()),
                    _ => unreachable!(),
                });
                self.convert(dst_reg, dst_ty, target_reg, Ty::Int)?
            }
            PrimExpr::Next(pv) => {
                let (iter_reg, iter_ty) = self.get_reg(pv)?;
                let elt_ty = iter_ty.iter()?;
                let target_reg = if dst_ty == elt_ty {
                    dst_reg
                } else {
                    reg_of_ty!(self.regs, elt_ty)
                };
                self.push(match elt_ty {
                    Ty::Int => Instr::IterGetNextInt(target_reg.into(), iter_reg.into()),
                    Ty::Str => Instr::IterGetNextStr(target_reg.into(), iter_reg.into()),
                    _ => unreachable!(),
                });
                self.convert(dst_reg, dst_ty, target_reg, elt_ty)?
            }
            PrimExpr::LoadBuiltin(bv) => {
                let target_ty = Ty::of_var(*bv);
                let target_reg = if target_ty == dst_ty {
                    dst_reg
                } else {
                    reg_of_ty!(self.regs, target_ty)
                };
                self.push(match target_ty {
                    Ty::Str => Instr::LoadVarStr(target_reg.into(), *bv),
                    Ty::Int => Instr::LoadVarInt(target_reg.into(), *bv),
                    Ty::MapIntStr => Instr::LoadVarIntMap(target_reg.into(), *bv),
                    _ => unreachable!(),
                });
                self.convert(dst_reg, dst_ty, target_reg, target_ty)?
            }
        };
        Ok(())
    }
    fn stmt(&mut self, stmt: &cfg::PrimStmt<'a>) -> Result<()> {
        match stmt {
            PrimStmt::AsgnIndex(arr, pv, pe) => {
                let (a_reg, a_ty) = self.reg_of_ident(arr);
                let (mut k_reg, k_ty) = self.get_reg(pv)?;
                let a_key_ty = a_ty.key()?;
                if k_ty != a_key_ty {
                    let conv_reg = reg_of_ty!(self.regs, a_key_ty);
                    self.convert(conv_reg, a_key_ty, k_reg, k_ty)?;
                    k_reg = conv_reg;
                }
                let v_ty = a_ty.val()?;
                let v_reg = reg_of_ty!(self.regs, v_ty);
                self.expr(v_reg, v_ty, pe)?;
                use Ty::*;
                self.push(match a_ty {
                    MapIntInt => Instr::StoreIntInt(a_reg.into(), k_reg.into(), v_reg.into()),
                    MapIntFloat => Instr::StoreIntFloat(a_reg.into(), k_reg.into(), v_reg.into()),
                    MapIntStr => Instr::StoreIntStr(a_reg.into(), k_reg.into(), v_reg.into()),
                    MapStrInt => Instr::StoreStrInt(a_reg.into(), k_reg.into(), v_reg.into()),
                    MapStrFloat => Instr::StoreStrFloat(a_reg.into(), k_reg.into(), v_reg.into()),
                    MapStrStr => Instr::StoreStrStr(a_reg.into(), k_reg.into(), v_reg.into()),
                    Int | Float | Str | IterInt | IterStr => {
                        return err!(
                            "in stmt {:?} computed type is non-map type {:?}",
                            stmt,
                            a_ty
                        )
                    }
                });
            }
            PrimStmt::AsgnVar(id, pe) => {
                let (dst_reg, dst_ty) = self.reg_of_ident(id);
                self.expr(dst_reg, dst_ty, pe)?;
            }
            PrimStmt::SetBuiltin(v, pe) => {
                let ty = Ty::of_var(*v);
                let reg = reg_of_ty!(self.regs, ty);
                self.expr(reg, ty, pe)?;
                use Ty::*;
                self.push(match ty {
                    Str => Instr::StoreVarStr(*v, reg.into()),
                    MapIntStr => Instr::StoreVarIntMap(*v, reg.into()),
                    Int => Instr::StoreVarInt(*v, reg.into()),
                    _ => return err!("unexpected type for variable {} : {:?}", v, ty),
                });
            }
            // TODO move v into return register (converting if necessary)
            // TODO insert Return.
            PrimStmt::Return(v) => unimplemented!(),
        };
        Ok(())
    }

    fn push(&mut self, i: Instr<'a>) {
        self.frame.instrs.push(i)
    }
    fn len(&self) -> usize {
        self.frame.instrs.len()
    }
}

impl Generator {
    // Get the register associated with a given identifier and assign a new one if it does not yet
    // have one.
    fn reg_of_ident(&mut self, id: &Ident) -> (u32, Ty) {
        if is_unused(*id) {
            // We should not actually store into the "unused" identifier.
            // TODO: remove this once there's better test coverage and we use more unsafe code.
            return (UNUSED, Ty::Int);
        }
        match self.registers.entry(*id) {
            Entry::Occupied(o) => o.get().clone(),
            Entry::Vacant(v) => {
                let ty: Ty = self
                    .ts
                    .get(v.key())
                    .expect("identifiers must be given types")
                    .clone();
                let reg = reg_of_ty!(self, ty);
                v.insert((reg, ty));
                (reg, ty)
            }
        }
    }
}

struct Instrs<'a>(Vec<Instr<'a>>);

impl<'a> Instrs<'a> {
    // Move src into dst at type Ty.
    fn mov(&mut self, dst_reg: u32, src_reg: u32, ty: Ty) -> Result<()> {
        use Ty::*;
        if dst_reg == UNUSED || src_reg == UNUSED {
            return Ok(());
        }
        let res = match ty {
            Int => Instr::MovInt(dst_reg.into(), src_reg.into()),
            Float => Instr::MovFloat(dst_reg.into(), src_reg.into()),
            Str => Instr::MovStr(dst_reg.into(), src_reg.into()),

            MapIntInt => Instr::MovMapIntInt(dst_reg.into(), src_reg.into()),
            MapIntFloat => Instr::MovMapIntFloat(dst_reg.into(), src_reg.into()),
            MapIntStr => Instr::MovMapIntStr(dst_reg.into(), src_reg.into()),

            MapStrInt => Instr::MovMapStrInt(dst_reg.into(), src_reg.into()),
            MapStrFloat => Instr::MovMapStrFloat(dst_reg.into(), src_reg.into()),
            MapStrStr => Instr::MovMapStrStr(dst_reg.into(), src_reg.into()),

            IterInt | IterStr => return err!("attempt to move values of type {:?}", ty),
        };
        Ok(self.0.push(res))
    }

    // Get the register associated with a value. For identifiers this has the same semantics as
    // reg_of_ident. For literals, a new register of the appropriate type is allocated.
    fn get_reg(&mut self, gen: &mut Generator, v: &PrimVal<'a>) -> Result<(u32, Ty)> {
        match v {
            PrimVal::ILit(i) => {
                let nreg = reg_of_ty!(gen, Ty::Int);
                self.push(Instr::StoreConstInt(nreg.into(), *i));
                Ok((nreg, Ty::Int))
            }
            PrimVal::FLit(f) => {
                let nreg = reg_of_ty!(gen, Ty::Float);
                self.push(Instr::StoreConstFloat(nreg.into(), *f));
                Ok((nreg, Ty::Float))
            }
            PrimVal::StrLit(s) => {
                let nreg = reg_of_ty!(gen, Ty::Str);
                self.push(Instr::StoreConstStr(nreg.into(), (*s).into()));
                Ok((nreg, Ty::Str))
            }
            PrimVal::Var(v) => Ok(gen.reg_of_ident(v)),
        }
    }

    // Convert src into dst. If the types match this is just a move. If both the types and the
    // registers match this is a noop.
    fn convert(&mut self, dst_reg: u32, dst_ty: Ty, src_reg: u32, src_ty: Ty) -> Result<()> {
        use Ty::*;
        if dst_reg == UNUSED {
            return Ok(());
        }
        if dst_reg == src_reg && dst_ty == src_ty {
            return Ok(());
        }
        let res = match (dst_ty, src_ty) {
            (Float, Int) => Instr::IntToFloat(dst_reg.into(), src_reg.into()),
            (Str, Int) => Instr::IntToStr(dst_reg.into(), src_reg.into()),

            (Int, Float) => Instr::FloatToInt(dst_reg.into(), src_reg.into()),
            (Str, Float) => Instr::FloatToStr(dst_reg.into(), src_reg.into()),

            (Int, Str) => Instr::StrToInt(dst_reg.into(), src_reg.into()),
            (Float, Str) => Instr::StrToFloat(dst_reg.into(), src_reg.into()),

            (MapIntFloat, MapIntFloat) => Instr::MovMapIntFloat(dst_reg.into(), src_reg.into()),
            (MapIntStr, MapIntStr) => Instr::MovMapIntStr(dst_reg.into(), src_reg.into()),

            (MapStrFloat, MapStrFloat) => Instr::MovMapStrFloat(dst_reg.into(), src_reg.into()),
            (MapStrStr, MapStrStr) => Instr::MovMapStrStr(dst_reg.into(), src_reg.into()),
            (dst, src) => {
                if dst == src {
                    return self.mov(dst_reg, src_reg, dst);
                }
                return err!(
                    "invalid attempt to convert or move dst={:?} src={:?}",
                    dst,
                    src
                );
            }
        };

        Ok(self.0.push(res))
    }

    // Store values into a register at a given type, converting if necessary.
    fn store(
        &mut self,
        gen: &mut Generator,
        dst_reg: u32,
        dst_ty: Ty,
        src: &PrimVal<'a>,
    ) -> Result<()> {
        match src {
            PrimVal::Var(id2) => {
                let (src_reg, src_ty) = gen.reg_of_ident(id2);
                self.convert(dst_reg, dst_ty, src_reg, src_ty)?;
            }
            PrimVal::ILit(i) => {
                if dst_ty == Ty::Int {
                    self.push(Instr::StoreConstInt(dst_reg.into(), *i));
                } else {
                    let ir = reg_of_ty!(gen, Ty::Int);
                    self.push(Instr::StoreConstInt(ir.into(), *i));
                    self.convert(dst_reg, dst_ty, ir, Ty::Int)?;
                }
            }
            PrimVal::FLit(f) => {
                if dst_ty == Ty::Float {
                    self.push(Instr::StoreConstFloat(dst_reg.into(), *f));
                } else {
                    let ir = reg_of_ty!(gen, Ty::Float);
                    self.push(Instr::StoreConstFloat(ir.into(), *f));
                    self.convert(dst_reg, dst_ty, ir, Ty::Float)?;
                }
            }
            PrimVal::StrLit(s) => {
                if dst_ty == Ty::Str {
                    self.push(Instr::StoreConstStr(dst_reg.into(), (*s).into()));
                } else {
                    let ir = reg_of_ty!(gen, Ty::Str);
                    self.push(Instr::StoreConstStr(ir.into(), (*s).into()));
                    self.convert(dst_reg, dst_ty, ir, Ty::Str)?;
                }
            }
        };
        Ok(())
    }

    // Generate bytecode for map lookups.
    fn load_map(
        &mut self,
        gen: &mut Generator,
        dst_reg: u32,
        dst_ty: Ty,
        arr_reg: u32,
        arr_ty: Ty,
        key: &PrimVal<'a>,
    ) -> Result<()> {
        if dst_reg == UNUSED {
            return Ok(());
        }
        // Convert `key` if necessary.
        let target_ty = arr_ty.key()?;
        let (mut key_reg, key_ty) = self.get_reg(gen, key)?;
        if target_ty != key_ty {
            let inter = reg_of_ty!(gen, target_ty);
            self.convert(inter, target_ty, key_reg, key_ty)?;
            key_reg = inter;
        }

        // Determine if we will need to convert the result when storing the variable.
        let arr_val_ty = arr_ty.val()?;
        let load_reg = if dst_ty == arr_val_ty {
            dst_reg
        } else {
            reg_of_ty!(gen, arr_val_ty)
        };

        // Emit the corresponding instruction.
        use Ty::*;
        self.push(match arr_ty {
            MapIntInt => Instr::LookupIntInt(load_reg.into(), arr_reg.into(), key_reg.into()),
            MapIntFloat => Instr::LookupIntFloat(load_reg.into(), arr_reg.into(), key_reg.into()),
            MapIntStr => Instr::LookupIntStr(load_reg.into(), arr_reg.into(), key_reg.into()),
            MapStrInt => Instr::LookupStrInt(load_reg.into(), arr_reg.into(), key_reg.into()),
            MapStrFloat => Instr::LookupStrFloat(load_reg.into(), arr_reg.into(), key_reg.into()),
            MapStrStr => Instr::LookupStrStr(load_reg.into(), arr_reg.into(), key_reg.into()),
            Int | Float | Str | IterInt | IterStr => {
                return err!("[load_map] expected map type, found {:?}", arr_ty)
            }
        });
        // Convert the result: note that if we had load_reg == dst_reg, then this is a noop.
        self.convert(dst_reg, dst_ty, load_reg, arr_val_ty)
    }
    fn builtin(
        &mut self,
        gen: &mut Generator,
        dst_reg: u32,
        dst_ty: Ty,
        bf: &builtins::Function,
        args: &cfg::SmallVec<PrimVal<'a>>,
    ) -> Result<()> {
        use crate::ast::{Binop::*, Unop::*};
        use builtins::Function::*;

        // Compile the argument values
        let mut args_regs = cfg::SmallVec::with_capacity(args.len());
        let mut args_tys = cfg::SmallVec::with_capacity(args.len());
        for arg in args.iter() {
            let (reg, ty) = self.get_reg(gen, arg)?;
            args_regs.push(reg);
            args_tys.push(ty);
        }

        // Now, perform any necessary conversions if input types do not match the argument types.
        let mut conv_regs: cfg::SmallVec<_> = smallvec![!0u32; args.len()];
        let (conv_tys, res_ty) = bf.type_sig(&args_tys[..])?;

        for (areg, (aty, (creg, cty))) in args_regs.iter().cloned().zip(
            args_tys
                .iter()
                .cloned()
                .zip(conv_regs.iter_mut().zip(conv_tys.iter().cloned())),
        ) {
            if aty == cty {
                *creg = areg;
            } else {
                let reg = reg_of_ty!(gen, cty);
                self.convert(reg, cty, areg, aty)?;
                *creg = reg;
            }
        }

        let res_reg = if dst_ty == res_ty {
            reg_of_ty!(gen, res_ty)
        } else {
            dst_reg
        };

        // Helper macro for generating code for binary operators
        macro_rules! gen_op {
            ($op:tt, $( [$ty:tt, $inst:tt]),* ) => {
                match conv_tys[0] {
                    $( Ty::$ty => self.push(Instr::$inst(res_reg.into(),
                                        conv_regs[0].into(),
                                        conv_regs[1].into())), )*
                    _ => return err!("unexpected operands for {}", stringify!($op)),
                }
            }
        };

        match bf {
            Unop(Column) => self.push(Instr::GetColumn(res_reg.into(), conv_regs[0].into())),
            Unop(Not) => self.push(if conv_tys[0] == Ty::Str {
                Instr::NotStr(res_reg.into(), conv_regs[0].into())
            } else {
                debug_assert_eq!(conv_tys[0], Ty::Int);
                Instr::Not(res_reg.into(), conv_regs[0].into())
            }),
            Unop(Neg) => self.push(if conv_tys[0] == Ty::Float {
                Instr::NegFloat(res_reg.into(), conv_regs[0].into())
            } else {
                Instr::NegInt(res_reg.into(), conv_regs[0].into())
            }),
            Unop(Pos) => self.mov(res_reg, conv_regs[0], conv_tys[0])?,
            Binop(Plus) => gen_op!(Plus, [Float, AddFloat], [Int, AddInt]),
            Binop(Minus) => gen_op!(Minus, [Float, MinusFloat], [Int, MinusInt]),
            Binop(Mult) => gen_op!(Minus, [Float, MulFloat], [Int, MulInt]),
            Binop(Div) => gen_op!(Div, [Float, Div]),
            Binop(Mod) => gen_op!(Mod, [Float, ModFloat], [Int, ModInt]),
            Binop(Concat) => gen_op!(Concat, [Str, Concat]),
            Binop(Match) => gen_op!(Match, [Str, Match]),
            Binop(LT) => gen_op!(LT, [Float, LTFloat], [Int, LTInt], [Str, LTStr]),
            Binop(GT) => gen_op!(GT, [Float, GTFloat], [Int, GTInt], [Str, GTStr]),
            Binop(LTE) => gen_op!(LTE, [Float, LTEFloat], [Int, LTEInt], [Str, LTEStr]),
            Binop(GTE) => gen_op!(GTE, [Float, GTEFloat], [Int, GTEInt], [Str, GTEStr]),
            Binop(EQ) => gen_op!(EQ, [Float, EQFloat], [Int, EQInt], [Str, EQStr]),
            Contains => gen_op!(
                Contains,
                [MapIntInt, ContainsIntInt],
                [MapIntStr, ContainsIntStr],
                [MapIntFloat, ContainsIntFloat],
                [MapStrInt, ContainsStrInt],
                [MapStrStr, ContainsStrStr],
                [MapStrFloat, ContainsStrFloat]
            ),
            ReadErr => self.push(Instr::ReadErr(res_reg.into(), conv_regs[0].into())),
            Nextline => self.push(Instr::NextLine(res_reg.into(), conv_regs[0].into())),
            ReadErrStdin => self.push(Instr::ReadErrStdin(res_reg.into())),
            NextlineStdin => self.push(Instr::NextLineStdin(res_reg.into())),
            Setcol => self.push(Instr::SetColumn(conv_regs[0].into(), conv_regs[1].into())),
            Split => self.push(if conv_tys[1] == Ty::MapIntStr {
                Instr::SplitInt(
                    res_reg.into(),
                    conv_regs[0].into(),
                    conv_regs[1].into(),
                    conv_regs[2].into(),
                )
            } else if conv_tys[1] == Ty::MapStrStr {
                Instr::SplitStr(
                    res_reg.into(),
                    conv_regs[0].into(),
                    conv_regs[1].into(),
                    conv_regs[2].into(),
                )
            } else {
                return err!("invalid input types to split: {:?}", &conv_tys[..]);
            }),
            Length => self.push(match conv_tys[0] {
                Ty::MapIntInt => Instr::LenIntInt(res_reg.into(), conv_regs[0].into()),
                Ty::MapIntStr => Instr::LenIntStr(res_reg.into(), conv_regs[0].into()),
                Ty::MapIntFloat => Instr::LenIntFloat(res_reg.into(), conv_regs[0].into()),
                Ty::MapStrInt => Instr::LenStrInt(res_reg.into(), conv_regs[0].into()),
                Ty::MapStrStr => Instr::LenStrStr(res_reg.into(), conv_regs[0].into()),
                Ty::MapStrFloat => Instr::LenStrFloat(res_reg.into(), conv_regs[0].into()),
                Ty::Str => Instr::LenStr(res_reg.into(), conv_regs[0].into()),
                _ => return err!("invalid input type for length: {:?}", &conv_tys[..]),
            }),
            Print => {
                // XXX this imports a specific assumption on how the PrimStmt is generated, we may
                // want to make the bool parameter to Print dynamic.
                if let cfg::PrimVal::ILit(i) = &args[2] {
                    self.push(Instr::Print(
                        conv_regs[0].into(),
                        conv_regs[1].into(),
                        *i != 0,
                    ));
                    return Ok(());
                } else {
                    return err!("must pass constant append parameter to print");
                }
            }
            PrintStdout => {
                self.push(Instr::PrintStdout(conv_regs[0].into()));
                return Ok(());
            }
            Delete => match &conv_tys[0] {
                Ty::MapIntInt => self.push(Instr::DeleteIntInt(
                    conv_regs[0].into(),
                    conv_regs[1].into(),
                )),
                Ty::MapIntStr => self.push(Instr::DeleteIntStr(
                    conv_regs[0].into(),
                    conv_regs[1].into(),
                )),
                Ty::MapIntFloat => self.push(Instr::DeleteIntFloat(
                    conv_regs[0].into(),
                    conv_regs[1].into(),
                )),
                Ty::MapStrInt => self.push(Instr::DeleteStrInt(
                    conv_regs[0].into(),
                    conv_regs[1].into(),
                )),
                Ty::MapStrStr => self.push(Instr::DeleteStrStr(
                    conv_regs[0].into(),
                    conv_regs[1].into(),
                )),
                Ty::MapStrFloat => self.push(Instr::DeleteStrFloat(
                    conv_regs[0].into(),
                    conv_regs[1].into(),
                )),
                _ => return err!("incorrect parameter types for Delete: {:?}", &conv_tys[..]),
            },
        };
        self.convert(dst_reg, dst_ty, res_reg, res_ty)
    }

    fn expr(
        &mut self,
        gen: &mut Generator,
        dst_reg: u32,
        dst_ty: Ty,
        exp: &cfg::PrimExpr<'a>,
    ) -> Result<()> {
        match exp {
            PrimExpr::Val(v) => self.store(gen, dst_reg, dst_ty, v)?,
            // Phi functions are handled elsewhere
            PrimExpr::Phi(_) => {}
            PrimExpr::CallBuiltin(bf, vs) => self.builtin(gen, dst_reg, dst_ty, bf, vs)?,
            PrimExpr::CallUDF(_func, _vs) => unimplemented!(),
            PrimExpr::Index(arr, k) => {
                let (arr_reg, arr_ty) = if let PrimVal::Var(arr_id) = arr {
                    gen.reg_of_ident(arr_id)
                } else {
                    return err!("attempt to index into scalar literal: {}", arr);
                };
                self.load_map(gen, dst_reg, dst_ty, arr_reg, arr_ty, k)?;
            }
            PrimExpr::IterBegin(pv) => {
                let (arr_reg, arr_ty) = self.get_reg(gen, pv)?;
                if dst_ty.iter()? != arr_ty.key()? {
                    return err!(
                        "illegal iterator assignment {:?} = begin({:?})",
                        dst_ty,
                        arr_ty
                    );
                }
                self.push(match arr_ty {
                    Ty::MapIntInt => Instr::IterBeginIntInt(dst_reg.into(), arr_reg.into()),
                    Ty::MapIntFloat => Instr::IterBeginIntFloat(dst_reg.into(), arr_reg.into()),
                    Ty::MapIntStr => Instr::IterBeginIntStr(dst_reg.into(), arr_reg.into()),
                    Ty::MapStrInt => Instr::IterBeginStrInt(dst_reg.into(), arr_reg.into()),
                    Ty::MapStrFloat => Instr::IterBeginStrFloat(dst_reg.into(), arr_reg.into()),
                    Ty::MapStrStr => Instr::IterBeginStrStr(dst_reg.into(), arr_reg.into()),
                    Ty::Int | Ty::Float | Ty::Str | Ty::IterInt | Ty::IterStr => {
                        // covered by the error check above
                        unreachable!()
                    }
                });
            }
            PrimExpr::HasNext(pv) => {
                let target_reg = if dst_ty == Ty::Int {
                    dst_reg
                } else {
                    reg_of_ty!(gen, Ty::Int)
                };
                let (iter_reg, iter_ty) = self.get_reg(gen, pv)?;
                self.push(match iter_ty.iter()? {
                    Ty::Int => Instr::IterHasNextInt(target_reg.into(), iter_reg.into()),
                    Ty::Str => Instr::IterHasNextStr(target_reg.into(), iter_reg.into()),
                    _ => unreachable!(),
                });
                self.convert(dst_reg, dst_ty, target_reg, Ty::Int)?
            }
            PrimExpr::Next(pv) => {
                let (iter_reg, iter_ty) = self.get_reg(gen, pv)?;
                let elt_ty = iter_ty.iter()?;
                let target_reg = if dst_ty == elt_ty {
                    dst_reg
                } else {
                    reg_of_ty!(gen, elt_ty)
                };
                self.push(match elt_ty {
                    Ty::Int => Instr::IterGetNextInt(target_reg.into(), iter_reg.into()),
                    Ty::Str => Instr::IterGetNextStr(target_reg.into(), iter_reg.into()),
                    _ => unreachable!(),
                });
                self.convert(dst_reg, dst_ty, target_reg, elt_ty)?
            }
            PrimExpr::LoadBuiltin(bv) => {
                let target_ty = Ty::of_var(*bv);
                let target_reg = if target_ty == dst_ty {
                    dst_reg
                } else {
                    reg_of_ty!(gen, target_ty)
                };
                self.push(match target_ty {
                    Ty::Str => Instr::LoadVarStr(target_reg.into(), *bv),
                    Ty::Int => Instr::LoadVarInt(target_reg.into(), *bv),
                    Ty::MapIntStr => Instr::LoadVarIntMap(target_reg.into(), *bv),
                    _ => unreachable!(),
                });
                self.convert(dst_reg, dst_ty, target_reg, target_ty)?
            }
        };
        Ok(())
    }
    fn stmt(&mut self, gen: &mut Generator, stmt: &cfg::PrimStmt<'a>) -> Result<()> {
        match stmt {
            PrimStmt::AsgnIndex(arr, pv, pe) => {
                let (a_reg, a_ty) = gen.reg_of_ident(arr);
                let (mut k_reg, k_ty) = self.get_reg(gen, pv)?;
                let a_key_ty = a_ty.key()?;
                if k_ty != a_key_ty {
                    let conv_reg = reg_of_ty!(gen, a_key_ty);
                    self.convert(conv_reg, a_key_ty, k_reg, k_ty)?;
                    k_reg = conv_reg;
                }
                let v_ty = a_ty.val()?;
                let v_reg = reg_of_ty!(gen, v_ty);
                self.expr(gen, v_reg, v_ty, pe)?;
                use Ty::*;
                self.push(match a_ty {
                    MapIntInt => Instr::StoreIntInt(a_reg.into(), k_reg.into(), v_reg.into()),
                    MapIntFloat => Instr::StoreIntFloat(a_reg.into(), k_reg.into(), v_reg.into()),
                    MapIntStr => Instr::StoreIntStr(a_reg.into(), k_reg.into(), v_reg.into()),
                    MapStrInt => Instr::StoreStrInt(a_reg.into(), k_reg.into(), v_reg.into()),
                    MapStrFloat => Instr::StoreStrFloat(a_reg.into(), k_reg.into(), v_reg.into()),
                    MapStrStr => Instr::StoreStrStr(a_reg.into(), k_reg.into(), v_reg.into()),
                    Int | Float | Str | IterInt | IterStr => {
                        return err!(
                            "in stmt {:?} computed type is non-map type {:?}",
                            stmt,
                            a_ty
                        )
                    }
                });
            }
            PrimStmt::AsgnVar(id, pe) => {
                let (dst_reg, dst_ty) = gen.reg_of_ident(id);
                self.expr(gen, dst_reg, dst_ty, pe)?;
            }
            PrimStmt::SetBuiltin(v, pe) => {
                let ty = Ty::of_var(*v);
                let reg = reg_of_ty!(gen, ty);
                self.expr(gen, reg, ty, pe)?;
                use Ty::*;
                self.push(match ty {
                    Str => Instr::StoreVarStr(*v, reg.into()),
                    MapIntStr => Instr::StoreVarIntMap(*v, reg.into()),
                    Int => Instr::StoreVarInt(*v, reg.into()),
                    _ => return err!("unexpected type for variable {} : {:?}", v, ty),
                });
            }
            PrimStmt::Return(_) => unimplemented!(),
        };
        Ok(())
    }

    fn push(&mut self, i: Instr<'a>) {
        self.0.push(i)
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn into_vec(self) -> Vec<Instr<'a>> {
        self.0
    }
}

pub(crate) fn bytecode<'a, 'b>(
    ctx: &cfg::Context<'a, &'b str>,
    // default to std::io::stdin()
    rdr: impl std::io::Read + 'static,
    // default to std::io::BufWriter::new(std::io::stdout())
    writer: impl std::io::Write + 'static,
) -> Result<Interp<'a>> {
    let mut instrs: Instrs<'a> = Instrs(Vec::new());
    let mut gen = Generator {
        registers: Default::default(),
        reg_counts: [0u32; NUM_TYPES],
        jmps: Default::default(),
        bb_to_instr: vec![0; ctx.cfg().node_count()],
        ts: types::get_types(ctx.cfg())?,
    };

    for (i, n) in ctx.cfg().raw_nodes().iter().enumerate() {
        gen.bb_to_instr[i] = instrs.len();
        for stmt in n.weight.0.iter() {
            instrs.stmt(&mut gen, stmt)?;
        }

        let ix = NodeIx::new(i);
        let mut branches: cfg::SmallVec<petgraph::graph::EdgeIndex> = Default::default();
        let mut walker = ctx.cfg().neighbors(ix).detach();
        while let Some(e) = walker.next_edge(ctx.cfg()) {
            branches.push(e)
        }

        // Petgraph gives us edges back in reverse order.
        branches.reverse();

        // Replace Phi functions in successors with assignments at this point in the stream.
        //
        // NB Why is it sufficient to do all assignments here? Shouldn't we limit code gen to  the
        // branch we are actually going to take? No, because one must first go through another
        // block that assigns to the node again before actually reading the variable.
        for n in ctx.cfg().neighbors(NodeIx::new(i)) {
            let weight = ctx.cfg().node_weight(n).unwrap();
            for stmt in weight.0.iter() {
                if let PrimStmt::AsgnVar(id, PrimExpr::Phi(preds)) = stmt {
                    let mut found = false;
                    for (pred, src) in preds.iter() {
                        if pred.index() == i {
                            found = true;
                            // now do the assignment
                            let (dst_reg, dst_ty) = gen.reg_of_ident(id);
                            let (src_reg, src_ty) = gen.reg_of_ident(src);
                            instrs.convert(dst_reg, dst_ty, src_reg, src_ty)?;
                        }
                    }
                    if !found {
                        return err!("malformed phi node: preds={:?} cur={:?}", &preds[..], i);
                    }
                } else {
                    break;
                }
            }
        }

        // Insert code for the branches at the end of this basic block. At first, labels just
        // indicate the basic block in the CFG. They are re-mapped at the end of execution.
        let mut is_end = true;
        for b in branches.iter().cloned() {
            let next = ctx.cfg().edge_endpoints(b).unwrap().1;
            match &ctx.cfg().edge_weight(b).unwrap().0 {
                Some(v) => {
                    let (mut reg, ty) = instrs.get_reg(&mut gen, v)?;
                    match ty {
                        Ty::Int => {}
                        Ty::Float => {
                            let dst = reg_of_ty!(&mut gen, Ty::Int);
                            instrs.convert(dst, Ty::Int, reg, Ty::Float)?;
                            reg = dst;
                        }
                        Ty::Str => {
                            let dst = reg_of_ty!(&mut gen, Ty::Int);
                            instrs.push(Instr::LenStr(dst.into(), reg.into()));
                            reg = dst;
                        }
                        _ => return err!("invalid type for branch: {:?}: {:?}", v, ty),
                    }

                    gen.jmps.push(instrs.len());
                    instrs.push(Instr::JmpIf(reg.into(), next.index().into()));
                }
                None => {
                    is_end = false;
                    // There's no point issuing an unconditional jump to the next basic block
                    // because it comes next in the instruction stream.
                    if next.index() != i + 1 {
                        gen.jmps.push(instrs.len());
                        instrs.push(Instr::Jmp(next.index().into()))
                    }
                }
            }
        }
        if is_end {
            instrs.push(Instr::Halt);
        }
    }

    // Rewrite jumps to point to proper offsets.
    for jmp in gen.jmps.iter().cloned() {
        match &mut instrs.0[jmp] {
            Instr::Jmp(bytecode::Label(l)) => *l = gen.bb_to_instr[*l as usize] as u32,
            Instr::JmpIf(_, bytecode::Label(l)) => *l = gen.bb_to_instr[*l as usize] as u32,
            _ => unreachable!(),
        }
    }

    Ok(Interp::new(
        instrs.into_vec(),
        |ty| gen.reg_counts[ty as usize] as usize,
        rdr,
        writer,
    ))
}
