use crate::builtins;
use crate::bytecode;
use crate::cfg::{self, is_unused, Function, Ident, PrimExpr, PrimStmt, PrimVal, ProgramContext};
use crate::common::{Either, Graph, NodeIx, NumTy, Result};
use crate::compile::{Ty, NUM_TYPES, UNUSED};
use crate::smallvec::{self, smallvec};
use crate::types;

use hashbrown::{hash_map::Entry, HashMap, HashSet};
use std::mem;

type SmallVec<T> = smallvec::SmallVec<[T; 2]>;

enum HighLevel {
    Call {
        func_id: NumTy, /* monomorphized function id */
        dst_reg: NumTy,
        dst_ty: Ty, // TODO dst_ty may not be needed
        args: SmallVec<(NumTy, Ty)>,
    },
    Ret(NumTy, Ty), // TODO Same here, we may not need return types here explicitly
    Phi(NumTy, Ty, SmallVec<(NodeIx /*pred*/, NumTy /*register*/)>),
}

#[derive(Default)]
struct Registers {
    stats: RegStatuses,
    globals: HashMap<Ident, (u32, Ty)>,
}

#[derive(Debug, Copy, Clone)]
enum RegStatus {
    Local,
    Global,
    Ret,
}

#[derive(Default, Debug)]
struct RegStatuses([Vec<RegStatus>; NUM_TYPES]);

impl RegStatuses {
    fn reg_of_ty(&mut self, ty: Ty) -> NumTy {
        self.new_reg(ty, RegStatus::Local)
    }
    fn new_reg(&mut self, ty: Ty, status: RegStatus) -> NumTy {
        let v = &mut self.0[ty as usize];
        let res = v.len();
        v.push(status);
        res as NumTy
    }

    fn count(&self, ty: Ty) -> NumTy {
        self.0[ty as usize].len() as NumTy
    }

    fn get_status(&self, reg: NumTy, ty: Ty) -> RegStatus {
        self.0[ty as usize][reg as usize]
    }
}

type LL<'a> = bytecode::Instr<'a>;
type Instr<'a> = Either<LL<'a>, HighLevel>;
type CFG<'a> = Graph<Vec<Instr<'a>>, Option<(NumTy, Ty)>>;

#[derive(Default)]
struct Typer<'a> {
    regs: Registers,
    id_map: HashMap<
        // TODO: make newtypes for these different Ids?
        (
            NumTy,        /* cfg-level func id */
            SmallVec<Ty>, /* arg types */
        ),
        NumTy, /* bytecode-level func id */
    >,
    arity: HashMap<NumTy /* cfg-level func id */, NumTy>,
    // Why not just store FuncInfo's fields in a Frame?
    // We access Frames one at a time (through a View); but we need access to function arity and
    // return types across invidual views. We expose these fields in a separate type immutably to
    // facilitate that.
    //
    // Another option would be to pass a mutable reference to `frames` for all of bytecode-building
    // functions below, but then each access to frame would have the form of
    // self.frames[current_index], which is marginally less efficient and (more importantly)
    // error-prone.
    func_info: Vec<FuncInfo>,
    frames: Vec<Frame<'a>>,
    main_offset: usize,
}

impl<'a> Typer<'a> {
    fn init_from_ctx(pc: &ProgramContext<'a, &'a str>) -> Result<Typer<'a>> {
        // Type-check the code, then initialize a Typer, assigning registers to local
        // and global variables.

        let mut gen = Typer::default();
        let types::TypeInfo { var_tys, func_tys } = types::get_types(pc)?;
        macro_rules! init_entry {
            ($v:expr, $func_id:expr, $args:expr) => {
                let res = gen.frames.len() as NumTy;
                $v.insert(res);
                let ret_ty = func_tys[&($func_id, $args.clone())];
                let mut f = Frame::default();
                f.src_function = $func_id;
                f.cur_ident = res;
                gen.frames.push(f);
                gen.func_info.push(FuncInfo {
                    ret_ty,
                    // Want to allocate registers for args here, $args just contains the types.
                    arg_tys: $args.clone(),
                });
            };
        }
        for (func_id, func) in pc.funcs.iter().enumerate() {
            let arity = func.args.len() as NumTy;
            gen.arity.insert(func_id as NumTy, arity);
            if arity == 0 {
                let args: SmallVec<_> = Default::default();
                if let Entry::Vacant(v) = gen.id_map.entry((func_id as u32, args.clone())) {
                    init_entry!(v, func_id as u32, args);
                }
            }
        }
        for ((id, func_id, args), ty) in var_tys.iter() {
            let map = if id.global {
                &mut gen.regs.globals
            } else {
                if let Entry::Vacant(v) = gen.id_map.entry((*func_id, args.clone())) {
                    init_entry!(v, *func_id, args);
                }
                &mut gen.frames[gen.id_map[&(*func_id, args.clone())] as usize].locals
            };
            let reg = gen.regs.stats.new_reg(
                *ty,
                if id.global {
                    RegStatus::Global
                } else {
                    RegStatus::Local
                },
            );
            if let Some(old) = map.insert(*id, (reg, *ty)) {
                return err!(
                    "internal error: duplicate entries for same local in types  at id={:?}; {:?} vs {:?}",
                    id,
                    old,
                    (reg, *ty)
                );
            }
        }
        let main_offset = gen.id_map[&(pc.main_offset as NumTy, Default::default())];
        gen.main_offset = main_offset as usize;
        for frame in gen.frames.iter_mut() {
            let src_func = frame.src_function as usize;
            let mut stream = Vec::new();
            View {
                frame,
                regs: &mut gen.regs,
                id_map: &gen.id_map,
                arity: &gen.arity,
                func_info: &gen.func_info,
                stream: &mut stream,
            }
            .process_function(&pc.funcs[src_func])?;
        }
        Ok(gen)
    }
}

#[derive(Debug)]
struct FuncInfo {
    ret_ty: Ty,
    // For bytecode, we pop into each of these registers at the specified type.
    arg_tys: SmallVec<Ty>,
}

#[derive(Default)]
struct Frame<'a> {
    src_function: NumTy,
    cur_ident: NumTy,
    locals: HashMap<Ident, (u32, Ty)>,
    arg_regs: SmallVec<NumTy>,
    cfg: CFG<'a>,
}

struct View<'a, 'b> {
    frame: &'b mut Frame<'a>,
    regs: &'b mut Registers,
    id_map: &'b HashMap<(NumTy, SmallVec<Ty>), NumTy>,
    arity: &'b HashMap<NumTy, NumTy>,
    func_info: &'b Vec<FuncInfo>,
    // The current basic block being filled; It'll be swaped into `frame.cfg` as we translate a
    // given function cfg.
    stream: &'b mut Vec<Instr<'a>>,
}

impl<'a, 'b> View<'a, 'b> {
    fn process_function(&mut self, func: &Function<'a, &'a str>) -> Result<()> {
        // Record registers for arguments.
        for arg in func.args.iter() {
            let (reg, _) = self.reg_of_ident(&arg.id);
            self.frame.arg_regs.push(reg);
        }
        // Allocate basic blocks in CFG.
        for _ in 0..func.cfg.node_count() {
            self.frame.cfg.add_node(Default::default());
        }
        // Fill them in.
        for (i, n) in func.cfg.raw_nodes().iter().enumerate() {
            for stmt in n.weight.q.iter() {
                self.stmt(stmt)?;
            }
            let ix = NodeIx::new(i);
            let mut branches: SmallVec<petgraph::graph::EdgeIndex> = Default::default();
            let mut walker = func.cfg.neighbors(ix).detach();
            while let Some(e) = walker.next_edge(&func.cfg) {
                branches.push(e);
            }
            // We get branches back in reverse order.
            branches.reverse();
            for eix in branches.iter().cloned() {
                let transition = func.cfg.edge_weight(eix).unwrap();
                let edge = match &transition.0 {
                    Some(val) => Some(self.get_reg(val)?),
                    None => None,
                };
                let (src, dst) = func.cfg.edge_endpoints(eix).unwrap();
                self.frame.cfg.add_edge(src, dst, edge);
            }
            // In the interim, someone may have added some instructions to our basic block when
            // processing a Phi function. Merge in any of those changes.
            let cur_bb = self.frame.cfg.node_weight_mut(ix).unwrap();
            self.stream.extend(cur_bb.drain(..));
            mem::swap(cur_bb, self.stream);
        }
        Ok(())
    }

    // Get the register associated with a given identifier and assign a new one if it does not yet
    // have one.
    fn reg_of_ident(&mut self, id: &Ident) -> (u32, Ty) {
        let (reg, ty, _status) = self.reg_of_ident_status(id);
        (reg, ty)
    }
    fn reg_of_ident_status(&mut self, id: &Ident) -> (u32, Ty, RegStatus) {
        if is_unused(*id) {
            // We should not actually store into the "unused" identifier.
            // TODO: remove this once there's better test coverage and we use more unsafe code.
            return (UNUSED, Ty::Int, RegStatus::Local);
        }

        let ((res_reg, res_ty), status) = if id.global {
            (self.regs.globals[id], RegStatus::Global)
        } else {
            (self.frame.locals[id], RegStatus::Local)
        };
        (res_reg, res_ty, status)
    }
    fn pushl(&mut self, i: LL<'a>) {
        self.stream.push(Either::Left(i))
    }
    fn pushr(&mut self, i: HighLevel) {
        self.stream.push(Either::Right(i))
    }
    // Get the register associated with a value. For identifiers this has the same semantics as
    // reg_of_ident. For literals, a new register of the appropriate type is allocated.
    fn get_reg(&mut self, v: &PrimVal<'a>) -> Result<(u32, Ty)> {
        let (reg, ty, _status) = self.get_reg_status(v)?;
        Ok((reg, ty))
    }

    fn get_reg_status(&mut self, v: &PrimVal<'a>) -> Result<(u32, Ty, RegStatus)> {
        use RegStatus::*;
        match v {
            PrimVal::ILit(i) => {
                let nreg = self.regs.stats.new_reg(Ty::Int, Local);
                self.pushl(LL::StoreConstInt(nreg.into(), *i));
                Ok((nreg, Ty::Int, Local))
            }
            PrimVal::FLit(f) => {
                let nreg = self.regs.stats.new_reg(Ty::Float, Local);
                self.pushl(LL::StoreConstFloat(nreg.into(), *f));
                Ok((nreg, Ty::Float, Local))
            }
            PrimVal::StrLit(s) => {
                let nreg = self.regs.stats.new_reg(Ty::Str, Local);
                self.pushl(LL::StoreConstStr(nreg.into(), (*s).into()));
                Ok((nreg, Ty::Str, Local))
            }
            PrimVal::Var(v) => Ok(self.reg_of_ident_status(v)),
        }
    }

    // Move src into dst at type Ty.
    fn mov(&mut self, dst_reg: u32, src_reg: u32, ty: Ty) -> Result<()> {
        use Ty::*;
        if dst_reg == UNUSED || src_reg == UNUSED {
            return Ok(());
        }

        let res = match ty {
            Int => LL::MovInt(dst_reg.into(), src_reg.into()),
            Float => LL::MovFloat(dst_reg.into(), src_reg.into()),
            Str => LL::MovStr(dst_reg.into(), src_reg.into()),

            MapIntInt => LL::MovMapIntInt(dst_reg.into(), src_reg.into()),
            MapIntFloat => LL::MovMapIntFloat(dst_reg.into(), src_reg.into()),
            MapIntStr => LL::MovMapIntStr(dst_reg.into(), src_reg.into()),

            MapStrInt => LL::MovMapStrInt(dst_reg.into(), src_reg.into()),
            MapStrFloat => LL::MovMapStrFloat(dst_reg.into(), src_reg.into()),
            MapStrStr => LL::MovMapStrStr(dst_reg.into(), src_reg.into()),

            IterInt | IterStr => return err!("attempt to move values of type {:?}", ty),
        };
        Ok(self.pushl(res))
    }

    fn convert(&mut self, dst_reg: u32, dst_ty: Ty, src_reg: u32, src_ty: Ty) -> Result<()> {
        use Ty::*;
        if dst_reg == UNUSED || src_reg == UNUSED {
            return Ok(());
        }
        if dst_reg == src_reg && dst_ty == src_ty {
            return Ok(());
        }

        let res = match (dst_ty, src_ty) {
            (Float, Int) => LL::IntToFloat(dst_reg.into(), src_reg.into()),
            (Str, Int) => LL::IntToStr(dst_reg.into(), src_reg.into()),

            (Int, Float) => LL::FloatToInt(dst_reg.into(), src_reg.into()),
            (Str, Float) => LL::FloatToStr(dst_reg.into(), src_reg.into()),

            (Int, Str) => LL::StrToInt(dst_reg.into(), src_reg.into()),
            (Float, Str) => LL::StrToFloat(dst_reg.into(), src_reg.into()),

            (MapIntFloat, MapIntFloat) => LL::MovMapIntFloat(dst_reg.into(), src_reg.into()),
            (MapIntStr, MapIntStr) => LL::MovMapIntStr(dst_reg.into(), src_reg.into()),

            (MapStrFloat, MapStrFloat) => LL::MovMapStrFloat(dst_reg.into(), src_reg.into()),
            (MapStrStr, MapStrStr) => LL::MovMapStrStr(dst_reg.into(), src_reg.into()),
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

        Ok(self.pushl(res))
    }

    // Store values into a register at a given type, converting if necessary.
    fn store(&mut self, dst_reg: u32, dst_ty: Ty, src: &PrimVal<'a>) -> Result<()> {
        match src {
            PrimVal::Var(id) => {
                let (src_reg, src_ty) = self.reg_of_ident(id);
                self.convert(dst_reg, dst_ty, src_reg, src_ty)?;
            }
            PrimVal::ILit(i) => {
                if dst_ty == Ty::Int {
                    self.pushl(LL::StoreConstInt(dst_reg.into(), *i));
                } else {
                    let ir = self.regs.stats.reg_of_ty(Ty::Int);
                    self.pushl(LL::StoreConstInt(ir.into(), *i));
                    self.convert(dst_reg, dst_ty, ir, Ty::Int)?;
                }
            }
            PrimVal::FLit(f) => {
                if dst_ty == Ty::Float {
                    self.pushl(LL::StoreConstFloat(dst_reg.into(), *f));
                } else {
                    let ir = self.regs.stats.reg_of_ty(Ty::Float);
                    self.pushl(LL::StoreConstFloat(ir.into(), *f));
                    self.convert(dst_reg, dst_ty, ir, Ty::Float)?;
                }
            }
            PrimVal::StrLit(s) => {
                if dst_ty == Ty::Str {
                    self.pushl(LL::StoreConstStr(dst_reg.into(), (*s).into()));
                } else {
                    let ir = self.regs.stats.reg_of_ty(Ty::Str);
                    self.pushl(LL::StoreConstStr(ir.into(), (*s).into()));
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
            let inter = self.regs.stats.reg_of_ty(target_ty);
            self.convert(inter, target_ty, key_reg, key_ty)?;
            key_reg = inter;
        }

        // Determine if we will need to convert the result when storing the variable.
        let arr_val_ty = arr_ty.val()?;
        let load_reg = if dst_ty == arr_val_ty {
            dst_reg
        } else {
            self.regs.stats.reg_of_ty(arr_val_ty)
        };

        // Emit the corresponding instruction.
        use Ty::*;
        self.pushl(match arr_ty {
            MapIntInt => LL::LookupIntInt(load_reg.into(), arr_reg.into(), key_reg.into()),
            MapIntFloat => LL::LookupIntFloat(load_reg.into(), arr_reg.into(), key_reg.into()),
            MapIntStr => LL::LookupIntStr(load_reg.into(), arr_reg.into(), key_reg.into()),
            MapStrInt => LL::LookupStrInt(load_reg.into(), arr_reg.into(), key_reg.into()),
            MapStrFloat => LL::LookupStrFloat(load_reg.into(), arr_reg.into(), key_reg.into()),
            MapStrStr => LL::LookupStrStr(load_reg.into(), arr_reg.into(), key_reg.into()),
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
        let mut conv_regs: cfg::SmallVec<_> = smallvec![UNUSED; args.len()];
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
                let reg = self.regs.stats.reg_of_ty(cty);
                self.convert(reg, cty, areg, aty)?;
                *creg = reg;
            }
        }

        let mut res_reg = if dst_ty == res_ty {
            dst_reg
        } else {
            self.regs.stats.reg_of_ty(res_ty)
        };

        // Helper macro for generating code for binary operators
        macro_rules! gen_op {
            ($op:tt, $([$ty:tt, $inst:tt]),* ) => {
                match conv_tys[0] {
                    // TODO implement better handling of "NULL" and get rid of hacks like this.
                    $( Ty::$ty  => if res_reg != UNUSED {
                        self.pushl(LL::$inst(res_reg.into(),
                                        conv_regs[0].into(),
                                        conv_regs[1].into()));
                    }, )*
                    _ => return err!("unexpected operands for {}", stringify!($op)),
                }
            }
        };

        match bf {
            Unop(Column) => self.pushl(LL::GetColumn(res_reg.into(), conv_regs[0].into())),
            Unop(Not) => self.pushl(if conv_tys[0] == Ty::Str {
                LL::NotStr(res_reg.into(), conv_regs[0].into())
            } else {
                debug_assert_eq!(conv_tys[0], Ty::Int);
                LL::Not(res_reg.into(), conv_regs[0].into())
            }),
            Unop(Neg) => self.pushl(if conv_tys[0] == Ty::Float {
                LL::NegFloat(res_reg.into(), conv_regs[0].into())
            } else {
                LL::NegInt(res_reg.into(), conv_regs[0].into())
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
            ReadErr => self.pushl(LL::ReadErr(res_reg.into(), conv_regs[0].into())),
            Nextline => self.pushl(LL::NextLine(res_reg.into(), conv_regs[0].into())),
            ReadErrStdin => self.pushl(LL::ReadErrStdin(res_reg.into())),
            NextlineStdin => self.pushl(LL::NextLineStdin(res_reg.into())),
            Setcol => self.pushl(LL::SetColumn(conv_regs[0].into(), conv_regs[1].into())),
            Split => {
                if res_reg == UNUSED {
                    res_reg = self.regs.stats.reg_of_ty(res_ty);
                }
                self.pushl(if conv_tys[1] == Ty::MapIntStr {
                    LL::SplitInt(
                        res_reg.into(),
                        conv_regs[0].into(),
                        conv_regs[1].into(),
                        conv_regs[2].into(),
                    )
                } else if conv_tys[1] == Ty::MapStrStr {
                    LL::SplitStr(
                        res_reg.into(),
                        conv_regs[0].into(),
                        conv_regs[1].into(),
                        conv_regs[2].into(),
                    )
                } else {
                    return err!("invalid input types to split: {:?}", &conv_tys[..]);
                })
            }
            Length => self.pushl(match conv_tys[0] {
                Ty::MapIntInt => LL::LenIntInt(res_reg.into(), conv_regs[0].into()),
                Ty::MapIntStr => LL::LenIntStr(res_reg.into(), conv_regs[0].into()),
                Ty::MapIntFloat => LL::LenIntFloat(res_reg.into(), conv_regs[0].into()),
                Ty::MapStrInt => LL::LenStrInt(res_reg.into(), conv_regs[0].into()),
                Ty::MapStrStr => LL::LenStrStr(res_reg.into(), conv_regs[0].into()),
                Ty::MapStrFloat => LL::LenStrFloat(res_reg.into(), conv_regs[0].into()),
                Ty::Str => LL::LenStr(res_reg.into(), conv_regs[0].into()),
                _ => return err!("invalid input type for length: {:?}", &conv_tys[..]),
            }),
            Print => {
                // XXX this imports a specific assumption on how the PrimStmt is generated, we may
                // want to make the bool parameter to Print dynamic.
                if let cfg::PrimVal::ILit(i) = &args[2] {
                    self.pushl(LL::Print(conv_regs[0].into(), conv_regs[1].into(), *i != 0));
                    return Ok(());
                } else {
                    return err!("must pass constant append parameter to print");
                }
            }
            PrintStdout => {
                self.pushl(LL::PrintStdout(conv_regs[0].into()));
                return Ok(());
            }
            Delete => match &conv_tys[0] {
                Ty::MapIntInt => {
                    self.pushl(LL::DeleteIntInt(conv_regs[0].into(), conv_regs[1].into()))
                }
                Ty::MapIntStr => {
                    self.pushl(LL::DeleteIntStr(conv_regs[0].into(), conv_regs[1].into()))
                }
                Ty::MapIntFloat => {
                    self.pushl(LL::DeleteIntFloat(conv_regs[0].into(), conv_regs[1].into()))
                }
                Ty::MapStrInt => {
                    self.pushl(LL::DeleteStrInt(conv_regs[0].into(), conv_regs[1].into()))
                }
                Ty::MapStrStr => {
                    self.pushl(LL::DeleteStrStr(conv_regs[0].into(), conv_regs[1].into()))
                }
                Ty::MapStrFloat => {
                    self.pushl(LL::DeleteStrFloat(conv_regs[0].into(), conv_regs[1].into()))
                }
                _ => return err!("incorrect parameter types for Delete: {:?}", &conv_tys[..]),
            },
        };
        self.convert(dst_reg, dst_ty, res_reg, res_ty)
    }

    fn expr(&mut self, dst_reg: u32, dst_ty: Ty, exp: &cfg::PrimExpr<'a>) -> Result<()> {
        match exp {
            PrimExpr::Val(v) => self.store(dst_reg, dst_ty, v)?,
            PrimExpr::Phi(preds) => {
                let mut pred_regs = SmallVec::with_capacity(preds.len());
                for (prev, id) in preds.iter().cloned() {
                    let (id_reg, id_ty) = self.reg_of_ident(&id);
                    if id_ty == dst_ty {
                        pred_regs.push((prev, id_reg));
                        continue;
                    }
                    // `id` doesn't have the right type. To handle this, we need to reach into the
                    // cfg, grab the stream, and append conversion code to it.
                    //
                    // We do two loads into cfg for the sake of our friend the borrow checker.
                    // NB this is mostly required because Phis have to be at the top of
                    // their basic block. We could probably do the conversions here otherwise.
                    {
                        let other_bb = self.frame.cfg.node_weight_mut(prev).unwrap();
                        mem::swap(self.stream, other_bb);
                    }
                    let conv_reg = self.regs.stats.reg_of_ty(dst_ty);
                    self.convert(conv_reg, dst_ty, id_reg, id_ty)?;
                    {
                        let other_bb = self.frame.cfg.node_weight_mut(prev).unwrap();
                        mem::swap(self.stream, other_bb);
                    }
                    pred_regs.push((prev, conv_reg));
                }
                self.pushr(HighLevel::Phi(dst_reg, dst_ty, pred_regs));
            }
            PrimExpr::CallBuiltin(bf, vs) => self.builtin(dst_reg, dst_ty, bf, vs)?,
            PrimExpr::CallUDF(func_id, vs) => {
                let mut args = SmallVec::with_capacity(vs.len());
                for v in vs.iter() {
                    args.push(self.get_reg(v)?);
                }
                // Normalize the call
                let true_arity = self.arity[func_id] as usize;
                if args.len() < true_arity {
                    // Not enough arguments; fill in the rest with nulls.
                    // TODO reuse registers here. This is wasteful for functions with a lot of
                    // arguments.
                    let null_ty = types::null_ty();
                    let null_reg = self.regs.stats.reg_of_ty(null_ty);
                    for _ in 0..(true_arity - args.len()) {
                        args.push((null_reg, null_ty));
                    }
                }
                if args.len() > true_arity {
                    // Too many arguments; don't pass the extra.
                    args.truncate(true_arity);
                }
                let monomorphized =
                    self.id_map[&(*func_id, args.iter().map(|(_, y)| y).cloned().collect())];
                let info = &self.func_info[monomorphized as usize];

                if dst_ty != info.ret_ty {
                    let inter_reg = self.regs.stats.reg_of_ty(info.ret_ty);
                    self.pushr(HighLevel::Call {
                        dst_reg: inter_reg,
                        dst_ty: info.ret_ty,
                        func_id: monomorphized,
                        args,
                    });
                    self.convert(dst_reg, dst_ty, inter_reg, info.ret_ty)?;
                } else {
                    self.pushr(HighLevel::Call {
                        dst_reg,
                        dst_ty,
                        func_id: monomorphized,
                        args,
                    });
                }
            }
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
                self.pushl(match arr_ty {
                    Ty::MapIntInt => LL::IterBeginIntInt(dst_reg.into(), arr_reg.into()),
                    Ty::MapIntFloat => LL::IterBeginIntFloat(dst_reg.into(), arr_reg.into()),
                    Ty::MapIntStr => LL::IterBeginIntStr(dst_reg.into(), arr_reg.into()),
                    Ty::MapStrInt => LL::IterBeginStrInt(dst_reg.into(), arr_reg.into()),
                    Ty::MapStrFloat => LL::IterBeginStrFloat(dst_reg.into(), arr_reg.into()),
                    Ty::MapStrStr => LL::IterBeginStrStr(dst_reg.into(), arr_reg.into()),
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
                    self.regs.stats.reg_of_ty(Ty::Int)
                };
                let (iter_reg, iter_ty) = self.get_reg(pv)?;
                self.pushl(match iter_ty.iter()? {
                    Ty::Int => LL::IterHasNextInt(target_reg.into(), iter_reg.into()),
                    Ty::Str => LL::IterHasNextStr(target_reg.into(), iter_reg.into()),
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
                    self.regs.stats.reg_of_ty(elt_ty)
                };
                self.pushl(match elt_ty {
                    Ty::Int => LL::IterGetNextInt(target_reg.into(), iter_reg.into()),
                    Ty::Str => LL::IterGetNextStr(target_reg.into(), iter_reg.into()),
                    _ => unreachable!(),
                });
                self.convert(dst_reg, dst_ty, target_reg, elt_ty)?
            }
            PrimExpr::LoadBuiltin(bv) => {
                let target_ty = Ty::of_var(*bv);
                let target_reg = if target_ty == dst_ty {
                    dst_reg
                } else {
                    self.regs.stats.reg_of_ty(target_ty)
                };
                self.pushl(match target_ty {
                    Ty::Str => LL::LoadVarStr(target_reg.into(), *bv),
                    Ty::Int => LL::LoadVarInt(target_reg.into(), *bv),
                    Ty::MapIntStr => LL::LoadVarIntMap(target_reg.into(), *bv),
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
                    let conv_reg = self.regs.stats.reg_of_ty(a_key_ty);
                    self.convert(conv_reg, a_key_ty, k_reg, k_ty)?;
                    k_reg = conv_reg;
                }
                let v_ty = a_ty.val()?;
                let v_reg = self.regs.stats.reg_of_ty(v_ty);
                self.expr(v_reg, v_ty, pe)?;
                use Ty::*;
                self.pushl(match a_ty {
                    MapIntInt => LL::StoreIntInt(a_reg.into(), k_reg.into(), v_reg.into()),
                    MapIntFloat => LL::StoreIntFloat(a_reg.into(), k_reg.into(), v_reg.into()),
                    MapIntStr => LL::StoreIntStr(a_reg.into(), k_reg.into(), v_reg.into()),
                    MapStrInt => LL::StoreStrInt(a_reg.into(), k_reg.into(), v_reg.into()),
                    MapStrFloat => LL::StoreStrFloat(a_reg.into(), k_reg.into(), v_reg.into()),
                    MapStrStr => LL::StoreStrStr(a_reg.into(), k_reg.into(), v_reg.into()),
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
                let reg = self.regs.stats.reg_of_ty(ty);
                self.expr(reg, ty, pe)?;
                use Ty::*;
                self.pushl(match ty {
                    Str => LL::StoreVarStr(*v, reg.into()),
                    MapIntStr => LL::StoreVarIntMap(*v, reg.into()),
                    Int => LL::StoreVarInt(*v, reg.into()),
                    _ => return err!("unexpected type for variable {} : {:?}", v, ty),
                });
            }
            PrimStmt::Return(v) => {
                let (v_reg, v_ty) = self.get_reg(v)?;
                let ret_ty = self.func_info[self.frame.cur_ident as usize].ret_ty;
                if v_ty != ret_ty {
                    let ret_reg = self.regs.stats.reg_of_ty(ret_ty);
                    self.convert(ret_reg, ret_ty, v_reg, v_ty)?;
                    self.pushr(HighLevel::Ret(ret_reg, ret_ty));
                } else {
                    self.pushr(HighLevel::Ret(v_reg, v_ty));
                }
            }
        };
        Ok(())
    }
}
