use std::borrow::Borrow;

use hashbrown::{hash_map::Entry, HashMap};
use smallvec::smallvec;

use crate::builtins::{self, Variable};
use crate::bytecode::{Instr, Interp};
use crate::cfg::{self, Ident, PrimExpr, PrimStmt, PrimVal};
use crate::common::Result;
use crate::types::{get_types, Scalar, TVar};

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
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

impl Ty {
    fn from_scalar(o: &Option<Scalar>) -> Ty {
        match o {
            Some(Scalar::Int) => Ty::Int,
            Some(Scalar::Float) => Ty::Float,
            Some(Scalar::Str) => Ty::Str,
            // To respect printing
            None => Ty::Str,
        }
    }
    fn of_var(v: Variable) -> Ty {
        use Variable::*;
        match v {
            ARGC | FS | RS | FILENAME => Ty::Str,
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

impl<Q: Borrow<TVar<Option<Scalar>>>> From<Q> for Ty {
    fn from(t: Q) -> Ty {
        match t.borrow() {
            TVar::Scalar(s) => Ty::from_scalar(s),
            TVar::Iter(t) => match Ty::from_scalar(t) {
                Ty::Int => Ty::IterInt,
                Ty::Str => Ty::IterStr,
                _ => panic!("deduced invalid iterator value type: {:?}", t),
            },
            TVar::Map { key, val } => match (Ty::from_scalar(key), Ty::from_scalar(val)) {
                (Ty::Int, Ty::Int) => Ty::MapIntInt,
                (Ty::Int, Ty::Float) => Ty::MapIntFloat,
                (Ty::Int, Ty::Str) => Ty::MapIntStr,
                (Ty::Str, Ty::Int) => Ty::MapStrInt,
                (Ty::Str, Ty::Float) => Ty::MapStrFloat,
                (Ty::Str, Ty::Str) => Ty::MapStrStr,
                (x, y) => panic!("deduced invalid map type key={:?} val={:?}", x, y),
            },
        }
    }
}

const NUM_TYPES: usize = Ty::IterStr as usize + 1;

struct Generator {
    registers: HashMap<Ident, (u32, Ty)>,
    reg_counts: [u32; NUM_TYPES],
    jmps: Vec<usize>,
    bb_to_instr: Vec<usize>,
    ts: HashMap<Ident, TVar<Option<Scalar>>>,
}

// This is a macro to defeat the borrow checker when used inside methods for `Generator`.
macro_rules! reg_of_ty {
    ($slf:expr, $ty:expr) => {{
        let cnt = &mut $slf.reg_counts[$ty as usize];
        let res = *cnt;
        *cnt += 1;
        res
    }};
}

impl Generator {
    fn reg_of_ident(&mut self, id: &Ident) -> (u32, Ty) {
        match self.registers.entry(*id) {
            Entry::Occupied(o) => o.get().clone(),
            Entry::Vacant(v) => {
                let ty: Ty = self
                    .ts
                    .get(v.key())
                    .expect("identifiers must be given types")
                    .into();
                let reg = reg_of_ty!(self, ty);
                v.insert((reg, ty));
                (reg, ty)
            }
        }
    }
}

struct Instrs<'a>(Vec<Instr<'a>>);

impl<'a> Instrs<'a> {
    fn mov(&mut self, dst_reg: u32, src_reg: u32, ty: Ty) -> Result<()> {
        use Ty::*;
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
    fn convert(&mut self, dst_reg: u32, dst_ty: Ty, src_reg: u32, src_ty: Ty) -> Result<()> {
        use Ty::*;
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
    fn load_map(
        &mut self,
        gen: &mut Generator,
        dst_reg: u32,
        dst_ty: Ty,
        arr_reg: u32,
        arr_ty: Ty,
        key: &PrimVal<'a>,
    ) -> Result<()> {
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
            Unop(Column) => self.push(Instr::SetColumn(res_reg.into(), conv_regs[0].into())),
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
            Hasline => self.push(Instr::HasLine(res_reg.into(), conv_regs[0].into())),
            Nextline => self.push(Instr::NextLine(res_reg.into(), conv_regs[0].into())),
            Setcol => self.push(Instr::SetColumn(conv_regs[0].into(), conv_regs[1].into())),
            Split => self.push(if conv_tys[2] == Ty::MapIntStr {
                Instr::SplitInt(
                    res_reg.into(),
                    conv_regs[0].into(),
                    conv_regs[1].into(),
                    conv_regs[2].into(),
                )
            } else {
                Instr::SplitStr(
                    res_reg.into(),
                    conv_regs[0].into(),
                    conv_regs[1].into(),
                    conv_regs[2].into(),
                )
            }),
            Print => {
                // XXX this imports a specific assumption on how the PrimStmt is generated, we may
                // want to make the bool parameter to Print dynamic.
                if let cfg::PrimVal::ILit(i) = &args[2] {
                    self.push(Instr::Print(
                        conv_regs[0].into(),
                        conv_regs[1].into(),
                        *i != 0,
                    ))
                } else {
                    return err!("must pass constant append parameter to print");
                }
            }
            PrintStdout => self.push(Instr::PrintStdout(conv_regs[0].into())),
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

    fn push(&mut self, i: Instr<'a>) {
        self.0.push(i)
    }
}

pub(crate) fn bytecode<'a, 'b>(ctx: &cfg::Context<'a, &'b str>) -> Result<Interp<'a>> {
    let mut instrs: Instrs<'a> = Instrs(Vec::new());
    let mut gen = Generator {
        registers: Default::default(),
        reg_counts: [0u32; NUM_TYPES],
        jmps: Default::default(),
        bb_to_instr: Default::default(),
        ts: get_types(ctx.cfg(), ctx.num_idents())?,
    };

    // * We want a mapping from identifier -> register of its specific type.
    //   That can be a HashMap<Ident, (type, u32)>.
    // * To compute registers, we want a HashMap<type, u32>  (or we could use a vector and
    //   coerce these to numbers? YES)
    // * Upon getting to the end of a basic block B, traverse all neighbors, read prefix until it
    //   doesn't have a phi. For each phi block that reads x = [... B : y ...], append a stmt x=y to
    //   the instruction stream.
    // * Edges are handled in order (need to check! -- it's reverse order..) as jmp/jmpif statements.
    // * Have a vector keeping track of indexes of Jmp and JmpIf nodes in output stream. They start
    //   off with the node index for the basic block. As we start basic blocks, add a new entry to
    //   a vector keeping track of the start index. Once the stream is done, iterate over all
    //   Jmp/JmpIf nodes and replace the index with the right one.
    //
    // The actual instruction translation _should_ be pretty easy it ought to be a pretty direct
    // analog to the existing enum + type.
    for n in ctx.cfg().raw_nodes() {
        for stmt in n.weight.0.iter() {
            // use cfg::{PrimExpr::*, PrimStmt::*, PrimVal::*};
            match stmt {
                PrimStmt::AsgnIndex(arr, pv, pe) => {
                    let (a_reg, a_ty) = gen.reg_of_ident(arr);
                    let (k_reg, _k_ty) = instrs.get_reg(&mut gen, pv)?;
                    debug_assert_eq!(a_ty.key()?, _k_ty);
                    let v_ty = a_ty.val()?;
                    let v_reg = reg_of_ty!(&mut gen, v_ty);
                    instrs.expr(&mut gen, v_reg, v_ty, pe)?;
                    use Ty::*;
                    instrs.push(match a_ty {
                        MapIntInt => Instr::StoreIntInt(a_reg.into(), k_reg.into(), v_reg.into()),
                        MapIntFloat => {
                            Instr::StoreIntFloat(a_reg.into(), k_reg.into(), v_reg.into())
                        }
                        MapIntStr => Instr::StoreIntStr(a_reg.into(), k_reg.into(), v_reg.into()),
                        MapStrInt => Instr::StoreStrInt(a_reg.into(), k_reg.into(), v_reg.into()),
                        MapStrFloat => {
                            Instr::StoreStrFloat(a_reg.into(), k_reg.into(), v_reg.into())
                        }
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
                    instrs.expr(&mut gen, dst_reg, dst_ty, pe)?;
                }
                PrimStmt::SetBuiltin(v, pe) => {
                    let ty = Ty::of_var(*v);
                    let reg = reg_of_ty!(&mut gen, ty);
                    instrs.expr(&mut gen, reg, ty, pe)?;
                    use Ty::*;
                    instrs.push(match ty {
                        Str => Instr::StoreVarStr(*v, reg.into()),
                        MapIntStr => Instr::StoreVarIntMap(*v, reg.into()),
                        Int => Instr::StoreVarInt(*v, reg.into()),
                        _ => return err!("unexpected type for variable {} : {:?}", v, ty),
                    });
                }
            };
        }
        // TODO: handle phis
        // TODO: handle branches
    }

    unimplemented!()
}
