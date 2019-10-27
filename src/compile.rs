use std::borrow::Borrow;

use hashbrown::{hash_map::Entry, HashMap};

use crate::bytecode::{Instr, Interp};
use crate::cfg::{self, Ident, PrimExpr, PrimStmt, PrimVal};
use crate::common::Result;
use crate::types::{get_types, Scalar, TVar};

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum Ty {
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
    registers: HashMap<Ident, (Ty, u32)>,
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
    fn reg_of_ident(&mut self, id: &Ident) -> (Ty, u32) {
        match self.registers.entry(*id) {
            Entry::Occupied(o) => o.get().clone(),
            Entry::Vacant(v) => {
                let ty: Ty = self
                    .ts
                    .get(v.key())
                    .expect("identifiers must be given types")
                    .into();
                let reg = reg_of_ty!(self, ty);
                v.insert((ty, reg));
                (ty, reg)
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
    fn convert(&mut self, dst_reg: u32, dst_ty: Ty, src_reg: u32, src_ty: Ty) -> Result<()> {
        use Ty::*;
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
            use cfg::{PrimExpr::*, PrimStmt::*, PrimVal::*};
            match stmt {
                PrimStmt::AsgnIndex(arr, pv, pe) => unimplemented!(),
                PrimStmt::AsgnVar(id, pe) => {
                    let (dst_ty, dst_reg) = gen.reg_of_ident(id);
                    match pe {
                        PrimExpr::Val(v) => match v {
                            PrimVal::Var(id2) => {
                                let (src_ty, src_reg) = gen.reg_of_ident(id2);
                                instrs.convert(dst_reg, dst_ty, src_reg, src_ty)?;
                            }
                            PrimVal::ILit(i) => {
                                if dst_ty == Ty::Int {
                                    instrs.push(Instr::StoreConstInt(dst_reg.into(), *i));
                                } else {
                                    let ir = reg_of_ty!(&mut gen, Ty::Int);
                                    instrs.push(Instr::StoreConstInt(ir.into(), *i));
                                    instrs.convert(dst_reg, dst_ty, ir, Ty::Int)?;
                                }
                            }
                            PrimVal::FLit(f) => {
                                if dst_ty == Ty::Float {
                                    instrs.push(Instr::StoreConstFloat(dst_reg.into(), *f));
                                } else {
                                    let ir = reg_of_ty!(&mut gen, Ty::Float);
                                    instrs.push(Instr::StoreConstFloat(ir.into(), *f));
                                    instrs.convert(dst_reg, dst_ty, ir, Ty::Float)?;
                                }
                            }
                            PrimVal::StrLit(s) => {
                                if dst_ty == Ty::Str {
                                    instrs.push(Instr::StoreConstStr(dst_reg.into(), (*s).into()));
                                } else {
                                    let ir = reg_of_ty!(&mut gen, Ty::Str);
                                    instrs.push(Instr::StoreConstStr(ir.into(), (*s).into()));
                                    instrs.convert(dst_reg, dst_ty, ir, Ty::Str)?;
                                }
                            }
                        },
                        PrimExpr::Phi(_) => {}
                        PrimExpr::CallBuiltin(bf, vs) => unimplemented!(),
                        PrimExpr::Index(arr, k) => unimplemented!(),
                        PrimExpr::IterBegin(pv) => unimplemented!(),
                        PrimExpr::HasNext(pv) => unimplemented!(),
                        PrimExpr::Next(pv) => unimplemented!(),
                        PrimExpr::LoadBuiltin(bv) => unimplemented!(),
                    }
                }
                PrimStmt::SetBuiltin(v, pe) => unimplemented!(),
            };
        }
        // TODO: handle phis
    }

    unimplemented!()
}
