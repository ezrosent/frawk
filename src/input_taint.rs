//! TODO: comment on what we allow and what we don't, emphasize that we're starting things off
//! pretty conservative.
use crate::bytecode::{Accum, Instr};
use crate::common::{FileSpec, Graph, NodeIx, NumTy, WorkList};
use crate::compile::{HighLevel, Ty};

use hashbrown::HashMap;
use petgraph::Direction;

#[derive(Eq, PartialEq, Hash, Clone)]
enum Key {
    Reg(NumTy, Ty),
    Rng,
}

impl<'a, T: Accum> From<&'a T> for Key {
    fn from(t: &T) -> Key {
        let (reg, ty) = t.reflect();
        Key::Reg(reg, ty)
    }
}

#[derive(Default)]
pub struct TaintedStringAnalysis {
    flows: Graph</*tainted=*/ bool, ()>,
    regs: HashMap<Key, NodeIx>,
    queries: Vec<Key>,
    wl: WorkList<NodeIx>,
}

impl TaintedStringAnalysis {
    pub(crate) fn visit_ll<'a>(&mut self, inst: &Instr<'a>) {
        use Instr::*;
        match inst {
            StoreConstStr(dst, _) => self.add_src(dst, /*tainted=*/ false),
            StoreConstInt(dst, _) => self.add_src(dst, /*tainted=*/ false),
            StoreConstFloat(dst, _) => self.add_src(dst, /*tainted=*/ false),

            IntToStr(dst, src) => self.add_dep(dst, src),
            IntToFloat(dst, src) => self.add_dep(dst, src),
            FloatToStr(dst, src) => self.add_dep(dst, src),
            FloatToInt(dst, src) => self.add_dep(dst, src),
            StrToFloat(dst, src) => self.add_dep(dst, src),
            LenStr(dst, src) | StrToInt(dst, src) | HexStrToInt(dst, src) => self.add_dep(dst, src),

            Mov(ty, dst, src) => self.add_dep(Key::Reg(*dst, *ty), Key::Reg(*src, *ty)),
            AddInt(dst, x, y)
            | MulInt(dst, x, y)
            | MinusInt(dst, x, y)
            | ModInt(dst, x, y)
            | Int2(_, dst, x, y) => {
                self.add_dep(dst, x);
                self.add_dep(dst, y);
            }
            AddFloat(dst, x, y)
            | MulFloat(dst, x, y)
            | MinusFloat(dst, x, y)
            | ModFloat(dst, x, y)
            | Div(dst, x, y)
            | Pow(dst, x, y)
            | Float2(_, dst, x, y) => {
                self.add_dep(dst, x);
                self.add_dep(dst, y);
            }

            Not(dst, src) | NegInt(dst, src) | Int1(_, dst, src) => self.add_dep(dst, src),
            NegFloat(dst, src) | Float1(_, dst, src) => self.add_dep(dst, src),
            NotStr(dst, src) => self.add_dep(dst, src),
            Rand(dst) => self.add_dep(dst, Key::Rng),
            Srand(old, new) => {
                self.add_dep(old, Key::Rng);
                self.add_dep(Key::Rng, new);
            }
            ReseedRng(new) => self.add_dep(Key::Rng, new),
            Concat(dst, x, y) => {
                self.add_dep(dst, x);
                self.add_dep(dst, y);
            }
            IsMatch(dst, x, y) | Match(dst, x, y) | SubstrIndex(dst, x, y) => {
                self.add_dep(dst, x);
                self.add_dep(dst, y);
            }
            GSub(dst, x, y, dstin) | Sub(dst, x, y, dstin) => {
                self.add_dep(dst, x);
                self.add_dep(dst, y);
                self.add_dep(dstin, x);
                self.add_dep(dstin, y);
            }
            EscapeTSV(dst, src) | EscapeCSV(dst, src) => self.add_dep(dst, src),
            Substr(dst, x, y, z) => {
                self.add_dep(dst, x);
                self.add_dep(dst, y);
                self.add_dep(dst, z);
            }
            LTFloat(dst, x, y)
            | GTFloat(dst, x, y)
            | LTEFloat(dst, x, y)
            | GTEFloat(dst, x, y)
            | EQFloat(dst, x, y) => {
                self.add_dep(dst, x);
                self.add_dep(dst, y);
            }
            LTInt(dst, x, y)
            | GTInt(dst, x, y)
            | LTEInt(dst, x, y)
            | GTEInt(dst, x, y)
            | EQInt(dst, x, y) => {
                self.add_dep(dst, x);
                self.add_dep(dst, y);
            }
            LTStr(dst, x, y)
            | GTStr(dst, x, y)
            | LTEStr(dst, x, y)
            | GTEStr(dst, x, y)
            | EQStr(dst, x, y) => {
                self.add_dep(dst, x);
                self.add_dep(dst, y);
            }
            GetColumn(dst, _) => self.add_src(dst, true),
            JoinTSV(dst, start, end) | JoinCSV(dst, start, end) => {
                self.add_dep(dst, start);
                self.add_dep(dst, end);
            }
            JoinColumns(dst, x, y, z) => {
                self.add_dep(dst, x);
                self.add_dep(dst, y);
                self.add_dep(dst, z);
            }
            // maybe a bit paranoid, but may as well.
            ReadErr(dst, _) => self.add_src(dst, true),
            ReadErrStdin(dst) => self.add_src(dst, true),
            NextLine(dst, _) => self.add_src(dst, true),
            NextLineStdin(dst) => self.add_src(dst, true),
            SplitInt(dst1, src1, dst2, src2) => {
                self.add_dep(dst1, src1);
                self.add_dep(dst1, src2);
                self.add_dep(dst2, src1);
                self.add_dep(dst2, src2);
            }
            SplitStr(dst1, src1, dst2, src2) => {
                self.add_dep(dst1, src1);
                self.add_dep(dst1, src2);
                self.add_dep(dst2, src1);
                self.add_dep(dst2, src2);
            }
            Sprintf { dst, fmt, args } => {
                self.add_dep(dst, fmt);
                for (reg, ty) in args.iter() {
                    self.add_dep(dst, Key::Reg(*reg, *ty));
                }
            }
            Printf {
                output: Some((cmd, FileSpec::Cmd)),
                ..
            } => self.queries.push(cmd.into()),
            Print(_, out, FileSpec::Cmd) => self.queries.push(out.into()),
            // NEXT UP: here
            Lookup{map_ty, dst, map, key } => unimplemented!(),
            Printf { .. }
            | PrintStdout(_)
            | Print(..)
            | Close(_)
            | NextLineStdinFused()
            | NextFile()
            | SetColumn(_, _)
            | AllocMap(_, _) => {}
            _ => unimplemented!(),
        }
    }
    fn get_node(&mut self, k: Key) -> NodeIx {
        let flows = &mut self.flows;
        let wl = &mut self.wl;
        self.regs
            .entry(k)
            .or_insert_with(|| {
                let ix = flows.add_node(false);
                wl.insert(ix);
                ix
            })
            .clone()
    }
    fn get_reg(&mut self, (reg, ty): (NumTy, Ty)) -> NodeIx {
        self.get_node(Key::Reg(reg, ty))
    }
    fn add_dep(&mut self, dst_reg: impl Into<Key>, src_reg: impl Into<Key>) {
        let src_node = self.get_node(src_reg.into());
        let dst_node = self.get_node(dst_reg.into());
        self.flows.add_edge(src_node, dst_node, ());
    }
    fn add_src(&mut self, reg: impl Into<Key>, tainted: bool) {
        let ix = self.get_node(reg.into());
        let w = self.flows.node_weight_mut(ix).unwrap();
        if *w != tainted {
            *w = tainted;
            self.wl.insert(ix);
        }
    }

    fn query(&mut self, reg: (NumTy, Ty)) -> bool {
        self.solve();
        let ix = self.get_reg(reg);
        *self.flows.node_weight(ix).unwrap()
    }

    fn solve(&mut self) {
        while let Some(n) = self.wl.pop() {
            let start = *self.flows.node_weight(n).unwrap();
            if start {
                continue;
            }
            let mut new = start;
            for n in self.flows.neighbors_directed(n, Direction::Incoming) {
                new |= *self.flows.node_weight(n).unwrap();
            }
            if !new {
                continue;
            }
            *self.flows.node_weight_mut(n).unwrap() = new;
            for n in self.flows.neighbors_directed(n, Direction::Outgoing) {
                self.wl.insert(n)
            }
        }
    }
}
