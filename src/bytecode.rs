use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use crate::builtins::{Bitwise, FloatFunc, Variable};
use crate::common::NumTy;
use crate::compile::{self, Ty};
use crate::interp::{index, index_mut, pop, push, Storage};
use crate::runtime::{self, Float, Int, Str, UniqueStr};

pub(crate) use crate::interp::Interp;

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub(crate) struct Label(pub usize);

impl std::fmt::Debug for Label {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "@{}", self.0)
    }
}

impl From<usize> for Label {
    fn from(u: usize) -> Label {
        Label(u)
    }
}

pub struct Reg<T>(u32, PhantomData<*const T>);

impl<T> std::fmt::Debug for Reg<T> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "<{}>", self.0)
    }
}

impl<T> From<u32> for Reg<T> {
    fn from(u: u32) -> Reg<T> {
        assert_ne!(u, compile::UNUSED, "creating an unused register");
        assert_ne!(u, compile::NULL_REG, "creating a null register");
        Reg(u, PhantomData)
    }
}
impl<T> Clone for Reg<T> {
    fn clone(&self) -> Reg<T> {
        Reg(self.0, PhantomData)
    }
}
impl<T> Copy for Reg<T> {}
impl<T> Hash for Reg<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}
impl<T> PartialEq for Reg<T> {
    fn eq(&self, other: &Reg<T>) -> bool {
        self.0 == other.0
    }
}
impl<T> Eq for Reg<T> {}
// PhantomData gets in the way here.
unsafe impl<T> Send for Reg<T> {}

#[derive(Debug, Clone)]
pub(crate) enum Instr<'a> {
    // By default, instructions have destination first, and src(s) second.
    StoreConstStr(Reg<Str<'a>>, UniqueStr<'a>),
    StoreConstInt(Reg<Int>, Int),
    StoreConstFloat(Reg<Float>, Float),

    // Conversions
    IntToStr(Reg<Str<'a>>, Reg<Int>),
    FloatToStr(Reg<Str<'a>>, Reg<Float>),
    StrToInt(Reg<Int>, Reg<Str<'a>>),
    HexStrToInt(Reg<Int>, Reg<Str<'a>>),
    FloatToInt(Reg<Int>, Reg<Float>),
    IntToFloat(Reg<Float>, Reg<Int>),
    StrToFloat(Reg<Float>, Reg<Str<'a>>),

    // Assignment
    // Note, for now we do not support iterator moves. Iterators own their own copy of an array,
    // and there is no reason we should be emitting movs for them.
    Mov(Ty, NumTy, NumTy),

    AllocMap(Ty, NumTy),

    // Math
    AddInt(Reg<Int>, Reg<Int>, Reg<Int>),
    AddFloat(Reg<Float>, Reg<Float>, Reg<Float>),
    MulFloat(Reg<Float>, Reg<Float>, Reg<Float>),
    MulInt(Reg<Int>, Reg<Int>, Reg<Int>),
    Div(Reg<Float>, Reg<Float>, Reg<Float>),
    Pow(Reg<Float>, Reg<Float>, Reg<Float>),
    MinusFloat(Reg<Float>, Reg<Float>, Reg<Float>),
    MinusInt(Reg<Int>, Reg<Int>, Reg<Int>),
    ModFloat(Reg<Float>, Reg<Float>, Reg<Float>),
    ModInt(Reg<Int>, Reg<Int>, Reg<Int>),
    Not(Reg<Int>, Reg<Int>),
    NotStr(Reg<Int>, Reg<Str<'a>>),
    NegInt(Reg<Int>, Reg<Int>),
    NegFloat(Reg<Float>, Reg<Float>),
    Float1(FloatFunc, Reg<Float>, Reg<Float>),
    Float2(FloatFunc, Reg<Float>, Reg<Float>, Reg<Float>),
    Int1(Bitwise, Reg<Int>, Reg<Int>),
    Int2(Bitwise, Reg<Int>, Reg<Int>, Reg<Int>),
    Rand(Reg<Float>),
    Srand(
        /* previous seed */ Reg<Int>,
        /* new seed */ Reg<Int>,
    ),
    ReseedRng(/* previous seed */ Reg<Int>),

    // String processing
    Concat(Reg<Str<'a>>, Reg<Str<'a>>, Reg<Str<'a>>),
    IsMatch(Reg<Int>, Reg<Str<'a>>, Reg<Str<'a>>),
    Match(Reg<Int>, Reg<Str<'a>>, Reg<Str<'a>>),
    // index(s, t) returns index of substring t in s, 0 if it does not appear.
    SubstrIndex(Reg<Int>, Reg<Str<'a>>, Reg<Str<'a>>),
    LenStr(Reg<Int>, Reg<Str<'a>>),
    Sub(
        Reg<Int>,
        /*pat*/ Reg<Str<'a>>,
        /*for*/ Reg<Str<'a>>,
        /*in*/ Reg<Str<'a>>,
    ),
    GSub(
        Reg<Int>,
        /*pat*/ Reg<Str<'a>>,
        /*for*/ Reg<Str<'a>>,
        /*in*/ Reg<Str<'a>>,
    ),
    EscapeCSV(Reg<Str<'a>>, Reg<Str<'a>>),
    EscapeTSV(Reg<Str<'a>>, Reg<Str<'a>>),
    Substr(Reg<Str<'a>>, Reg<Str<'a>>, Reg<Int>, Reg<Int>),

    // Comparison
    LTFloat(Reg<Int>, Reg<Float>, Reg<Float>),
    LTInt(Reg<Int>, Reg<Int>, Reg<Int>),
    LTStr(Reg<Int>, Reg<Str<'a>>, Reg<Str<'a>>),
    GTFloat(Reg<Int>, Reg<Float>, Reg<Float>),
    GTInt(Reg<Int>, Reg<Int>, Reg<Int>),
    GTStr(Reg<Int>, Reg<Str<'a>>, Reg<Str<'a>>),
    LTEFloat(Reg<Int>, Reg<Float>, Reg<Float>),
    LTEInt(Reg<Int>, Reg<Int>, Reg<Int>),
    LTEStr(Reg<Int>, Reg<Str<'a>>, Reg<Str<'a>>),
    GTEFloat(Reg<Int>, Reg<Float>, Reg<Float>),
    GTEInt(Reg<Int>, Reg<Int>, Reg<Int>),
    GTEStr(Reg<Int>, Reg<Str<'a>>, Reg<Str<'a>>),
    EQFloat(Reg<Int>, Reg<Float>, Reg<Float>),
    EQInt(Reg<Int>, Reg<Int>, Reg<Int>),
    EQStr(Reg<Int>, Reg<Str<'a>>, Reg<Str<'a>>),

    // Columns
    SetColumn(Reg<Int> /* dst column */, Reg<Str<'a>>),
    GetColumn(Reg<Str<'a>>, Reg<Int>),
    JoinCSV(
        Reg<Str<'a>>, /* dst */
        Reg<Int>,     /* start col */
        Reg<Int>,     /* end col */
    ),
    JoinTSV(
        Reg<Str<'a>>, /* dst */
        Reg<Int>,     /* start col */
        Reg<Int>,     /* end col */
    ),
    JoinColumns(
        Reg<Str<'a>>, /* dst */
        Reg<Int>,     /* start col */
        Reg<Int>,     /* end col */
        Reg<Str<'a>>, /* sep */
    ),

    // File reading.
    ReadErr(Reg<Int>, Reg<Str<'a>>),
    NextLine(Reg<Str<'a>>, Reg<Str<'a>>),
    ReadErrStdin(Reg<Int>),
    NextLineStdin(Reg<Str<'a>>),
    // Fetches line directly into $0.
    NextLineStdinFused(),
    // Advances early to the next file in our sequence
    NextFile(),

    // Split
    SplitInt(
        Reg<Int>,
        Reg<Str<'a>>,
        Reg<runtime::IntMap<Str<'a>>>,
        Reg<Str<'a>>,
    ),
    SplitStr(
        Reg<Int>,
        Reg<Str<'a>>,
        Reg<runtime::StrMap<'a, Str<'a>>>,
        Reg<Str<'a>>,
    ),
    Sprintf {
        dst: Reg<Str<'a>>,
        fmt: Reg<Str<'a>>,
        args: Vec<(NumTy, Ty)>,
    },
    Printf {
        output: Option<(Reg<Str<'a>>, bool)>,
        fmt: Reg<Str<'a>>,
        args: Vec<(NumTy, Ty)>,
    },
    PrintStdout(Reg<Str<'a>> /*text*/),
    Print(
        Reg<Str<'a>>, /*text*/
        Reg<Str<'a>>, /*output*/
        bool,         /*append*/
    ),
    Close(Reg<Str<'a>>),

    // Map operations
    Lookup {
        map_ty: Ty,
        dst: NumTy,
        map: NumTy,
        key: NumTy,
    },
    Contains {
        map_ty: Ty,
        dst: NumTy,
        map: NumTy,
        key: NumTy,
    },
    Delete {
        map_ty: Ty,
        map: NumTy,
        key: NumTy,
    },
    LenIntInt(Reg<Int>, Reg<runtime::IntMap<Int>>),
    LenIntFloat(Reg<Int>, Reg<runtime::IntMap<Float>>),
    LenIntStr(Reg<Int>, Reg<runtime::IntMap<Str<'a>>>),
    LenStrInt(Reg<Int>, Reg<runtime::StrMap<'a, Int>>),
    LenStrFloat(Reg<Int>, Reg<runtime::StrMap<'a, Float>>),
    LenStrStr(Reg<Int>, Reg<runtime::StrMap<'a, Str<'a>>>),

    IterBeginIntInt(Reg<runtime::Iter<Int>>, Reg<runtime::IntMap<Int>>),
    IterBeginIntStr(Reg<runtime::Iter<Int>>, Reg<runtime::IntMap<Str<'a>>>),
    IterBeginIntFloat(Reg<runtime::Iter<Int>>, Reg<runtime::IntMap<Float>>),
    IterBeginStrInt(Reg<runtime::Iter<Str<'a>>>, Reg<runtime::StrMap<'a, Int>>),
    IterBeginStrStr(
        Reg<runtime::Iter<Str<'a>>>,
        Reg<runtime::StrMap<'a, Str<'a>>>,
    ),
    IterBeginStrFloat(Reg<runtime::Iter<Str<'a>>>, Reg<runtime::StrMap<'a, Float>>),
    IterHasNextInt(Reg<Int>, Reg<runtime::Iter<Int>>),
    IterHasNextStr(Reg<Int>, Reg<runtime::Iter<Str<'a>>>),
    IterGetNextInt(Reg<Int>, Reg<runtime::Iter<Int>>),
    IterGetNextStr(Reg<Str<'a>>, Reg<runtime::Iter<Str<'a>>>),
    StoreIntInt(Reg<runtime::IntMap<Int>>, Reg<Int>, Reg<Int>),
    StoreIntStr(Reg<runtime::IntMap<Str<'a>>>, Reg<Int>, Reg<Str<'a>>),
    StoreIntFloat(Reg<runtime::IntMap<Float>>, Reg<Int>, Reg<Float>),
    StoreStrInt(Reg<runtime::StrMap<'a, Int>>, Reg<Str<'a>>, Reg<Int>),
    StoreStrStr(
        Reg<runtime::StrMap<'a, Str<'a>>>,
        Reg<Str<'a>>,
        Reg<Str<'a>>,
    ),
    StoreStrFloat(Reg<runtime::StrMap<'a, Float>>, Reg<Str<'a>>, Reg<Float>),

    // Special variables
    LoadVarStr(Reg<Str<'a>>, Variable),
    StoreVarStr(Variable, Reg<Str<'a>>),
    LoadVarInt(Reg<Int>, Variable),
    StoreVarInt(Variable, Reg<Int>),
    LoadVarIntMap(Reg<runtime::IntMap<Str<'a>>>, Variable),
    StoreVarIntMap(Variable, Reg<runtime::IntMap<Str<'a>>>),

    LoadSlotInt(Reg<Int>, Int),
    LoadSlotFloat(Reg<Float>, Int),
    LoadSlotStr(Reg<Str<'a>>, Int),
    LoadSlotIntInt(Reg<runtime::IntMap<Int>>, Int),
    LoadSlotIntFloat(Reg<runtime::IntMap<Float>>, Int),
    LoadSlotIntStr(Reg<runtime::IntMap<Str<'a>>>, Int),
    LoadSlotStrInt(Reg<runtime::StrMap<'a, Int>>, Int),
    LoadSlotStrFloat(Reg<runtime::StrMap<'a, Float>>, Int),
    LoadSlotStrStr(Reg<runtime::StrMap<'a, Str<'a>>>, Int),

    StoreSlotInt(Reg<Int>, Int),
    StoreSlotFloat(Reg<Float>, Int),
    StoreSlotStr(Reg<Str<'a>>, Int),
    StoreSlotIntInt(Reg<runtime::IntMap<Int>>, Int),
    StoreSlotIntFloat(Reg<runtime::IntMap<Float>>, Int),
    StoreSlotIntStr(Reg<runtime::IntMap<Str<'a>>>, Int),
    StoreSlotStrInt(Reg<runtime::StrMap<'a, Int>>, Int),
    StoreSlotStrFloat(Reg<runtime::StrMap<'a, Float>>, Int),
    StoreSlotStrStr(Reg<runtime::StrMap<'a, Str<'a>>>, Int),

    // Control
    JmpIf(Reg<Int>, Label),
    Jmp(Label),
    Halt,

    // Functions
    PushInt(Reg<Int>),
    PushFloat(Reg<Float>),
    PushStr(Reg<Str<'a>>),
    PushIntInt(Reg<runtime::IntMap<Int>>),
    PushIntFloat(Reg<runtime::IntMap<Float>>),
    PushIntStr(Reg<runtime::IntMap<Str<'a>>>),

    PushStrInt(Reg<runtime::StrMap<'a, Int>>),
    PushStrFloat(Reg<runtime::StrMap<'a, Float>>),
    PushStrStr(Reg<runtime::StrMap<'a, Str<'a>>>),

    PopInt(Reg<Int>),
    PopFloat(Reg<Float>),
    PopStr(Reg<Str<'a>>),
    PopIntInt(Reg<runtime::IntMap<Int>>),
    PopIntFloat(Reg<runtime::IntMap<Float>>),
    PopIntStr(Reg<runtime::IntMap<Str<'a>>>),

    PopStrInt(Reg<runtime::StrMap<'a, Int>>),
    PopStrFloat(Reg<runtime::StrMap<'a, Float>>),
    PopStrStr(Reg<runtime::StrMap<'a, Str<'a>>>),

    Call(usize),
    Ret,
}

impl<T> Reg<T> {
    pub(crate) fn index(&self) -> usize {
        self.0 as usize
    }
}

// For accumulating register-specific metadata
pub(crate) trait Accum {
    fn reflect(&self) -> (NumTy, compile::Ty);
    fn accum(&self, mut f: impl FnMut(NumTy, compile::Ty)) {
        let (reg, ty) = self.reflect();
        f(reg, ty)
    }
}

pub(crate) trait Get<T> {
    fn get(&self, r: Reg<T>) -> &T;
    fn get_mut(&mut self, r: Reg<T>) -> &mut T;
}
pub(crate) trait Pop<T> {
    fn push(&mut self, r: Reg<T>);
    fn pop(&mut self, r: Reg<T>);
}

fn _dbg_check_index<T>(desc: &str, Storage { regs, .. }: &Storage<T>, r: usize) {
    assert!(
        r < regs.len(),
        "[{}] index {} is out of bounds (len={})",
        desc,
        r,
        regs.len()
    );
}

macro_rules! impl_pop {
    ($t:ty, $fld:ident) => {
        impl<'a, LR: runtime::LineReader> Pop<$t> for Interp<'a, LR> {
            fn push(&mut self, r: Reg<$t>) {
                push(&mut self.$fld, &r)
            }
            fn pop(&mut self, r: Reg<$t>) {
                let v = pop(&mut self.$fld);
                *self.get_mut(r) = v;
            }
        }
    };
}

macro_rules! impl_accum  {
    ($t:ty, $ty:tt, $($lt:tt),+) => {
        impl<$($lt),*> Accum for Reg<$t> {
            fn reflect(&self) -> (NumTy, compile::Ty) {
                (self.index() as NumTy, compile::Ty::$ty)
            }
        }
    };
    ($t:ty, $ty:tt,) => {
        impl Accum for Reg<$t> {
            fn reflect(&self) -> (NumTy, compile::Ty) {
                (self.index() as NumTy, compile::Ty::$ty)
            }
        }
    };
}

macro_rules! impl_get {
    ($t:ty, $fld:ident, $ty:tt $(,$lt:tt)*) => {
        impl_accum!($t, $ty, $($lt),*);
        impl<'a, LR: runtime::LineReader> Get<$t> for Interp<'a, LR> {
            #[inline(always)]
            fn get(&self, r: Reg<$t>) -> &$t {
                #[cfg(debug_assertions)]
                _dbg_check_index(
                    concat!(stringify!($t), "_", stringify!($fld)),
                    &self.$fld,
                    r.index(),
                );
                index(&self.$fld, &r)
            }
            #[inline(always)]
            fn get_mut(&mut self, r: Reg<$t>) -> &mut $t {
                #[cfg(debug_assertions)]
                _dbg_check_index(
                    concat!(stringify!($t), "_", stringify!($fld)),
                    &self.$fld,
                    r.index(),
                );
                index_mut(&mut self.$fld, &r)
            }
        }
    };
}

macro_rules! impl_all {
    ($t:ty, $fld:ident, $ty:tt $(,$lt:tt)*) => {
        impl_get!($t, $fld, $ty $(,$lt)* );
        impl_pop!($t, $fld);
    };
}

impl_all!(Int, ints, Int);
impl_all!(Str<'a>, strs, Str, 'a);
impl_all!(Float, floats, Float);
impl_all!(runtime::IntMap<Float>, maps_int_float, MapIntFloat);
impl_all!(runtime::IntMap<Int>, maps_int_int, MapIntInt);
impl_all!(runtime::IntMap<Str<'a>>, maps_int_str, MapIntStr, 'a);
impl_all!(runtime::StrMap<'a, Float>, maps_str_float, MapStrFloat, 'a);
impl_all!(runtime::StrMap<'a, Int>, maps_str_int, MapStrInt, 'a);
impl_all!(runtime::StrMap<'a, Str<'a>>, maps_str_str, MapStrStr, 'a);
impl_get!(runtime::Iter<Int>, iters_int, IterInt);
impl_get!(runtime::Iter<Str<'a>>, iters_str, IterStr, 'a);

// Helpful for avoiding big match statements when computing basic walks of the bytecode.
impl<'a> Instr<'a> {
    pub(crate) fn accum(&self, mut f: impl FnMut(NumTy, compile::Ty)) {
        use Instr::*;
        match self {
            StoreConstStr(sr, _s) => sr.accum(&mut f),
            StoreConstInt(ir, _i) => ir.accum(&mut f),
            StoreConstFloat(fr, _f) => fr.accum(&mut f),
            IntToStr(sr, ir) => {
                sr.accum(&mut f);
                ir.accum(&mut f)
            }
            FloatToStr(sr, fr) => {
                sr.accum(&mut f);
                fr.accum(&mut f);
            }
            StrToInt(ir, sr) | HexStrToInt(ir, sr) => {
                ir.accum(&mut f);
                sr.accum(&mut f);
            }
            StrToFloat(fr, sr) => {
                fr.accum(&mut f);
                sr.accum(&mut f);
            }
            FloatToInt(ir, fr) => {
                ir.accum(&mut f);
                fr.accum(&mut f);
            }
            IntToFloat(fr, ir) => {
                fr.accum(&mut f);
                ir.accum(&mut f);
            }
            AddInt(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            AddFloat(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            MulInt(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            MulFloat(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            MinusInt(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            MinusFloat(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            ModInt(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            ModFloat(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            Pow(res, l, r) | Div(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            Not(res, ir) => {
                res.accum(&mut f);
                ir.accum(&mut f)
            }
            NotStr(res, sr) => {
                res.accum(&mut f);
                sr.accum(&mut f)
            }
            NegInt(res, ir) => {
                res.accum(&mut f);
                ir.accum(&mut f)
            }
            NegFloat(res, fr) => {
                res.accum(&mut f);
                fr.accum(&mut f)
            }
            Float1(_, dst, src) => {
                dst.accum(&mut f);
                src.accum(&mut f);
            }
            Float2(_, dst, x, y) => {
                dst.accum(&mut f);
                x.accum(&mut f);
                y.accum(&mut f);
            }
            Int1(_, dst, src) => {
                dst.accum(&mut f);
                src.accum(&mut f);
            }
            Int2(_, dst, x, y) => {
                dst.accum(&mut f);
                x.accum(&mut f);
                y.accum(&mut f);
            }
            Rand(res) => res.accum(&mut f),
            Srand(res, seed) => {
                res.accum(&mut f);
                seed.accum(&mut f)
            }
            ReseedRng(res) => res.accum(&mut f),
            Concat(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            Match(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            IsMatch(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            SubstrIndex(res, s, t) => {
                res.accum(&mut f);
                s.accum(&mut f);
                t.accum(&mut f);
            }
            LenStr(res, s) => {
                res.accum(&mut f);
                s.accum(&mut f)
            }
            GSub(res, pat, s, in_s) | Sub(res, pat, s, in_s) => {
                res.accum(&mut f);
                pat.accum(&mut f);
                s.accum(&mut f);
                in_s.accum(&mut f);
            }
            EscapeCSV(res, s) | EscapeTSV(res, s) => {
                res.accum(&mut f);
                s.accum(&mut f);
            }
            Substr(res, base, l, r) => {
                res.accum(&mut f);
                base.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            LTFloat(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            LTInt(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            LTStr(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            GTFloat(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            GTInt(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            GTStr(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            LTEFloat(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            LTEInt(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            LTEStr(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            GTEFloat(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            GTEInt(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            GTEStr(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            EQFloat(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            EQInt(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            EQStr(res, l, r) => {
                res.accum(&mut f);
                l.accum(&mut f);
                r.accum(&mut f);
            }
            SetColumn(dst, src) => {
                dst.accum(&mut f);
                src.accum(&mut f)
            }
            GetColumn(dst, src) => {
                dst.accum(&mut f);
                src.accum(&mut f)
            }
            JoinCSV(dst, start, end) | JoinTSV(dst, start, end) => {
                dst.accum(&mut f);
                start.accum(&mut f);
                end.accum(&mut f);
            }
            JoinColumns(dst, start, end, sep) => {
                dst.accum(&mut f);
                start.accum(&mut f);
                end.accum(&mut f);
                sep.accum(&mut f);
            }
            SplitInt(flds, to_split, arr, pat) => {
                flds.accum(&mut f);
                to_split.accum(&mut f);
                arr.accum(&mut f);
                pat.accum(&mut f);
            }
            SplitStr(flds, to_split, arr, pat) => {
                flds.accum(&mut f);
                to_split.accum(&mut f);
                arr.accum(&mut f);
                pat.accum(&mut f);
            }
            Sprintf { dst, fmt, args } => {
                dst.accum(&mut f);
                fmt.accum(&mut f);
                for (reg, ty) in args.iter().cloned() {
                    f(reg, ty);
                }
            }
            Printf { output, fmt, args } => {
                if let Some((path_reg, _)) = output {
                    path_reg.accum(&mut f);
                }
                fmt.accum(&mut f);
                for (reg, ty) in args.iter().cloned() {
                    f(reg, ty);
                }
            }
            PrintStdout(txt) => txt.accum(&mut f),
            Print(txt, out, _append) => {
                txt.accum(&mut f);
                out.accum(&mut f)
            }
            Close(file) => file.accum(&mut f),
            Lookup {
                map_ty,
                dst,
                map,
                key,
            } => {
                let (k, v) = (map_ty.key().unwrap(), map_ty.val().unwrap());
                f(*dst, v);
                f(*key, k);
                f(*map, *map_ty);
            }
            Contains {
                map_ty,
                dst,
                map,
                key,
            } => {
                let k = map_ty.key().unwrap();
                f(*dst, Ty::Int);
                f(*key, k);
                f(*map, *map_ty);
            }
            Delete { map_ty, map, key } => {
                let k = map_ty.key().unwrap();
                f(*key, k);
                f(*map, *map_ty);
            }
            LenIntInt(res, arr) => {
                res.accum(&mut f);
                arr.accum(&mut f)
            }
            LenIntFloat(res, arr) => {
                res.accum(&mut f);
                arr.accum(&mut f)
            }
            LenIntStr(res, arr) => {
                res.accum(&mut f);
                arr.accum(&mut f)
            }
            LenStrInt(res, arr) => {
                res.accum(&mut f);
                arr.accum(&mut f)
            }
            LenStrFloat(res, arr) => {
                res.accum(&mut f);
                arr.accum(&mut f)
            }
            LenStrStr(res, arr) => {
                res.accum(&mut f);
                arr.accum(&mut f)
            }
            StoreIntInt(arr, k, v) => {
                arr.accum(&mut f);
                k.accum(&mut f);
                v.accum(&mut f)
            }
            StoreIntFloat(arr, k, v) => {
                arr.accum(&mut f);
                k.accum(&mut f);
                v.accum(&mut f)
            }
            StoreIntStr(arr, k, v) => {
                arr.accum(&mut f);
                k.accum(&mut f);
                v.accum(&mut f)
            }
            StoreStrInt(arr, k, v) => {
                arr.accum(&mut f);
                k.accum(&mut f);
                v.accum(&mut f)
            }
            StoreStrFloat(arr, k, v) => {
                arr.accum(&mut f);
                k.accum(&mut f);
                v.accum(&mut f)
            }
            StoreStrStr(arr, k, v) => {
                arr.accum(&mut f);
                k.accum(&mut f);
                v.accum(&mut f)
            }
            LoadVarStr(dst, _var) => dst.accum(&mut f),
            StoreVarStr(_var, src) => src.accum(&mut f),
            LoadVarInt(dst, _var) => dst.accum(&mut f),
            StoreVarInt(_var, src) => src.accum(&mut f),
            LoadVarIntMap(dst, _var) => dst.accum(&mut f),
            StoreVarIntMap(_var, src) => src.accum(&mut f),

            LoadSlotInt(dst, _) => dst.accum(&mut f),
            LoadSlotFloat(dst, _) => dst.accum(&mut f),
            LoadSlotStr(dst, _) => dst.accum(&mut f),
            LoadSlotIntInt(dst, _) => dst.accum(&mut f),
            LoadSlotIntFloat(dst, _) => dst.accum(&mut f),
            LoadSlotIntStr(dst, _) => dst.accum(&mut f),
            LoadSlotStrInt(dst, _) => dst.accum(&mut f),
            LoadSlotStrFloat(dst, _) => dst.accum(&mut f),
            LoadSlotStrStr(dst, _) => dst.accum(&mut f),

            StoreSlotInt(src, _) => src.accum(&mut f),
            StoreSlotFloat(src, _) => src.accum(&mut f),
            StoreSlotStr(src, _) => src.accum(&mut f),
            StoreSlotIntInt(src, _) => src.accum(&mut f),
            StoreSlotIntFloat(src, _) => src.accum(&mut f),
            StoreSlotIntStr(src, _) => src.accum(&mut f),
            StoreSlotStrInt(src, _) => src.accum(&mut f),
            StoreSlotStrFloat(src, _) => src.accum(&mut f),
            StoreSlotStrStr(src, _) => src.accum(&mut f),

            IterBeginIntInt(dst, arr) => {
                dst.accum(&mut f);
                arr.accum(&mut f)
            }
            IterBeginIntFloat(dst, arr) => {
                dst.accum(&mut f);
                arr.accum(&mut f)
            }
            IterBeginIntStr(dst, arr) => {
                dst.accum(&mut f);
                arr.accum(&mut f)
            }
            IterBeginStrInt(dst, arr) => {
                dst.accum(&mut f);
                arr.accum(&mut f)
            }
            IterBeginStrFloat(dst, arr) => {
                dst.accum(&mut f);
                arr.accum(&mut f)
            }
            IterBeginStrStr(dst, arr) => {
                dst.accum(&mut f);
                arr.accum(&mut f)
            }
            IterHasNextInt(dst, iter) => {
                dst.accum(&mut f);
                iter.accum(&mut f)
            }
            IterHasNextStr(dst, iter) => {
                dst.accum(&mut f);
                iter.accum(&mut f)
            }
            IterGetNextInt(dst, iter) => {
                dst.accum(&mut f);
                iter.accum(&mut f)
            }
            IterGetNextStr(dst, iter) => {
                dst.accum(&mut f);
                iter.accum(&mut f)
            }
            Mov(ty, dst, src) => {
                f(*dst, *ty);
                f(*src, *ty);
            }
            AllocMap(ty, reg) => f(*reg, *ty),
            ReadErr(dst, file) => {
                dst.accum(&mut f);
                file.accum(&mut f)
            }
            NextLine(dst, file) => {
                dst.accum(&mut f);
                file.accum(&mut f)
            }
            ReadErrStdin(dst) => dst.accum(&mut f),
            NextLineStdin(dst) => dst.accum(&mut f),
            JmpIf(cond, _lbl) => cond.accum(&mut f),
            PushInt(reg) => reg.accum(&mut f),
            PushFloat(reg) => reg.accum(&mut f),
            PushStr(reg) => reg.accum(&mut f),
            PushIntInt(reg) => reg.accum(&mut f),
            PushIntFloat(reg) => reg.accum(&mut f),
            PushIntStr(reg) => reg.accum(&mut f),
            PushStrInt(reg) => reg.accum(&mut f),
            PushStrFloat(reg) => reg.accum(&mut f),
            PushStrStr(reg) => reg.accum(&mut f),
            PopInt(reg) => reg.accum(&mut f),
            PopFloat(reg) => reg.accum(&mut f),
            PopStr(reg) => reg.accum(&mut f),
            PopIntInt(reg) => reg.accum(&mut f),
            PopIntFloat(reg) => reg.accum(&mut f),
            PopIntStr(reg) => reg.accum(&mut f),
            PopStrInt(reg) => reg.accum(&mut f),
            PopStrFloat(reg) => reg.accum(&mut f),
            PopStrStr(reg) => reg.accum(&mut f),
            NextFile() | NextLineStdinFused() | Call(_) | Jmp(_) | Ret | Halt => {}
        }
    }
}
