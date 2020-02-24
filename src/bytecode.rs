use std::marker::PhantomData;

use crate::builtins::Variable;
use crate::common::NumTy;
use crate::compile::{self, Ty};
use crate::interp::{index, index_mut, pop, push, Storage};
use crate::runtime::{self, Float, Int, Str};

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

pub(crate) struct Reg<T>(u32, PhantomData<*const T>);

impl<T> std::fmt::Debug for Reg<T> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "<{}>", self.0)
    }
}

impl<T> From<u32> for Reg<T> {
    fn from(u: u32) -> Reg<T> {
        Reg(u, PhantomData)
    }
}
impl<T> Clone for Reg<T> {
    fn clone(&self) -> Reg<T> {
        Reg(self.0, PhantomData)
    }
}
impl<T> Copy for Reg<T> {}

#[derive(Debug, Clone)]
pub(crate) enum Instr<'a> {
    // By default, instructions have destination first, and src(s) second.
    StoreConstStr(Reg<Str<'a>>, Str<'a>),
    StoreConstInt(Reg<Int>, Int),
    StoreConstFloat(Reg<Float>, Float),

    // Conversions
    IntToStr(Reg<Str<'a>>, Reg<Int>),
    FloatToStr(Reg<Str<'a>>, Reg<Float>),
    StrToInt(Reg<Int>, Reg<Str<'a>>),
    FloatToInt(Reg<Int>, Reg<Float>),
    IntToFloat(Reg<Float>, Reg<Int>),
    StrToFloat(Reg<Float>, Reg<Str<'a>>),

    // Assignment
    MovInt(Reg<Int>, Reg<Int>),
    MovFloat(Reg<Float>, Reg<Float>),
    MovStr(Reg<Str<'a>>, Reg<Str<'a>>),

    MovMapIntInt(Reg<runtime::IntMap<Int>>, Reg<runtime::IntMap<Int>>),
    MovMapIntFloat(Reg<runtime::IntMap<Float>>, Reg<runtime::IntMap<Float>>),
    MovMapIntStr(Reg<runtime::IntMap<Str<'a>>>, Reg<runtime::IntMap<Str<'a>>>),

    MovMapStrInt(Reg<runtime::StrMap<'a, Int>>, Reg<runtime::StrMap<'a, Int>>),
    MovMapStrFloat(
        Reg<runtime::StrMap<'a, Float>>,
        Reg<runtime::StrMap<'a, Float>>,
    ),
    MovMapStrStr(
        Reg<runtime::StrMap<'a, Str<'a>>>,
        Reg<runtime::StrMap<'a, Str<'a>>>,
    ),
    // Note, for now we do not support iterator moves. Iterators own their own copy of an array,
    // and there is no reason we should be emitting movs for them.

    // Math
    AddInt(Reg<Int>, Reg<Int>, Reg<Int>),
    AddFloat(Reg<Float>, Reg<Float>, Reg<Float>),
    MulFloat(Reg<Float>, Reg<Float>, Reg<Float>),
    MulInt(Reg<Int>, Reg<Int>, Reg<Int>),
    Div(Reg<Float>, Reg<Float>, Reg<Float>),
    MinusFloat(Reg<Float>, Reg<Float>, Reg<Float>),
    MinusInt(Reg<Int>, Reg<Int>, Reg<Int>),
    ModFloat(Reg<Float>, Reg<Float>, Reg<Float>),
    ModInt(Reg<Int>, Reg<Int>, Reg<Int>),
    Not(Reg<Int>, Reg<Int>),
    NotStr(Reg<Int>, Reg<Str<'a>>),
    NegInt(Reg<Int>, Reg<Int>),
    NegFloat(Reg<Float>, Reg<Float>),

    // String processing
    Concat(Reg<Str<'a>>, Reg<Str<'a>>, Reg<Str<'a>>),
    IsMatch(Reg<Int>, Reg<Str<'a>>, Reg<Str<'a>>),
    Match(Reg<Int>, Reg<Str<'a>>, Reg<Str<'a>>),
    LenStr(Reg<Int>, Reg<Str<'a>>),

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

    // File reading.
    ReadErr(Reg<Int>, Reg<Str<'a>>),
    NextLine(Reg<Str<'a>>, Reg<Str<'a>>),
    ReadErrStdin(Reg<Int>),
    NextLineStdin(Reg<Str<'a>>),

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
    LookupIntInt(Reg<Int>, Reg<runtime::IntMap<Int>>, Reg<Int>),
    LookupIntStr(Reg<Str<'a>>, Reg<runtime::IntMap<Str<'a>>>, Reg<Int>),
    LookupIntFloat(Reg<Float>, Reg<runtime::IntMap<Float>>, Reg<Int>),
    LookupStrInt(Reg<Int>, Reg<runtime::StrMap<'a, Int>>, Reg<Str<'a>>),
    LookupStrStr(
        Reg<Str<'a>>,
        Reg<runtime::StrMap<'a, Str<'a>>>,
        Reg<Str<'a>>,
    ),
    LookupStrFloat(Reg<Float>, Reg<runtime::StrMap<'a, Float>>, Reg<Str<'a>>),
    ContainsIntInt(Reg<Int>, Reg<runtime::IntMap<Int>>, Reg<Int>),
    ContainsIntStr(Reg<Int>, Reg<runtime::IntMap<Str<'a>>>, Reg<Int>),
    ContainsIntFloat(Reg<Int>, Reg<runtime::IntMap<Float>>, Reg<Int>),
    ContainsStrInt(Reg<Int>, Reg<runtime::StrMap<'a, Int>>, Reg<Str<'a>>),
    ContainsStrStr(Reg<Int>, Reg<runtime::StrMap<'a, Str<'a>>>, Reg<Str<'a>>),
    ContainsStrFloat(Reg<Int>, Reg<runtime::StrMap<'a, Float>>, Reg<Str<'a>>),
    DeleteIntInt(Reg<runtime::IntMap<Int>>, Reg<Int>),
    DeleteIntStr(Reg<runtime::IntMap<Str<'a>>>, Reg<Int>),
    DeleteIntFloat(Reg<runtime::IntMap<Float>>, Reg<Int>),
    DeleteStrInt(Reg<runtime::StrMap<'a, Int>>, Reg<Str<'a>>),
    DeleteStrStr(Reg<runtime::StrMap<'a, Str<'a>>>, Reg<Str<'a>>),
    DeleteStrFloat(Reg<runtime::StrMap<'a, Float>>, Reg<Str<'a>>),
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
        impl<'a> Pop<$t> for Interp<'a> {
            #[inline(always)]
            fn push(&mut self, r: Reg<$t>) {
                push(&mut self.$fld, &r)
            }
            #[inline(always)]
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
        impl<'a> Get<$t> for Interp<'a> {
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
            StrToInt(ir, sr) => {
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
            Div(res, l, r) => {
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
            LenStr(res, s) => {
                res.accum(&mut f);
                s.accum(&mut f)
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
            LookupIntInt(res, arr, k) => {
                res.accum(&mut f);
                arr.accum(&mut f);
                k.accum(&mut f)
            }
            LookupIntStr(res, arr, k) => {
                res.accum(&mut f);
                arr.accum(&mut f);
                k.accum(&mut f)
            }
            LookupIntFloat(res, arr, k) => {
                res.accum(&mut f);
                arr.accum(&mut f);
                k.accum(&mut f)
            }
            LookupStrInt(res, arr, k) => {
                res.accum(&mut f);
                arr.accum(&mut f);
                k.accum(&mut f)
            }
            LookupStrStr(res, arr, k) => {
                res.accum(&mut f);
                arr.accum(&mut f);
                k.accum(&mut f)
            }
            LookupStrFloat(res, arr, k) => {
                res.accum(&mut f);
                arr.accum(&mut f);
                k.accum(&mut f)
            }
            ContainsIntInt(res, arr, k) => {
                res.accum(&mut f);
                arr.accum(&mut f);
                k.accum(&mut f)
            }
            ContainsIntStr(res, arr, k) => {
                res.accum(&mut f);
                arr.accum(&mut f);
                k.accum(&mut f)
            }
            ContainsIntFloat(res, arr, k) => {
                res.accum(&mut f);
                arr.accum(&mut f);
                k.accum(&mut f)
            }
            ContainsStrInt(res, arr, k) => {
                res.accum(&mut f);
                arr.accum(&mut f);
                k.accum(&mut f)
            }
            ContainsStrStr(res, arr, k) => {
                res.accum(&mut f);
                arr.accum(&mut f);
                k.accum(&mut f)
            }
            ContainsStrFloat(res, arr, k) => {
                res.accum(&mut f);
                arr.accum(&mut f);
                k.accum(&mut f)
            }
            DeleteIntInt(arr, k) => {
                arr.accum(&mut f);
                k.accum(&mut f)
            }
            DeleteIntFloat(arr, k) => {
                arr.accum(&mut f);
                k.accum(&mut f)
            }
            DeleteIntStr(arr, k) => {
                arr.accum(&mut f);
                k.accum(&mut f)
            }
            DeleteStrInt(arr, k) => {
                arr.accum(&mut f);
                k.accum(&mut f)
            }
            DeleteStrFloat(arr, k) => {
                arr.accum(&mut f);
                k.accum(&mut f)
            }
            DeleteStrStr(arr, k) => {
                arr.accum(&mut f);
                k.accum(&mut f)
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
            MovInt(dst, src) => {
                dst.accum(&mut f);
                src.accum(&mut f)
            }
            MovFloat(dst, src) => {
                dst.accum(&mut f);
                src.accum(&mut f)
            }
            MovStr(dst, src) => {
                dst.accum(&mut f);
                src.accum(&mut f)
            }
            MovMapIntInt(dst, src) => {
                dst.accum(&mut f);
                src.accum(&mut f)
            }
            MovMapIntFloat(dst, src) => {
                dst.accum(&mut f);
                src.accum(&mut f)
            }
            MovMapIntStr(dst, src) => {
                dst.accum(&mut f);
                src.accum(&mut f)
            }
            MovMapStrInt(dst, src) => {
                dst.accum(&mut f);
                src.accum(&mut f)
            }
            MovMapStrFloat(dst, src) => {
                dst.accum(&mut f);
                src.accum(&mut f)
            }
            MovMapStrStr(dst, src) => {
                dst.accum(&mut f);
                src.accum(&mut f)
            }
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
            Call(_) | Jmp(_) | Ret | Halt => {}
        }
    }
}
