use std::marker::PhantomData;

use crate::runtime::{self, Float, Int, Str};

#[derive(Copy, Clone)]
pub(crate) struct Label(u32);

pub(crate) struct Reg<T>(u32, PhantomData<*const T>);
impl<T> Clone for Reg<T> {
    fn clone(&self) -> Reg<T> {
        Reg(self.0, PhantomData)
    }
}
impl<T> Copy for Reg<T> {}

// TODO: figure out if we need nulls, and hence unions. That's another refactor, but not a hard
// one. Maybe look at MLSub for inspiration as well? (we wont need it to start)
// TODO: we will want a macro of some kind to eliminate some boilerplate. Play around with it some,
// but with a restricted set of instructions.
// TODO: implement runtime.
//   [x] * Strings (on the heap for now?)
//   [x] * Regexes (use rust syntax for now)
//   [ ] * Printf (skip for now?, see if we can use libc?)
//   [x] * Files
//          - Current plan:
//              - have a Bufreader in main thread: reads until current line separator, then calls
//                split for field separator and sets $0. (That's in the bytecode).
//              - Build up map from file name to output file ID. Send on channel to background thread
//                with file ID and payload. (but if we send the files over a channel, can we avoid
//                excessive allocations? I suppose allocations are the least of our worries if we are
//                also going to be writing output)
//   [x] * Conversions:
//          - Current plan: do pass with regex, then use simdjson (or stdlib). Benchmark with both.

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

    // Math
    AddInt(Reg<Int>, Reg<Int>, Reg<Int>),
    AddFloat(Reg<Float>, Reg<Float>, Reg<Float>),
    MulFloat(Reg<Float>, Reg<Float>, Reg<Float>),
    MulInt(Reg<Int>, Reg<Int>, Reg<Int>),
    DivFloat(Reg<Float>, Reg<Float>, Reg<Float>),
    DivInt(Reg<Float>, Reg<Int>, Reg<Int>),
    MinusFloat(Reg<Float>, Reg<Float>, Reg<Float>),
    MinusInt(Reg<Int>, Reg<Int>, Reg<Int>),
    ModFloat(Reg<Float>, Reg<Float>, Reg<Float>),
    ModInt(Reg<Int>, Reg<Int>, Reg<Int>),
    Not(Reg<Int>, Reg<Int>),
    NegInt(Reg<Int>, Reg<Int>),
    NegFloat(Reg<Float>, Reg<Float>),

    // String processing
    Concat(Reg<Str<'a>>, Reg<Str<'a>>, Reg<Str<'a>>),
    Match(Reg<Int>, Reg<Str<'a>>, Reg<Str<'a>>),

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
    IterBeginIntInt(Reg<runtime::Iter<Int>>, Reg<runtime::IntMap<Int>>),
    IterBeginIntStr(Reg<runtime::Iter<Int>>, Reg<runtime::IntMap<Str<'a>>>),
    IterBeginIntFloat(Reg<runtime::Iter<Int>>, Reg<runtime::IntMap<Float>>),
    IterBeginStrInt(Reg<runtime::Iter<Str<'a>>>, Reg<runtime::StrMap<'a, Int>>),
    IterBeginStrStr(
        Reg<runtime::Iter<Str<'a>>>,
        Reg<runtime::StrMap<'a, Str<'a>>>,
    ),
    IterBeginStrFloat(Reg<runtime::Iter<Str<'a>>>, Reg<runtime::StrMap<'a, Float>>),
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

    // Control
    JmpIf(Reg<Int>, Label),
    Jmp(Label),
    Halt,
}

impl<T> Reg<T> {
    pub(crate) fn new(i: u32) -> Self {
        Reg(i, PhantomData)
    }
    fn index(&self) -> usize {
        self.0 as usize
    }
}

pub(crate) struct Interp<'a> {
    instrs: Vec<Instr<'a>>,
    floats: Vec<Float>,
    ints: Vec<Int>,
    strs: Vec<Str<'a>>,

    // TODO: should these be smallvec<[T; 32]>?
    maps_int_float: Vec<runtime::IntMap<Float>>,
    maps_int_int: Vec<runtime::IntMap<Int>>,
    maps_int_str: Vec<runtime::IntMap<Str<'a>>>,

    maps_str_float: Vec<runtime::StrMap<'a, Float>>,
    maps_str_int: Vec<runtime::StrMap<'a, Int>>,
    maps_str_str: Vec<runtime::StrMap<'a, Str<'a>>>,

    iters_int: Vec<runtime::Iter<Int>>,
    iters_float: Vec<runtime::Iter<Float>>,
    iters_str: Vec<runtime::Iter<Str<'a>>>,
}

impl<'a> Interp<'a> {
    pub(crate) fn run(&mut self) {
        use Instr::*;
        let mut cur = 0;
        'outer: loop {
            // must end with Halt
            debug_assert!(cur < self.instrs.len());
            cur = loop {
                match unsafe { self.instrs.get_unchecked(cur) } {
                    StoreConstStr(sr, s) => {
                        let sr = *sr;
                        *self.get_mut(sr) = s.clone()
                    }
                    StoreConstInt(ir, i) => {
                        let ir = *ir;
                        *self.get_mut(ir) = *i
                    }
                    StoreConstFloat(fr, f) => {
                        let fr = *fr;
                        *self.get_mut(fr) = *f
                    }

                    IntToStr(sr, ir) => {
                        let s = runtime::convert::<_, Str>(*self.get(*ir));
                        let sr = *sr;
                        *self.get_mut(sr) = s;
                    }
                    FloatToStr(sr, fr) => {
                        let s = runtime::convert::<_, Str>(*self.get(*fr));
                        let sr = *sr;
                        *self.get_mut(sr) = s;
                    }
                    StrToInt(ir, sr) => {
                        let i = runtime::convert::<_, Int>(self.get(*sr));
                        let ir = *ir;
                        *self.get_mut(ir) = i;
                    }
                    StrToFloat(fr, sr) => {
                        let f = runtime::convert::<_, Float>(self.get(*sr));
                        let fr = *fr;
                        *self.get_mut(fr) = f;
                    }
                    FloatToInt(ir, fr) => {
                        let i = runtime::convert::<_, Int>(*self.get(*fr));
                        let ir = *ir;
                        *self.get_mut(ir) = i;
                    }
                    IntToFloat(fr, ir) => {
                        let f = runtime::convert::<_, Float>(*self.get(*ir));
                        let fr = *fr;
                        *self.get_mut(fr) = f;
                    }

                    AddInt(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = l + r;
                    }
                    AddFloat(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = l + r;
                    }
                    _ => unimplemented!(),
                };
                break cur + 1;
            };
        }
    }
}

trait Get<T> {
    fn get(&self, r: Reg<T>) -> &T;
    fn get_mut(&mut self, r: Reg<T>) -> &mut T;
}

macro_rules! impl_get {
    ($t:ty, $fld:ident) => {
        // TODO(ezr): test, then benchmark with get_unchecked()
        impl<'a> Get<$t> for Interp<'a> {
            fn get(&self, r: Reg<$t>) -> &$t {
                &self.$fld[r.index()]
            }
            fn get_mut(&mut self, r: Reg<$t>) -> &mut $t {
                &mut self.$fld[r.index()]
            }
        }
    };
}

impl_get!(Int, ints);
impl_get!(Str<'a>, strs);
impl_get!(Float, floats);
impl_get!(runtime::IntMap<Float>, maps_int_float);
impl_get!(runtime::IntMap<Int>, maps_int_int);
impl_get!(runtime::IntMap<Str<'a>>, maps_int_str);
impl_get!(runtime::StrMap<'a, Float>, maps_str_float);
impl_get!(runtime::StrMap<'a, Int>, maps_str_int);
impl_get!(runtime::StrMap<'a, Str<'a>>, maps_str_str);
impl_get!(runtime::Iter<Int>, iters_int);
impl_get!(runtime::Iter<Str<'a>>, iters_str);
impl_get!(runtime::Iter<Float>, iters_float);
