use std::marker::PhantomData;

use crate::builtins::Variable;
use crate::common::Result;
use crate::runtime::{self, Float, Int, LazyVec, Str};

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub(crate) struct Label(u32);

impl From<u32> for Label {
    fn from(u: u32) -> Label {
        Label(u)
    }
}

pub(crate) struct Reg<T>(u32, PhantomData<*const T>);

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
//   [ ] * HashMaps:
//          - For now, wrap hashbrown, but add in explicit handling for iteration to allow
//            insertions: wrap in refcell (needed anyway): add potential for stack of
//            modifications. (complex)
//          - Alternative: just clone the keys ahead of time. (issue: deletion? it actually looks
//            like gawk and mawk do this; deleting the key has it still show up in the iterator)
//          - Alternative: use and "ordmap" where keys actually index into a slice, etc...
//          - Decision: clone for now, but look at doing an optimization allowing for faster
//            iteration when there are no map accesses within the loop (also look at what others
//            are doing).
//   [x] * Line Splitting:
//          - Define trait for splitting lines, one that does splitting in larger batches, one that
//          does it line by line. Choose which one it is based on program behavior (do you set FS
//          after BEGIN?)
//          - Perhaps make this an enum and just add a Freeze instruction or
//          somethign?
//          - This may require adding instructions (ReadLineAndSplit?).
// Next: (1) finish interpreter (2) implement translator (3) implement parser (4) add
// functions/union type.

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

    // Lines
    HasLine(Reg<Int>, Reg<Str<'a>>),
    NextLine(Reg<Str<'a>>, Reg<Str<'a>>),

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

    // Print (TODO add more printing functions).
    Print(
        Reg<Str<'a>>, /*text*/
        Reg<Str<'a>>, /*output*/
        bool,         /*append*/
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
    vars: runtime::Variables<'a>,

    line: Str<'a>,
    split_line: LazyVec<Str<'a>>,
    regexes: runtime::RegexCache,
    write_files: runtime::FileWrite,
    read_files: runtime::FileRead,

    // TODO: should these be smallvec<[T; 32]>? We never add registers, so could we allocate one
    // contiguous region ahead of time?
    maps_int_float: Vec<runtime::IntMap<Float>>,
    maps_int_int: Vec<runtime::IntMap<Int>>,
    maps_int_str: Vec<runtime::IntMap<Str<'a>>>,

    maps_str_float: Vec<runtime::StrMap<'a, Float>>,
    maps_str_int: Vec<runtime::StrMap<'a, Int>>,
    maps_str_str: Vec<runtime::StrMap<'a, Str<'a>>>,

    iters_int: Vec<runtime::Iter<Int>>,
    iters_str: Vec<runtime::Iter<Str<'a>>>,
}

impl<'a> Interp<'a> {
    pub(crate) fn run(&mut self) -> Result<()> {
        use Instr::*;
        let mut cur = 0;
        'outer: loop {
            // must end with Halt
            cur = loop {
                debug_assert!(cur < self.instrs.len());
                use Variable::*;
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
                    MulInt(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = l * r;
                    }
                    MulFloat(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = l * r;
                    }
                    MinusInt(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = l - r;
                    }
                    MinusFloat(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = l - r;
                    }
                    ModInt(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = l % r;
                    }
                    ModFloat(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = l % r;
                    }
                    DivInt(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l) as Float;
                        let r = *self.get(*r) as Float;
                        *self.get_mut(res) = l / r;
                    }
                    DivFloat(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = l / r;
                    }
                    Not(res, ir) => {
                        let res = *res;
                        let i = *self.get(*ir);
                        *self.get_mut(res) = (i != 0) as Int;
                    }
                    NegInt(res, ir) => {
                        let res = *res;
                        let i = *self.get(*ir);
                        *self.get_mut(res) = -i;
                    }
                    NegFloat(res, fr) => {
                        let res = *res;
                        let f = *self.get(*fr);
                        *self.get_mut(res) = -f;
                    }
                    Concat(res, l, r) => {
                        let res = *res;
                        let l = self.get(*l).clone();
                        let r = self.get(*r).clone();
                        *self.get_mut(res) = Str::concat(l, r);
                    }
                    Match(res, l, r) => {
                        let res = *res;
                        let l = index(&self.strs, l);
                        let pat = index(&self.strs, r);
                        *self.get_mut(res) = self.regexes.match_regex(&pat, &l)? as Int;
                    }
                    LTFloat(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = (l < r) as Int;
                    }
                    LTInt(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = (l < r) as Int;
                    }
                    LTStr(res, l, r) => {
                        let res = *res;
                        let l = self.get(*l);
                        let r = self.get(*r);
                        *self.get_mut(res) = l.with_str(|l| r.with_str(|r| l < r)) as Int;
                    }
                    GTFloat(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = (l > r) as Int;
                    }
                    GTInt(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = (l > r) as Int;
                    }
                    GTStr(res, l, r) => {
                        let res = *res;
                        let l = self.get(*l);
                        let r = self.get(*r);
                        *self.get_mut(res) = l.with_str(|l| r.with_str(|r| l > r)) as Int;
                    }
                    LTEFloat(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = (l <= r) as Int;
                    }
                    LTEInt(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = (l <= r) as Int;
                    }
                    LTEStr(res, l, r) => {
                        let res = *res;
                        let l = self.get(*l);
                        let r = self.get(*r);
                        *self.get_mut(res) = l.with_str(|l| r.with_str(|r| l <= r)) as Int;
                    }
                    GTEFloat(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = (l >= r) as Int;
                    }
                    GTEInt(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = (l >= r) as Int;
                    }
                    GTEStr(res, l, r) => {
                        let res = *res;
                        let l = self.get(*l);
                        let r = self.get(*r);
                        *self.get_mut(res) = l.with_str(|l| r.with_str(|r| l >= r)) as Int;
                    }
                    EQFloat(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = (l == r) as Int;
                    }
                    EQInt(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = (l == r) as Int;
                    }
                    EQStr(res, l, r) => {
                        let res = *res;
                        let l = self.get(*l);
                        let r = self.get(*r);
                        *self.get_mut(res) = (l == r) as Int;
                    }
                    SetColumn(dst, src) => {
                        let col = *self.get(*dst);
                        if col < 0 {
                            return err!("attempt to access field {}", col);
                        }
                        if col == 0 {
                            self.split_line.clear();
                            self.line = self.get(*src).clone();
                            break cur + 1;
                        }
                        if self.split_line.len() == 0 {
                            self.regexes.split_regex(
                                &self.vars.fs,
                                &self.line,
                                &mut self.split_line,
                            )?;
                        }
                        self.split_line
                            .insert(col as usize - 1, self.get(*src).clone());
                    }
                    GetColumn(dst, src) => {
                        let col = *self.get(*src);
                        let dst = *dst;
                        if col < 0 {
                            return err!("attempt to access field {}", col);
                        }
                        if col == 0 {
                            let line = self.line.clone();
                            *self.get_mut(dst) = line;
                            break cur + 1;
                        }
                        if self.split_line.len() == 0 {
                            self.regexes.split_regex(
                                &self.vars.fs,
                                &self.line,
                                &mut self.split_line,
                            )?;
                        }
                        let res = self
                            .split_line
                            .get(col as usize - 1)
                            .unwrap_or_else(Default::default);
                        *self.get_mut(dst) = res;
                    }
                    SplitInt(flds, to_split, arr, pat) => {
                        // Index manually here to defeat the borrow checker.
                        let to_split = index(&self.strs, to_split);
                        let arr = index(&self.maps_int_str, arr);
                        let pat = index(&self.strs, pat);
                        let old_len = arr.len();
                        self.regexes.split_regex_intmap(&pat, &to_split, &arr)?;
                        let res = (arr.len() - old_len) as i64;
                        let flds = *flds;
                        *self.get_mut(flds) = res;
                    }
                    SplitStr(flds, to_split, arr, pat) => {
                        // Very similar to above
                        let to_split = index(&self.strs, to_split);
                        let arr = index(&self.maps_str_str, arr);
                        let pat = index(&self.strs, pat);
                        let old_len = arr.len();
                        self.regexes.split_regex_strmap(&pat, &to_split, &arr)?;
                        let res = (arr.len() - old_len) as i64;
                        let flds = *flds;
                        *self.get_mut(flds) = res;
                    }
                    Print(txt, out, append) => {
                        let txt = index(&self.strs, txt);
                        let out = index(&self.strs, out);
                        self.write_files.write_str(out, txt, *append)?;
                    }
                    LookupIntInt(res, arr, k) => {
                        let arr = index(&self.maps_int_int, arr);
                        let k = index(&self.ints, k);
                        let v = arr.get(k).unwrap_or(0);
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    LookupIntStr(res, arr, k) => {
                        let arr = index(&self.maps_int_str, arr);
                        let k = index(&self.ints, k);
                        let v = arr.get(k).unwrap_or_else(Default::default);
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    LookupIntFloat(res, arr, k) => {
                        let arr = index(&self.maps_int_float, arr);
                        let k = index(&self.ints, k);
                        let v = arr.get(k).unwrap_or(0.0);
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    LookupStrInt(res, arr, k) => {
                        let arr = index(&self.maps_str_int, arr);
                        let k = index(&self.strs, k);
                        let v = arr.get(k).unwrap_or(0);
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    LookupStrStr(res, arr, k) => {
                        let arr = index(&self.maps_str_str, arr);
                        let k = index(&self.strs, k);
                        let v = arr.get(k).unwrap_or_else(Default::default);
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    LookupStrFloat(res, arr, k) => {
                        let arr = index(&self.maps_str_float, arr);
                        let k = index(&self.strs, k);
                        let v = arr.get(k).unwrap_or(0.0);
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    ContainsIntInt(res, arr, k) => {
                        let arr = index(&self.maps_int_int, arr);
                        let k = index(&self.ints, k);
                        let v = arr.get(k).is_some() as i64;
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    ContainsIntStr(res, arr, k) => {
                        let arr = index(&self.maps_int_str, arr);
                        let k = index(&self.ints, k);
                        let v = arr.get(k).is_some() as i64;
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    ContainsIntFloat(res, arr, k) => {
                        let arr = index(&self.maps_int_float, arr);
                        let k = index(&self.ints, k);
                        let v = arr.get(k).is_some() as i64;
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    ContainsStrInt(res, arr, k) => {
                        let arr = index(&self.maps_str_int, arr);
                        let k = index(&self.strs, k);
                        let v = arr.get(k).is_some() as i64;
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    ContainsStrStr(res, arr, k) => {
                        let arr = index(&self.maps_str_str, arr);
                        let k = index(&self.strs, k);
                        let v = arr.get(k).is_some() as i64;
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    ContainsStrFloat(res, arr, k) => {
                        let arr = index(&self.maps_str_float, arr);
                        let k = index(&self.strs, k);
                        let v = arr.get(k).is_some() as i64;
                        let res = *res;
                        *self.get_mut(res) = v;
                    }
                    StoreIntInt(arr, k, v) => {
                        let arr = index(&self.maps_int_int, arr);
                        let k = index(&self.ints, k).clone();
                        let v = index(&self.ints, v).clone();
                        arr.insert(k, v);
                    }
                    StoreIntFloat(arr, k, v) => {
                        let arr = index(&self.maps_int_float, arr);
                        let k = index(&self.ints, k).clone();
                        let v = index(&self.floats, v).clone();
                        arr.insert(k, v);
                    }
                    StoreIntStr(arr, k, v) => {
                        let arr = index(&self.maps_int_str, arr);
                        let k = index(&self.ints, k).clone();
                        let v = index(&self.strs, v).clone();
                        arr.insert(k, v);
                    }
                    StoreStrInt(arr, k, v) => {
                        let arr = index(&self.maps_str_int, arr);
                        let k = index(&self.strs, k).clone();
                        let v = index(&self.ints, v).clone();
                        arr.insert(k, v);
                    }
                    StoreStrFloat(arr, k, v) => {
                        let arr = index(&self.maps_str_float, arr);
                        let k = index(&self.strs, k).clone();
                        let v = index(&self.floats, v).clone();
                        arr.insert(k, v);
                    }
                    StoreStrStr(arr, k, v) => {
                        let arr = index(&self.maps_str_str, arr);
                        let k = index(&self.strs, k).clone();
                        let v = index(&self.strs, v).clone();
                        arr.insert(k, v);
                    }
                    LoadVarStr(dst, var) => {
                        let s = match var {
                            FS => self.vars.fs.clone(),
                            RS => self.vars.rs.clone(),
                            FILENAME => self.vars.filename.clone(),
                            ARGC | ARGV | NF | NR => unreachable!(),
                        };
                        let dst = *dst;
                        *self.get_mut(dst) = s;
                    }
                    StoreVarStr(var, src) => {
                        let src = *src;
                        let s = self.get(src).clone();
                        match var {
                            FS => self.vars.fs = s,
                            RS => self.vars.rs = s,
                            FILENAME => self.vars.filename = s,
                            ARGC | ARGV | NF | NR => unreachable!(),
                        };
                    }
                    LoadVarInt(dst, var) => {
                        let i = match var {
                            ARGC => self.vars.argc,
                            NF => self.vars.nf,
                            NR => self.vars.nr,
                            FS | RS | FILENAME | ARGV => unreachable!(),
                        };
                        let dst = *dst;
                        *self.get_mut(dst) = i;
                    }
                    StoreVarInt(var, src) => {
                        let src = *src;
                        let s = *self.get(src);
                        match var {
                            ARGC => self.vars.argc = s,
                            NF => self.vars.nf = s,
                            NR => self.vars.nr = s,
                            FS | RS | FILENAME | ARGV => unreachable!(),
                        };
                    }
                    LoadVarIntMap(dst, var) => {
                        let arr = match var {
                            ARGV => self.vars.argv.clone(),
                            ARGC | NF | NR | FS | RS | FILENAME => unreachable!(),
                        };
                        let dst = *dst;
                        *self.get_mut(dst) = arr;
                    }
                    StoreVarIntMap(var, src) => {
                        let src = *src;
                        let s = self.get(src).clone();
                        match var {
                            ARGV => self.vars.argv = s,
                            ARGC | NF | NR | FS | RS | FILENAME => unreachable!(),
                        };
                    }
                    IterBeginIntInt(dst, arr) => {
                        let arr = *arr;
                        let iter = self.get(arr).to_iter();
                        let dst = *dst;
                        *self.get_mut(dst) = iter;
                    }
                    IterBeginIntFloat(dst, arr) => {
                        let arr = *arr;
                        let iter = self.get(arr).to_iter();
                        let dst = *dst;
                        *self.get_mut(dst) = iter;
                    }
                    IterBeginIntStr(dst, arr) => {
                        let arr = *arr;
                        let iter = self.get(arr).to_iter();
                        let dst = *dst;
                        *self.get_mut(dst) = iter;
                    }
                    IterBeginStrInt(dst, arr) => {
                        let arr = *arr;
                        let iter = self.get(arr).to_iter();
                        let dst = *dst;
                        *self.get_mut(dst) = iter;
                    }
                    IterBeginStrFloat(dst, arr) => {
                        let arr = *arr;
                        let iter = self.get(arr).to_iter();
                        let dst = *dst;
                        *self.get_mut(dst) = iter;
                    }
                    IterBeginStrStr(dst, arr) => {
                        let arr = *arr;
                        let iter = self.get(arr).to_iter();
                        let dst = *dst;
                        *self.get_mut(dst) = iter;
                    }
                    IterHasNextInt(dst, iter) => {
                        let res = self.get(*iter).has_next() as Int;
                        let dst = *dst;
                        *self.get_mut(dst) = res;
                    }
                    IterHasNextStr(dst, iter) => {
                        let res = self.get(*iter).has_next() as Int;
                        let dst = *dst;
                        *self.get_mut(dst) = res;
                    }
                    IterGetNextInt(dst, iter) => {
                        let res = unsafe { self.get(*iter).get_next().clone() };
                        let dst = *dst;
                        *self.get_mut(dst) = res;
                    }
                    IterGetNextStr(dst, iter) => {
                        let res = unsafe { self.get(*iter).get_next().clone() };
                        let dst = *dst;
                        *self.get_mut(dst) = res;
                    }
                    MovInt(dst, src) => {
                        let src = *src;
                        let src_contents = self.get(src).clone();
                        let dst = *dst;
                        *self.get_mut(dst) = src_contents;
                    }
                    MovFloat(dst, src) => {
                        let src = *src;
                        let src_contents = self.get(src).clone();
                        let dst = *dst;
                        *self.get_mut(dst) = src_contents;
                    }
                    MovStr(dst, src) => {
                        let src = *src;
                        let src_contents = self.get(src).clone();
                        let dst = *dst;
                        *self.get_mut(dst) = src_contents;
                    }
                    MovMapIntInt(dst, src) => {
                        let src = *src;
                        let src_contents = self.get(src).clone();
                        let dst = *dst;
                        *self.get_mut(dst) = src_contents;
                    }
                    MovMapIntFloat(dst, src) => {
                        let src = *src;
                        let src_contents = self.get(src).clone();
                        let dst = *dst;
                        *self.get_mut(dst) = src_contents;
                    }
                    MovMapIntStr(dst, src) => {
                        let src = *src;
                        let src_contents = self.get(src).clone();
                        let dst = *dst;
                        *self.get_mut(dst) = src_contents;
                    }
                    MovMapStrInt(dst, src) => {
                        let src = *src;
                        let src_contents = self.get(src).clone();
                        let dst = *dst;
                        *self.get_mut(dst) = src_contents;
                    }
                    MovMapStrFloat(dst, src) => {
                        let src = *src;
                        let src_contents = self.get(src).clone();
                        let dst = *dst;
                        *self.get_mut(dst) = src_contents;
                    }
                    MovMapStrStr(dst, src) => {
                        let src = *src;
                        let src_contents = self.get(src).clone();
                        let dst = *dst;
                        *self.get_mut(dst) = src_contents;
                    }
                    // TODO add error logging for these errors perhaps?
                    HasLine(dst, file) => {
                        let dst = *dst;
                        let file = index(&self.strs, file);
                        match self.read_files.has_line(file) {
                            Ok(b) => *self.get_mut(dst) = b as Int,
                            Err(_) => *self.get_mut(dst) = 0,
                        };
                    }
                    NextLine(dst, file) => {
                        let dst = *dst;
                        let file = index(&self.strs, file);
                        match self
                            .regexes
                            .get_line(file, &self.vars.fs, &mut self.read_files)
                        {
                            Ok(l) => *self.get_mut(dst) = l,
                            Err(_) => *self.get_mut(dst) = "".into(),
                        };
                    }
                    JmpIf(cond, lbl) => {
                        let cond = *cond;
                        if *self.get(cond) != 0 {
                            break lbl.0 as usize;
                        }
                    }
                    Jmp(lbl) => {
                        break lbl.0 as usize;
                    }
                    Halt => break 'outer Ok(()),
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

#[inline]
fn index<'a, T>(v: &'a Vec<T>, reg: &Reg<T>) -> &'a T {
    &v[reg.index()]
}
#[inline]
fn index_mut<'a, T>(v: &'a mut Vec<T>, reg: &Reg<T>) -> &'a mut T {
    &mut v[reg.index()]
}

macro_rules! impl_get {
    ($t:ty, $fld:ident) => {
        // TODO(ezr): test, then benchmark with get_unchecked()
        impl<'a> Get<$t> for Interp<'a> {
            fn get(&self, r: Reg<$t>) -> &$t {
                index(&self.$fld, &r)
            }
            fn get_mut(&mut self, r: Reg<$t>) -> &mut $t {
                index_mut(&mut self.$fld, &r)
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
