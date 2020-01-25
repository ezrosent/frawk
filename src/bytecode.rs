use std::marker::PhantomData;

use crate::builtins::Variable;
use crate::common::{NumTy, Result};
use crate::compile;
use crate::hashbrown::HashSet;
use crate::runtime::{self, Float, Int, LazyVec, Str};

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub(crate) struct Label(pub u32);

impl std::fmt::Debug for Label {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "@{}", self.0)
    }
}

impl From<u32> for Label {
    fn from(u: u32) -> Label {
        Label(u)
    }
}

impl From<usize> for Label {
    fn from(u: usize) -> Label {
        Label(u as u32)
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
//   [x] * HashMaps:
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

#[derive(Debug)]
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
    // TODO need NotStr
    NegInt(Reg<Int>, Reg<Int>),
    NegFloat(Reg<Float>, Reg<Float>),

    // String processing
    Concat(Reg<Str<'a>>, Reg<Str<'a>>, Reg<Str<'a>>),
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

    // Lines
    // perhaps we need
    // Getline(Reg<Int> /* -1,0,1 */ , Reg<Str<'a>> /* output line */, Reg<Str<'a>> /* input file */)
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
    PrintStdout(Reg<Str<'a>> /*text*/),
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
    fn index(&self) -> usize {
        self.0 as usize
    }
}

#[derive(Default)]
struct Storage<T> {
    regs: Vec<T>,
    stack: Vec<T>,
}

// TODO: Want a Vec<Vec<Instr>> indexed by function.
// TODO: Can we use the Rust stack to do calls? We should probably have a stack of (function index,
// instr index) to store the continuation. That'll make tail calls easier later on if we want to
// implement them.
// TODO: We probably want return to take an (optional) operand, for more alignment with LLVM

pub(crate) struct Interp<'a> {
    // index of `instrs` that contains "main"
    main_func: usize,
    instrs: Vec<Vec<Instr<'a>>>,
    stack: Vec<(usize /*function*/, Label /*instr*/)>,

    vars: runtime::Variables<'a>,

    line: Str<'a>,
    split_line: LazyVec<Str<'a>>,
    regexes: runtime::RegexCache,
    write_files: runtime::FileWrite,
    read_files: runtime::FileRead,

    // TODO: should these be smallvec<[T; 32]>? We never add registers, so could we allocate one
    // contiguous region ahead of time?
    floats: Storage<Float>,
    ints: Storage<Int>,
    strs: Storage<Str<'a>>,
    maps_int_float: Storage<runtime::IntMap<Float>>,
    maps_int_int: Storage<runtime::IntMap<Int>>,
    maps_int_str: Storage<runtime::IntMap<Str<'a>>>,

    maps_str_float: Storage<runtime::StrMap<'a, Float>>,
    maps_str_int: Storage<runtime::StrMap<'a, Int>>,
    maps_str_str: Storage<runtime::StrMap<'a, Str<'a>>>,

    iters_int: Storage<runtime::Iter<Int>>,
    iters_str: Storage<runtime::Iter<Str<'a>>>,
}

fn default_of<T: Default>(n: usize) -> Storage<T> {
    let mut regs = Vec::new();
    regs.resize_with(n, Default::default);
    Storage {
        regs,
        stack: Default::default(),
    }
}

impl<'a> Interp<'a> {
    pub(crate) fn instrs(&self) -> &Vec<Vec<Instr<'a>>> {
        &self.instrs
    }
    pub(crate) fn new(
        instrs: Vec<Vec<Instr<'a>>>,
        main_func: usize,
        regs: impl Fn(compile::Ty) -> usize,
        stdin: impl std::io::Read + 'static,
        stdout: impl std::io::Write + 'static,
    ) -> Interp<'a> {
        use compile::Ty::*;
        Interp {
            main_func,
            instrs,
            stack: Default::default(),
            floats: default_of(regs(Float)),
            ints: default_of(regs(Int)),
            strs: default_of(regs(Str)),
            vars: Default::default(),

            line: "".into(),
            split_line: LazyVec::new(),
            regexes: Default::default(),
            write_files: runtime::FileWrite::new(stdout),
            read_files: runtime::FileRead::new(stdin),

            maps_int_float: default_of(regs(MapIntFloat)),
            maps_int_int: default_of(regs(MapIntInt)),
            maps_int_str: default_of(regs(MapIntStr)),

            maps_str_float: default_of(regs(MapStrFloat)),
            maps_str_int: default_of(regs(MapStrInt)),
            maps_str_str: default_of(regs(MapStrStr)),

            iters_int: default_of(regs(IterInt)),
            iters_str: default_of(regs(IterStr)),
        }
    }
    pub(crate) fn run(&mut self) -> Result<()> {
        use Instr::*;
        let newline: Str = "\n".into();
        // We are only accessing one vector at a time here, but it's hard to convince the borrow
        // checker of this fact, so we access the vectors through raw pointers.
        let mut cur_fn = self.main_func;
        let mut instrs = (&mut self.instrs[cur_fn]) as *mut Vec<Instr<'a>>;
        let mut cur = 0;
        'outer: loop {
            // must end with Halt
            cur = loop {
                debug_assert!(cur < unsafe { (*instrs).len() });
                use Variable::*;
                match unsafe { (*instrs).get_unchecked(cur) } {
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
                    Div(res, l, r) => {
                        let res = *res;
                        let l = *self.get(*l);
                        let r = *self.get(*r);
                        *self.get_mut(res) = l / r;
                    }
                    Not(res, ir) => {
                        let res = *res;
                        let i = *self.get(*ir);
                        *self.get_mut(res) = (i == 0) as Int;
                    }
                    NotStr(res, sr) => {
                        let res = *res;
                        let sr = *sr;
                        let is_empty = self.get(sr).with_str(|s| s.len() == 0);
                        *self.get_mut(res) = is_empty as Int;
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
                    LenStr(res, s) => {
                        let res = *res;
                        let s = *s;
                        // TODO consider doing a with_str here or enforce elsewhere that strings
                        // cannot exceed u32::max.
                        let len = self.get(s).len();
                        *self.get_mut(res) = len as Int;
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
                            self.vars.nf = -1;
                            break cur + 1;
                        }
                        if self.split_line.len() == 0 {
                            self.regexes.split_regex(
                                &self.vars.fs,
                                &self.line,
                                &mut self.split_line,
                            )?;
                            self.vars.nf = self.split_line.len() as Int;
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
                            self.vars.nf = self.split_line.len() as Int;
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
                    PrintStdout(txt) => {
                        let txt = index(&self.strs, txt);
                        self.write_files.write_str_stdout(txt)?;
                        self.write_files.write_str_stdout(&newline)?;
                    }
                    Print(txt, out, append) => {
                        let txt = index(&self.strs, txt);
                        let out = index(&self.strs, out);
                        self.write_files.write_line(out, txt, *append)?;
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
                    DeleteIntInt(arr, k) => {
                        let arr = index(&self.maps_int_int, arr);
                        let k = index(&self.ints, k);
                        arr.delete(k);
                    }
                    DeleteIntFloat(arr, k) => {
                        let arr = index(&self.maps_int_float, arr);
                        let k = index(&self.ints, k);
                        arr.delete(k);
                    }
                    DeleteIntStr(arr, k) => {
                        let arr = index(&self.maps_int_str, arr);
                        let k = index(&self.ints, k);
                        arr.delete(k);
                    }
                    DeleteStrInt(arr, k) => {
                        let arr = index(&self.maps_str_int, arr);
                        let k = index(&self.strs, k);
                        arr.delete(k);
                    }
                    DeleteStrFloat(arr, k) => {
                        let arr = index(&self.maps_str_float, arr);
                        let k = index(&self.strs, k);
                        arr.delete(k);
                    }
                    DeleteStrStr(arr, k) => {
                        let arr = index(&self.maps_str_str, arr);
                        let k = index(&self.strs, k);
                        arr.delete(k);
                    }
                    LenIntInt(res, arr) => {
                        let arr = *arr;
                        let len = self.get(arr).len();
                        let res = *res;
                        *self.get_mut(res) = len as Int;
                    }
                    LenIntFloat(res, arr) => {
                        let arr = *arr;
                        let len = self.get(arr).len();
                        let res = *res;
                        *self.get_mut(res) = len as Int;
                    }
                    LenIntStr(res, arr) => {
                        let arr = *arr;
                        let len = self.get(arr).len();
                        let res = *res;
                        *self.get_mut(res) = len as Int;
                    }
                    LenStrInt(res, arr) => {
                        let arr = *arr;
                        let len = self.get(arr).len();
                        let res = *res;
                        *self.get_mut(res) = len as Int;
                    }
                    LenStrFloat(res, arr) => {
                        let arr = *arr;
                        let len = self.get(arr).len();
                        let res = *res;
                        *self.get_mut(res) = len as Int;
                    }
                    LenStrStr(res, arr) => {
                        let arr = *arr;
                        let len = self.get(arr).len();
                        let res = *res;
                        *self.get_mut(res) = len as Int;
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
                            OFS => self.vars.ofs.clone(),
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
                            OFS => self.vars.ofs = s,
                            RS => self.vars.rs = s,
                            FILENAME => self.vars.filename = s,
                            ARGC | ARGV | NF | NR => unreachable!(),
                        };
                    }
                    LoadVarInt(dst, var) => {
                        let i = match var {
                            ARGC => self.vars.argc,
                            NF => {
                                if self.split_line.len() == 0 {
                                    self.regexes.split_regex(
                                        &self.vars.fs,
                                        &self.line,
                                        &mut self.split_line,
                                    )?;
                                    self.vars.nf = self.split_line.len() as Int;
                                }
                                self.vars.nf
                            }
                            NR => self.vars.nr,
                            OFS | FS | RS | FILENAME | ARGV => unreachable!(),
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
                            OFS | FS | RS | FILENAME | ARGV => unreachable!(),
                        };
                    }
                    LoadVarIntMap(dst, var) => {
                        let arr = match var {
                            ARGV => self.vars.argv.clone(),
                            OFS | ARGC | NF | NR | FS | RS | FILENAME => unreachable!(),
                        };
                        let dst = *dst;
                        *self.get_mut(dst) = arr;
                    }
                    StoreVarIntMap(var, src) => {
                        let src = *src;
                        let s = self.get(src).clone();
                        match var {
                            ARGV => self.vars.argv = s,
                            OFS | ARGC | NF | NR | FS | RS | FILENAME => unreachable!(),
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
                    ReadErr(dst, file) => {
                        let dst = *dst;
                        let file = index(&self.strs, file);
                        let res = self.read_files.read_err(file)?;
                        *self.get_mut(dst) = res;
                    }
                    NextLine(dst, file) => {
                        let dst = *dst;
                        let file = index(&self.strs, file);
                        match self
                            .regexes
                            .get_line(file, &self.vars.rs, &mut self.read_files)
                        {
                            Ok(l) => *self.get_mut(dst) = l,
                            Err(_) => *self.get_mut(dst) = "".into(),
                        };
                    }
                    ReadErrStdin(dst) => {
                        let dst = *dst;
                        let res = self.read_files.read_err_stdin();
                        *self.get_mut(dst) = res;
                    }
                    NextLineStdin(dst) => {
                        let dst = *dst;
                        let res = self
                            .regexes
                            .get_line_stdin(&self.vars.rs, &mut self.read_files)?;
                        *self.get_mut(dst) = res;
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
                    PushInt(reg) => {
                        let reg = *reg;
                        self.push(reg)
                    }
                    PushFloat(reg) => {
                        let reg = *reg;
                        self.push(reg)
                    }
                    PushStr(reg) => {
                        let reg = *reg;
                        self.push(reg)
                    }
                    PushIntInt(reg) => {
                        let reg = *reg;
                        self.push(reg)
                    }
                    PushIntFloat(reg) => {
                        let reg = *reg;
                        self.push(reg)
                    }
                    PushIntStr(reg) => {
                        let reg = *reg;
                        self.push(reg)
                    }
                    PushStrInt(reg) => {
                        let reg = *reg;
                        self.push(reg)
                    }
                    PushStrFloat(reg) => {
                        let reg = *reg;
                        self.push(reg)
                    }
                    PushStrStr(reg) => {
                        let reg = *reg;
                        self.push(reg)
                    }

                    PopInt(reg) => {
                        let reg = *reg;
                        self.pop(reg)
                    }
                    PopFloat(reg) => {
                        let reg = *reg;
                        self.pop(reg)
                    }
                    PopStr(reg) => {
                        let reg = *reg;
                        self.pop(reg)
                    }
                    PopIntInt(reg) => {
                        let reg = *reg;
                        self.pop(reg)
                    }
                    PopIntFloat(reg) => {
                        let reg = *reg;
                        self.pop(reg)
                    }
                    PopIntStr(reg) => {
                        let reg = *reg;
                        self.pop(reg)
                    }
                    PopStrInt(reg) => {
                        let reg = *reg;
                        self.pop(reg)
                    }
                    PopStrFloat(reg) => {
                        let reg = *reg;
                        self.pop(reg)
                    }
                    PopStrStr(reg) => {
                        let reg = *reg;
                        self.pop(reg)
                    }
                    Call(func) => {
                        self.stack.push((cur_fn, Label(cur as u32 + 1)));
                        cur_fn = *func;
                        instrs = &mut self.instrs[*func];
                        break 0;
                    }
                    Ret => {
                        if let Some((func, Label(inst))) = self.stack.pop() {
                            cur_fn = func;
                            instrs = &mut self.instrs[func];
                            break inst as usize;
                        } else {
                            break 'outer Ok(());
                        }
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
trait Pop<T> {
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

// TODO: Add a pass that does checking of indexes once.
// That could justify no checking during interpretation.
const CHECKED: bool = false;

#[inline]
fn index<'a, T>(Storage { regs, .. }: &'a Storage<T>, reg: &Reg<T>) -> &'a T {
    if CHECKED {
        &regs[reg.index()]
    } else {
        debug_assert!(reg.index() < regs.len());
        unsafe { regs.get_unchecked(reg.index()) }
    }
}

#[inline]
fn index_mut<'a, T>(Storage { regs, .. }: &'a mut Storage<T>, reg: &Reg<T>) -> &'a mut T {
    if CHECKED {
        &mut regs[reg.index()]
    } else {
        debug_assert!(reg.index() < regs.len());
        unsafe { regs.get_unchecked_mut(reg.index()) }
    }
}

#[inline]
fn push<'a, T: Clone>(s: &'a mut Storage<T>, reg: &Reg<T>) {
    let v = index(s, reg).clone();
    s.stack.push(v);
}

#[inline]
fn pop<'a, T: Clone>(s: &'a mut Storage<T>) -> T {
    s.stack.pop().expect("pop must be called on nonempty stack")
}

// For accumulating register-specific metadata
pub(crate) trait Accum {
    fn accum(&self, f: impl FnMut(NumTy, compile::Ty));
}

macro_rules! impl_pop {
    ($t:ty, $fld:ident) => {
        impl<'a> Pop<$t> for Interp<'a> {
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
            fn accum(&self, mut f: impl FnMut(NumTy, compile::Ty)) {
                f(self.index() as NumTy, compile::Ty::$ty);
            }
        }
    };
    ($t:ty, $ty:tt,) => {
        impl Accum for Reg<$t> {
            fn accum(&self, mut f: impl FnMut(NumTy, compile::Ty)) {
                f(self.index() as NumTy, compile::Ty::$ty);
            }
        }
    };
}

macro_rules! impl_get {
    ($t:ty, $fld:ident, $ty:tt $(,$lt:tt)*) => {
        impl_accum!($t, $ty, $($lt),*);
        impl<'a> Get<$t> for Interp<'a> {
            fn get(&self, r: Reg<$t>) -> &$t {
                #[cfg(debug_assertions)]
                _dbg_check_index(
                    concat!(stringify!($t), "_", stringify!($fld)),
                    &self.$fld,
                    r.index(),
                );
                index(&self.$fld, &r)
            }
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
            Not(res, ir) => unimplemented!(),
            NotStr(res, sr) => unimplemented!(),
            NegInt(res, ir) => unimplemented!(),
            NegFloat(res, fr) => unimplemented!(),
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
            LenStr(res, s) => unimplemented!(),
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
            PrintStdout(txt) => txt.accum(&mut f),
            Print(txt, out, _append) => {
                txt.accum(&mut f);
                out.accum(&mut f)
            }
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

// Used in benchmarking code.

#[cfg(test)]
impl<T: Default> Storage<T> {
    fn reset(&mut self) {
        self.stack.clear();
        for i in self.regs.iter_mut() {
            *i = Default::default();
        }
    }
}

#[cfg(test)]
impl<'a> Interp<'a> {
    pub(crate) fn reset(&mut self) {
        self.stack = Default::default();
        self.vars = Default::default();
        self.line = "".into();
        self.split_line = LazyVec::new();
        self.regexes = Default::default();
        self.floats.reset();
        self.ints.reset();
        self.strs.reset();
        self.maps_int_int.reset();
        self.maps_int_float.reset();
        self.maps_int_str.reset();
        self.maps_str_int.reset();
        self.maps_str_float.reset();
        self.maps_str_str.reset();
        self.iters_int.reset();
        self.iters_str.reset();
    }
}
