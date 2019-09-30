use std::marker::PhantomData;
mod runtime {
    use smallvec::SmallVec;
    use std::cell::RefCell;
    use std::marker::PhantomData;
    use std::rc::Rc;
    pub(crate) trait Scalar {}
    impl Scalar for Int {}
    impl Scalar for Float {}
    impl<'a> Scalar for Str<'a> {}

    #[derive(Clone, Debug)]
    enum Inner<'a> {
        Literal(&'a str),
        Boxed(Rc<str>),
        // TODO: consider making it Branch {...Rc<Str<'a>>...}
        Concat(Rc<Branch<'a>>),
    }

    #[derive(Clone, Debug)]
    struct Branch<'a> {
        len: u32,
        left: Str<'a>,
        right: Str<'a>,
    }

    #[derive(Clone, Debug)]
    pub(crate) struct Str<'a>(RefCell<Inner<'a>>);

    impl<'a> Str<'a> {
        fn len_u32(&self) -> u32 {
            use Inner::*;
            match &*self.0.borrow() {
                Literal(s) => conv_len(s.len()),
                Boxed(s) => conv_len(s.len()),
                Concat(b) => b.len,
            }
        }
        pub(crate) fn len(&self) -> usize {
            use Inner::*;
            match &*self.0.borrow() {
                Literal(s) => s.len(),
                Boxed(s) => s.len(),
                Concat(b) => b.len as usize,
            }
        }
    }

    impl<'a> PartialEq for Str<'a> {
        fn eq(&self, other: &Str<'a>) -> bool {
            use Inner::*;
            if self.len() != other.len() {
                return false;
            }
            match (&*self.0.borrow(), &*other.0.borrow()) {
                (Literal(s1), Literal(s2)) => return s1 == s2,
                (Boxed(s1), Boxed(s2)) => return s1 == s2,
                (Literal(r), Boxed(b)) | (Boxed(b), Literal(r)) => return *r == &**b,
                (_, _) => {}
            }
            self.force();
            other.force();
            self == other
        }
    }

    fn conv_len(l: usize) -> u32 {
        if l > (u32::max_value() as usize) {
            u32::max_value()
        } else {
            l as u32
        }
    }

    impl<'a> Eq for Str<'a> {}

    impl<'a> From<&'a str> for Str<'a> {
        fn from(s: &'a str) -> Str<'a> {
            Str(RefCell::new(Inner::Literal(s)))
        }
    }
    impl<'a> From<String> for Str<'a> {
        fn from(s: String) -> Str<'a> {
            Str(RefCell::new(Inner::Boxed(s.into())))
        }
    }

    impl<'a> Str<'a> {
        pub(crate) fn with_str(&self, f: impl FnOnce(&str)) {
            self.force();
            match &*self.0.borrow() {
                Inner::Literal(l) => f(l),
                Inner::Boxed(b) => f(&*b),
                _ => unreachable!(),
            }
        }
        pub(crate) fn concat(s1: Str<'a>, s2: Str<'a>) -> Self {
            Str(RefCell::new(Inner::Concat(Rc::new(Branch {
                len: s1.len_u32().saturating_add(s2.len_u32()),
                left: s1,
                right: s2,
            }))))
        }
        /// force flattens the string by concatenating all components into a single boxed string.
        fn force(&self) {
            use Inner::*;
            if let Literal(_) | Boxed(_) = &*self.0.borrow() {
                return;
            }
            let mut cur = self.clone();
            let mut res = String::with_capacity(self.len());
            let mut todos = SmallVec::<[Str<'a>; 16]>::new();
            loop {
                cur = loop {
                    match &*cur.0.borrow() {
                        Literal(s) => res.push_str(s),
                        Boxed(s) => res.push_str(&*s),
                        Concat(rc) => {
                            todos.push(rc.right.clone());
                            break rc.left.clone();
                        }
                    }
                    if let Some(c) = todos.pop() {
                        break c;
                    }
                    self.0.replace(Boxed(res.into()));
                    return;
                };
            }
        }
    }

    #[cfg(test)]
    mod string_tests {
        use super::*;
        #[test]
        fn concat_test() {
            let s1 = Str::from("hi there fellow");
            let s2 = Str::concat(
                Str::concat(Str::from("hi"), Str::from(String::from(" there"))),
                Str::concat(
                    Str::from(" "),
                    Str::concat(Str::from("fel"), Str::from("low")),
                ),
            );
            assert_eq!(s1, s2);
            assert!(s1 != Str::concat(s1.clone(), s2));
        }
    }

    pub(crate) type Int = i64;
    pub(crate) type Float = f64;
    pub(crate) type IntMap<V> = hashbrown::HashMap<Int, V>;
    pub(crate) type StrMap<'a, V> = hashbrown::HashMap<Str<'a>, V>;
    pub(crate) struct Iter<S: Scalar>(PhantomData<*const S>);
}

use runtime::{Float, Int, Str};

#[derive(Copy, Clone)]
pub(crate) struct Label(u32);

#[derive(Copy, Clone)]
pub(crate) struct Reg<T>(u32, PhantomData<*const T>);

// TODO: figure out if we need nulls, and hence unions. That's another refactor, but not a hard
// one. Maybe look at MLSub for inspiration as well? (we wont need it to start)
// TODO: we will want a macro of some kind to eliminate some boilerplate. Play around with it some,
// but with a restricted set of instructions.
// TODO: implement runtime.
//   * Strings (on the heap for now?)
//   * Regexes (use rust syntax for now)
//   * Printf (skip for now?, see if we can use libc?)
//   * Files
//      - Current plan:
//          - have a Bufreader in main thread: reads until current line separator, then calls
//            split for field separator and sets $0. (That's in the bytecode).
//          - Build up map from file name to output file ID. Send on channel to background thread
//            with file ID and payload. (but if we send the files over a channel, can we avoid
//            excessive allocations? I suppose allocations are the least of our worries if we are
//            also going to be writing output)
//   * Conversions:
//      - Current plan: do pass with regex, then use simdjson (or stdlib). Benchmark with both.

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
