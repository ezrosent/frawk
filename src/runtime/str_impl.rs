use super::shared::Shared;
use regex::Regex;
use smallvec::SmallVec;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

mod new {
    use std::cell::Cell;
    use std::marker::PhantomData;
    use std::mem;
    use std::rc::Rc;

    #[repr(C)]
    struct SharedStr {
        start: *const u8,
        len: usize,
        base: Rc<String>,
    }

    #[repr(C)]
    struct Literal<'a> {
        ptr: *const u8,
        len: usize,
        marker: PhantomData<&'a str>,
    }

    struct ConcatInner<'a> {
        len: u32,
        left: Str<'a>,
        right: Str<'a>,
    }

    const EMPTY: usize = 0;
    const SHARED: usize = 1;
    const LITERAL: usize = 2;
    const CONCAT: usize = 3;
    const NUM_VARIANTS: usize = 4;

    #[repr(transparent)]
    struct Inner<'a>(usize, PhantomData<&'a ()>);

    impl<'a> Default for Inner<'a> {
        fn default() -> Inner<'a> {
            Inner(0, PhantomData)
        }
    }

    impl<'a> From<Rc<SharedStr>> for Inner<'a> {
        fn from(s: Rc<SharedStr>) -> Inner<'a> {
            unsafe {
                Inner(
                    mem::transmute::<Rc<SharedStr>, usize>(s) | SHARED,
                    PhantomData,
                )
            }
        }
    }

    impl<'a> From<Rc<ConcatInner<'a>>> for Inner<'a> {
        fn from(s: Rc<ConcatInner<'a>>) -> Inner<'a> {
            unsafe {
                Inner(
                    mem::transmute::<Rc<ConcatInner>, usize>(s) | CONCAT,
                    PhantomData,
                )
            }
        }
    }
    impl<'a> From<Rc<Literal<'a>>> for Inner<'a> {
        fn from(lit: Rc<Literal<'a>>) -> Inner<'a> {
            unsafe {
                Inner(
                    mem::transmute::<Rc<Literal<'a>>, usize>(lit) | LITERAL,
                    PhantomData,
                )
            }
        }
    }

    impl<'a> From<String> for Inner<'a> {
        fn from(s: String) -> Inner<'a> {
            if s.len() == 0 {
                return Inner::default();
            }
            let rcd = Rc::new(s);
            Rc::new(SharedStr {
                start: rcd.as_ptr(),
                len: rcd.len(),
                base: rcd.clone(),
            })
            .into()
        }
    }

    impl<'a> From<&'a str> for Inner<'a> {
        fn from(s: &'a str) -> Inner<'a> {
            Rc::new(Literal {
                ptr: s.as_ptr(),
                len: s.len(),
                marker: PhantomData,
            })
            .into()
        }
    }

    impl<'a> Clone for Inner<'a> {
        fn clone(&self) -> Inner<'a> {
            let tag = self.0 & 0x7;
            let addr = self.0 & !(0x7);
            debug_assert!(tag < NUM_VARIANTS);
            unsafe {
                match tag {
                    SHARED => mem::transmute::<usize, Rc<SharedStr>>(addr).clone().into(),
                    // for completeness. This drop should be trivial
                    LITERAL => mem::transmute::<usize, Rc<Literal<'a>>>(addr)
                        .clone()
                        .into(),
                    CONCAT => mem::transmute::<usize, Rc<ConcatInner<'a>>>(addr)
                        .clone()
                        .into(),
                    EMPTY => Inner::default(),
                    _ => unreachable!(),
                }
            }
        }
    }

    impl<'a> Drop for Inner<'a> {
        fn drop(&mut self) {
            let tag = self.0 & 0x7;
            let addr = self.0 & !(0x7);
            debug_assert!(tag < NUM_VARIANTS);
            unsafe {
                match tag {
                    // TODO fix
                    SHARED => mem::drop(mem::transmute::<usize, Rc<SharedStr>>(addr)),
                    // for completeness. This drop should be trivial
                    LITERAL => mem::drop(mem::transmute::<usize, Rc<Literal<'a>>>(addr)),
                    CONCAT => mem::drop(mem::transmute::<usize, Rc<ConcatInner<'a>>>(addr)),
                    _ => {}
                }
            }
        }
    }

    #[repr(transparent)]
    struct Str<'a>(Cell<Inner<'a>>);
}

#[derive(Clone, Debug)]
enum Inner<'a> {
    Literal(&'a str),
    Line(Shared<str>),
    Boxed(Rc<str>),
    Concat(Rc<Branch<'a>>),
}

#[derive(Clone, Debug)]
struct Branch<'a> {
    len: u32,
    left: Str<'a>,
    right: Str<'a>,
}

#[derive(Clone, Debug)]
pub struct Str<'a>(RefCell<Inner<'a>>);
impl<'a> Default for Str<'a> {
    fn default() -> Str<'a> {
        Str(RefCell::new(Inner::Literal("")))
    }
}

impl<'a> Hash for Str<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.with_str(|s| s.hash(state))
    }
}

impl<'a> Str<'a> {
    fn len_u32(&self) -> u32 {
        use Inner::*;
        match &*self.0.borrow() {
            Literal(s) => conv_len(s.len()),
            Boxed(s) => conv_len(s.len()),
            Line(s) => conv_len(s.get().len()),
            Concat(b) => b.len,
        }
    }
    pub(crate) fn len(&self) -> usize {
        use Inner::*;
        // XXX for now, this will return u32::max for large strings. That should be fine, but if it
        // becomes an issue, we can call force before taking the length.
        match &*self.0.borrow() {
            Literal(s) => s.len(),
            Boxed(s) => s.len(),
            Line(s) => s.get().len(),
            Concat(b) => {
                if b.len == u32::max_value() {
                    self.force();
                    self.len()
                } else {
                    b.len as usize
                }
            }
        }
    }
    /// We implement split here directly rather than just use with_str
    /// and split because it allows the runtime to share memory across
    /// different strings. Splitting a literal yields references
    /// into that literal. Splitting a boxed string or a line uses
    /// the Shared functionality to yield references into those
    /// strings, while also incrementing the reference count of the original string.
    pub(crate) fn split(&self, pat: &Regex, mut push: impl FnMut(Str<'a>)) {
        self.force();
        use Inner::*;
        let mut line = |s: &Shared<str>| {
            for s in s
                // This uses the `IterRefFn` implementation for `&Regex`
                .shared_iter(pat)
                .map(|s| Str(RefCell::new(Line(s))))
            {
                push(s)
            }
        };
        match &*self.0.borrow() {
            Literal(s) => {
                for s in pat.split(s).map(|s| Str(RefCell::new(Literal(s)))) {
                    push(s)
                }
            }
            Boxed(s) => line(&Shared::from(s.clone())),
            Line(s) => line(s),
            Concat(_) => unreachable!(),
        }
    }
}

impl<'a> PartialEq for Str<'a> {
    fn eq(&self, other: &Str<'a>) -> bool {
        if self.len_u32() != other.len_u32() {
            return false;
        }
        self.with_str(|s1| other.with_str(|s2| s1 == s2))
    }
}
impl<'a> Eq for Str<'a> {}

fn conv_len(l: usize) -> u32 {
    if l > (u32::max_value() as usize) {
        u32::max_value()
    } else {
        l as u32
    }
}

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

impl<'a> From<Shared<str>> for Str<'a> {
    fn from(s: Shared<str>) -> Str<'a> {
        Str(RefCell::new(Inner::Line(s)))
    }
}

impl<'a> Str<'a> {
    pub(crate) fn clone_str(&self) -> Rc<str> {
        self.force();
        match &*self.0.borrow() {
            Inner::Literal(l) => (*l).into(),
            Inner::Boxed(b) => b.clone(),
            Inner::Line(l) => l.get().into(),
            Inner::Concat(_) => unreachable!(),
        }
    }
    pub(crate) fn with_str<R>(&self, f: impl FnOnce(&str) -> R) -> R {
        self.force();
        match &*self.0.borrow() {
            Inner::Literal(l) => f(l),
            Inner::Boxed(b) => f(&*b),
            Inner::Line(l) => f(l.get()),
            Inner::Concat(_) => unreachable!(),
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
                    Line(s) => res.push_str(s.get()),
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
