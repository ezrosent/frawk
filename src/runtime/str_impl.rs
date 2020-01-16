use super::shared::Shared;
use regex::Regex;
use smallvec::SmallVec;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

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
