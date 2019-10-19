/// Immutable views on reference-counted data.
///
/// For now, we only support types that are of 'static lifetime. My
/// efforts so far to provide a safe API without this limitation have
/// not been successful, and the project currently only requires
/// types without internal references to non-'static members. This
/// simplification also allows us to have the `base` value be a `dyn
/// Any`, which simplifies the APIs considerably.
use smallvec::SmallVec;
use std::any::Any;
use std::fmt;
use std::iter::FromIterator;
use std::rc::Rc;

pub(crate) struct Shared<T: ?Sized> {
    base: Rc<dyn Any>,
    trans: *const T,
}

impl<T: 'static + ?Sized> fmt::Debug for Shared<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "Shared {{ trans: {:?} }}", self.get())
    }
}

impl<T: ?Sized> Clone for Shared<T> {
    fn clone(&self) -> Shared<T> {
        Shared {
            base: self.base.clone(),
            trans: self.trans,
        }
    }
}

// Using some unstable standard library features to avoid constructing
// an intermediate vector when making a slice.

pub(crate) trait IterRefFn<'a, A: 'a + ?Sized, B: 'a + ?Sized> {
    type I: Iterator<Item = &'a B>;
    fn invoke(self, a: &'a A) -> Self::I;
}

/*
impl<'a, A: 'a + ?Sized, B: 'a + ?Sized, F, I> IterRefFn<'a, A, B> for F
where
    I: Iterator<Item = &'a B>,
    F: std::ops::FnOnce<&'a A, Output = I>,
{
    type I = I;
    fn invoke(self, a: &'a A) -> I {
        self.call_once(a)
    }
}
// NOTE passing |s| s.split(" ") does not compile. Nor does:
// fn split_str<'a>(s: &'a str) -> impl Iterator<Item=&'a str> {
//   s.split(" ")
// }
// Neither of them actually seem to implement this. If we do this
// sort of custom-fn-trait hackery though, I think we can get rid of
// the unstable features.
*/

impl<T: ?Sized + 'static> Shared<T> {
    pub(crate) fn extend_slice_iter<R: ?Sized + 'static>(
        &self,
        f: impl for<'a> IterRefFn<'a, T, R>,
    ) -> SharedSlice<R> {
        SharedSlice {
            base: self.base.clone(),
            trans: Rc::from_iter(f.invoke(self.get()).map(|x| x as *const R)),
        }
    }
}

fn _test() {
    let s: Box<str> = "hi there".into();
    let s0: Shared<str> = s.into();
    #[derive(Copy, Clone)]
    struct Split;
    impl<'a> IterRefFn<'a, str, str> for Split {
        type I = std::str::Split<'a, &'a str>;
        fn invoke(self, a: &'a str) -> Self::I {
            a.split(" ")
        }
    }

    let s1: SharedSlice<str> = s0.extend_slice_iter(Split);
}

// What are the options here?
// 1. put everything under a Box (Rc<Box<str>>). It's annoying and slightly less efficient in that
//    we do an extra allocation, but at least we won't pay for the indirection.
// 2. pay for the extra type parameter (TODO: look into this if the
// allocations get too strenuous; we could tweak the reading API to build an Rc<str> from the data
// directly and then have a Shared<str,str> be the only relevant type)
impl<T: ?Sized + 'static> From<Box<T>> for Shared<T> {
    fn from(b: Box<T>) -> Self {
        let trans = &*b as *const T;
        let base: Rc<dyn Any> = Rc::new(b);
        Shared { base, trans }
    }
}

impl<T: ?Sized + 'static> From<Rc<T>> for Shared<T> {
    fn from(r: Rc<T>) -> Self {
        let trans = &*r as *const T;
        let base: Rc<dyn Any> = Rc::new(r);
        Shared { base, trans }
    }
}

impl<T: ?Sized + 'static> Shared<T> {
    pub(crate) fn get(&self) -> &T {
        unsafe { &*self.trans }
    }

    pub(crate) fn extend<R: ?Sized + 'static>(&self, f: impl FnOnce(&T) -> &R) -> Shared<R> {
        Shared {
            base: self.base.clone(),
            trans: f(self.get()) as *const R,
        }
    }
    pub(crate) fn extend_opt<R: ?Sized + 'static>(
        &self,
        f: impl FnOnce(&T) -> Option<&R>,
    ) -> Option<Shared<R>> {
        let trans = if let Some(r) = f(self.get()) {
            r as *const R
        } else {
            return None;
        };
        let base = self.base.clone();
        Some(Shared { base, trans })
    }

    // TODO: we may just need to have an iterator over Shared<R>. can we do that without an extra
    // allocation?
    pub(crate) fn extend_slice<R: ?Sized + 'static>(
        &self,
        f: impl FnOnce(&T) -> SmallVec<[&R; 8]>,
    ) -> SharedSlice<R> {
        SharedSlice {
            base: self.base.clone(),
            trans: Rc::from_iter(f(self.get()).into_iter().map(|x| x as *const R)),
        }
    }
}

pub(crate) struct SharedSlice<T: ?Sized> {
    base: Rc<dyn Any>,
    trans: Rc<[*const T]>,
}

impl<T: ?Sized> Clone for SharedSlice<T> {
    fn clone(&self) -> SharedSlice<T> {
        SharedSlice {
            base: self.base.clone(),
            trans: self.trans.clone(),
        }
    }
}

impl<T: ?Sized + 'static> SharedSlice<T> {
    pub(crate) fn len(&self) -> usize {
        self.trans.len()
    }
    pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
        self.trans.iter().map(|x| unsafe { &**x })
    }
    pub(crate) fn iter_shared(&self) -> impl Iterator<Item = Shared<T>> + '_{
        let base = self.base.clone();
        self.trans.iter().map(move |x| Shared { base: base.clone(), trans: *x })
    }
    pub(crate) fn unpack(&self) -> SmallVec<[Shared<T>; 4]> {
        self.trans
            .iter()
            .map(|x| Shared {
                base: self.base.clone(),
                trans: *x,
            })
            .collect()
    }

    pub(crate) fn get(&self, i: usize) -> Option<&T> {
        self.trans.get(i).map(|x| unsafe { &**x })
    }

    pub(crate) fn get_shared(&self, i: usize) -> Option<Shared<T>> {
        self.trans.get(i).map(|x| Shared {
            base: self.base.clone(),
            trans: *x,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::cell::RefCell;

    struct Notifier<T> {
        t: T,
        dropped: Rc<RefCell<bool>>,
    }

    impl<T> Notifier<T> {
        fn new(t: T) -> (Notifier<T>, Rc<RefCell<bool>>) {
            let dropped = Rc::new(RefCell::new(false));
            let res = dropped.clone();
            (Notifier { t, dropped }, res)
        }
    }

    impl<T> Drop for Notifier<T> {
        fn drop(&mut self) {
            *self.dropped.borrow_mut() = true;
        }
    }

    #[test]
    fn field_ref() {
        struct S((u32, u32), u32);
        let (n, dropped) = Notifier::new(S((0, 1), 2));
        assert_eq!(*dropped.borrow(), false);
        let s: Shared<u32> = {
            let s: Shared<S> = {
                let s: Shared<Notifier<S>> = Shared::from(Box::new(n));
                s.extend(|n| &n.t)
            };
            assert_eq!(*dropped.borrow(), false);
            assert_eq!((s.get().0).1, 1);
            s.extend(|s| &s.1)
        };
        assert_eq!(*dropped.borrow(), false);
        assert_eq!(*s.get(), 2);
        std::mem::drop(s);
        assert!(*dropped.borrow(), true);
    }

    #[test]
    fn string_split() {
        let x: Box<str> = "hello there".into();
        let y: Shared<str> = x.into();
        let z: SharedSlice<str> = y.extend_slice(|x| x.split(" ").into_iter().collect());
        assert_eq!(z.get(0), Some("hello"));
    }
}
