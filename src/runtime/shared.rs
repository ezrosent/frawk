/// Immutable views on reference-counted data.
///
/// For now, we only support types that are of 'static lifetime. My
/// efforts so far to provide a safe API without this limitation have
/// not been successful, and the project currently only requires
/// types without internal references to non-'static members. This
/// simplification also allows us to have the `base` value be a `dyn
/// Any`, which simplifies the APIs considerably.
use std::any::Any;
use std::fmt;
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

impl<'a, 'b> IterRefFn<'a, str, str> for &'b regex::Regex {
    type I = regex::Split<'b, 'a>;
    fn invoke(self, a: &'a str) -> Self::I {
        self.split(a)
    }
}

impl<T: ?Sized + 'static> Shared<T> {
    pub(crate) fn shared_iter<'b, R: ?Sized + 'static>(
        &'b self,
        f: impl for<'a> IterRefFn<'a, T, R> + 'b,
    ) -> impl Iterator<Item = Shared<R>> + '_ {
        let base = self.base.clone();
        f.invoke(self.get()).map(move |x| Shared {
            base: base.clone(),
            trans: x as *const R,
        })
    }
}

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
        let re = regex::Regex::new(" ").unwrap();
        let mut z_iter = y.shared_iter(&re);
        assert_eq!("hello", z_iter.next().unwrap().get());
    }
}
