/// Immutable views on reference-counted data.
///
/// For now, we only support types that are of 'static lifetime. My
/// efforts so far to provide a safe API without this limitation have
/// not been successful, and the project currently only requires types
/// without internal references to non-'static members.
use smallvec::SmallVec;
use std::fmt;
use std::iter::FromIterator;
use std::rc::Rc;
use std::any::Any;

pub(crate) struct Shared<B: ?Sized, T: ?Sized> {
    base: Rc<B>,
    trans: *const T,
}

impl<B: 'static + ?Sized, T: 'static + ?Sized> fmt::Debug for Shared<B, T>
where
    B: fmt::Debug,
    T: fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(
            fmt,
            "Shared {{ base: {:?}, trans: {:?} }}",
            &*self.base,
            self.get()
        )
    }
}

impl<B: ?Sized, T: ?Sized> Clone for Shared<B, T> {
    fn clone(&self) -> Shared<B, T> {
        Shared {
            base: self.base.clone(),
            trans: self.trans,
        }
    }
}

impl<T: ?Sized + 'static> From<Rc<T>> for Shared<T, T> {
    fn from(rc: Rc<T>) -> Self {
        Shared {
            base: rc.clone(),
            trans: &*rc as *const T,
        }
    }
}

impl<B: ?Sized + 'static, T: ?Sized + 'static> Shared<B, T> {
    pub(crate) fn get(&self) -> &T {
        unsafe { &*self.trans }
    }

    pub(crate) fn extend<R: ?Sized + 'static>(&self, f: impl FnOnce(&T) -> &R) -> Shared<B, R> {
        Shared {
            base: self.base.clone(),
            trans: f(self.get()) as *const R,
        }
    }
    pub(crate) fn extend_opt<R: ?Sized + 'static>(
        &self,
        f: impl FnOnce(&T) -> Option<&R>,
    ) -> Option<Shared<B, R>> {
        Some(Shared {
            base: self.base.clone(),
            trans: if let Some(r) = f(self.get()) {
                r as *const R
            } else {
                return None;
            },
        })
    }

    pub(crate) fn extend_slice<R: ?Sized + 'static>(
        &self,
        f: impl FnOnce(&T) -> SmallVec<[&R; 8]>,
    ) -> SharedSlice<B, R> {
        SharedSlice {
            base: self.base.clone(),
            trans: Rc::from_iter(f(self.get()).into_iter().map(|x| x as *const R)),
        }
    }
}

pub(crate) struct SharedSlice<B: ?Sized, T: ?Sized> {
    base: Rc<B>,
    trans: Rc<[*const T]>,
}

impl<B: ?Sized, T: ?Sized> Clone for SharedSlice<B, T> {
    fn clone(&self) -> SharedSlice<B, T> {
        SharedSlice {
            base: self.base.clone(),
            trans: self.trans.clone(),
        }
    }
}

impl<B: ?Sized + 'static, T: ?Sized + 'static> SharedSlice<B, T> {
    pub(crate) fn len(&self) -> usize {
        self.trans.len()
    }
    pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
        self.trans.iter().map(|x| unsafe { &**x })
    }

    pub(crate) fn get(&self, i: usize) -> Option<&T> {
        self.trans.get(i).map(|x| unsafe { &**x })
    }

    pub(crate) fn get_shared(&self, i: usize) -> Option<Shared<B, T>> {
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
        let s: Shared<_, u32> = {
            let s: Shared<_, S> = {
                let s: Shared<Notifier<S>, Notifier<S>> = Shared::from(Rc::new(n));
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
        let x: Rc<str> = "hello there".into();
        let y: Shared<str, str> = x.into();
        let z: SharedSlice<str, str> = y.extend_slice(|x| x.split(" ").into_iter().collect());
        assert_eq!(z.get(0), Some("hello"));
    }
}
