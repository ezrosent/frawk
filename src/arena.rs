use std::cell::{UnsafeCell, Cell};
use std::marker::PhantomData;
use std::mem::{self, ManuallyDrop};
use std::ptr;
// * Chunks + ref + borrow

struct PushOnlyVec<T>(UnsafeCell<Vec<T>>);

impl<T> Default for PushOnlyVec<T> {
    fn default() -> PushOnlyVec<T> {
        PushOnlyVec(UnsafeCell::new(Default::default()))
    }
}

impl<T> PushOnlyVec<T> {

    fn len(&self) -> usize {
        unsafe {
            (*self.0.get()).len()
        }
    }

    fn capacity(&self) -> usize {
        unsafe {
            (*self.0.get()).capacity()
        }
    }

    fn reserve(&self, n: usize) {
        unsafe {
            (*self.0.get()).reserve(n)
        }
    }

    fn push(&self, item: T) {
        unsafe {
            (&mut*self.0.get()).push(item)
        }
    }

    unsafe fn get_raw(&self, ix: usize) -> *const T {
        debug_assert!(ix <= self.capacity());
        (*self.0.get()).get_unchecked(ix)
    }

    unsafe fn get_raw_mut(&self, ix: usize) -> *mut T {
        debug_assert!(ix <= self.capacity());
        (*self.0.get()).get_unchecked_mut(ix)
    }

    fn drain(&mut self) -> impl Iterator<Item=T> {
        let rest = mem::replace(self, Default::default());
        rest.0.into_inner().into_iter()
    }
}

#[derive(Default)]
pub struct Arena {
    drops: ManuallyDrop<PushOnlyVec<(usize, fn(*mut u8))>>,
    data: PushOnlyVec<u8>,
    len: Cell<usize>,
}

pub struct Ref<'a, T> {
    offset: usize,
    _marker: PhantomData<(*const T, &'a Cell<u8>)>,
}

impl<'a, T> Copy for Ref<'a, T> {}

impl<'a, T> Clone for Ref<'a, T> {
    fn clone(&self) -> Ref<'a, T> {
        Ref {
            offset: self.offset,
            _marker: PhantomData,
        }
    }
}

impl Arena {
    pub fn alloc<'a,T>(&'a self, t: T) -> Ref<'a, T> {
        let size = mem::size_of::<T>();
        let align = mem::align_of::<T>();
        let len = self.len.get();
        let cap = self.data.capacity();

        let extra = unsafe { self.data.get_raw_mut(len).align_offset(align) };
        let start = len + extra;
        let new_len = start + size;
        if new_len > cap {
            self.data.reserve(new_len - cap);
        }
        unsafe { ptr::write(self.data.get_raw_mut(start) as *mut T, t) };
        self.len.set(new_len);
        if mem::needs_drop::<T>() {
            fn free<T>(p: *mut u8) {
                unsafe {
                    ptr::drop_in_place(p as *mut T)
                };
            }
            self.drops.push((start, free::<T>))
        }
        Ref {
            offset: start,
            _marker: PhantomData,
        }
    }
}
