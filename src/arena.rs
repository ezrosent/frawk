use std::cell::{Cell, UnsafeCell};
use std::marker::PhantomData;
use std::mem;
use std::ptr;

struct Buffer {
    len: Cell<usize>,
    data: PushOnlyVec<u8>,
}

impl Buffer {
    fn of_size(cap: usize) -> Buffer {
        Buffer {
            len: Cell::new(0),
            data: PushOnlyVec::with_capacity(cap),
        }
    }

    fn alloc<T>(&self, f: &impl Fn() -> T) -> Option<&T> {
        let size = mem::size_of::<T>();
        let align = mem::align_of::<T>();
        let len = self.len.get();
        let cap = self.data.capacity();

        let extra = unsafe { self.data.get_raw_mut(len).align_offset(align) };
        let start = len + extra;
        let new_len = start + size;
        if new_len > cap {
            return None;
        }
        unsafe { ptr::write(self.data.get_raw_mut(start) as *mut T, f()) };
        self.len.set(new_len);
        unsafe { Some(&*(self.data.get_raw(start) as *const T)) }
    }
}

pub struct Arena<'outer> {
    chunk_size: usize,
    data: PushOnlyVec<Buffer>,
    drops: PushOnlyVec<(*mut u8, fn(*mut u8))>,
    _marker: PhantomData<*const &'outer ()>,
}

impl<'outer> Drop for Arena<'outer> {
    fn drop(&mut self) {
        for (p, f) in self.drops.drain() {
            f(p)
        }
    }
}

impl<'outer> Arena<'outer> {
    pub fn with_size(s: usize) -> Arena<'outer> {
        let res = Arena {
            chunk_size: s,
            data: Default::default(),
            drops: Default::default(),
            _marker: PhantomData,
        };
        res.data.push(Buffer::of_size(s));
        res
    }

    fn head(&self) -> &Buffer {
        self.data.get(self.data.len() - 1)
    }

    pub fn alloc<T: 'outer>(&self, f: impl Fn() -> T) -> &T {
        if let Some(r) = self.head().alloc(&f) {
            fn free<T>(p: *mut u8) {
                unsafe { ptr::drop_in_place(p as *mut T) };
            }
            if mem::needs_drop::<T>() {
                self.drops.push((r as *const _ as *mut u8, free::<T>));
            }
            return r;
        }
        if mem::size_of::<T>() >= self.chunk_size / 2 {
            let b = Box::new(f());
            let p = Box::into_raw(b);
            fn free_large<T>(p: *mut u8) {
                unsafe { mem::drop(Box::from_raw(p as *mut T)) }
            }
            self.drops.push((p as *mut u8, free_large::<T>));
            unsafe { &*(p as *const T) }
        } else {
            self.data.push(Buffer::of_size(self.chunk_size));
            self.alloc(f)
        }
    }
}

struct PushOnlyVec<T>(UnsafeCell<Vec<T>>);

impl<T> Default for PushOnlyVec<T> {
    fn default() -> PushOnlyVec<T> {
        PushOnlyVec(UnsafeCell::new(Default::default()))
    }
}

impl<T> PushOnlyVec<T> {
    fn with_capacity(n: usize) -> PushOnlyVec<T> {
        PushOnlyVec(UnsafeCell::new(Vec::with_capacity(n)))
    }

    fn len(&self) -> usize {
        unsafe { (*self.0.get()).len() }
    }

    fn capacity(&self) -> usize {
        unsafe { (*self.0.get()).capacity() }
    }

    fn push(&self, item: T) {
        unsafe { (&mut *self.0.get()).push(item) }
    }

    fn get(&self, ix: usize) -> &T {
        unsafe { (*self.0.get()).get(ix).unwrap() }
    }

    unsafe fn get_raw(&self, ix: usize) -> *const T {
        debug_assert!(ix <= self.capacity());
        (*self.0.get()).get_unchecked(ix)
    }

    unsafe fn get_raw_mut(&self, ix: usize) -> *mut T {
        debug_assert!(ix <= self.capacity());
        (*self.0.get()).get_unchecked_mut(ix)
    }

    fn drain(&mut self) -> impl Iterator<Item = T> {
        let rest = mem::replace(self, Default::default());
        rest.0.into_inner().into_iter()
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use super::*;
    use test::{black_box, Bencher};

    #[test]
    fn basic_alloc() {
        let mut v = Vec::new();
        let a = Arena::with_size(1 << 10);
        let mut sum = 0;
        for i in 0..1024 {
            v.push(a.alloc(|| i));
            sum += i;
        }
        let mut rsum = 0;
        for i in v.iter() {
            rsum += *i;
        }
        assert_eq!(sum, rsum);
    }

    #[test]
    fn drop_small() {
        static mut N_DROPS: usize = 0;
        struct Dropper;
        impl Drop for Dropper {
            fn drop(&mut self) {
                unsafe {
                    N_DROPS += 1;
                }
            }
        }
        {
            let a = Arena::with_size(100);
            for _ in 0..128 {
                a.alloc(|| Dropper);
            }
        }
        assert_eq!(unsafe { N_DROPS }, 128);
    }

    #[test]
    fn drop_big() {
        static mut N_DROPS: usize = 0;
        #[derive(Default)]
        struct Dropper([u64; 32]);
        impl Drop for Dropper {
            fn drop(&mut self) {
                unsafe {
                    N_DROPS += 1;
                }
            }
        }
        {
            let a = Arena::with_size(100);
            for _ in 0..128 {
                a.alloc(Dropper::default);
            }
        }
    }
    enum Arith1 {
        N(i64),
        Add(Box<Arith1>, Box<Arith1>),
        Sub(Box<Arith1>, Box<Arith1>),
    }

    enum Arith2<'a> {
        N(i64),
        Add(&'a Arith2<'a>, &'a Arith2<'a>),
        Sub(&'a Arith2<'a>, &'a Arith2<'a>),
    }

    fn build_1(depth: usize) -> Box<Arith1> {
        use Arith1::*;
        let mut expr = Box::new(N(1));
        for i in 0..depth {
            if i % 2 == 0 {
                expr = Box::new(Add(expr, Box::new(N(i as i64))));
            } else {
                expr = Box::new(Sub(expr, Box::new(N(i as i64))));
            }
        }
        expr
    }

    fn build_2<'a, 'outer>(a: &'a Arena<'outer>, depth: usize) -> &'a Arith2<'a> {
        use Arith2::*;
        let mut expr = a.alloc(|| N(1));
        for i in 0..depth {
            if i % 2 == 0 {
                expr = a.alloc(|| Add(expr, a.alloc(|| N(i as i64))));
            } else {
                expr = a.alloc(|| Sub(expr, a.alloc(|| N(i as i64))));
            }
        }
        expr
    }

    fn build_2_cheat<'a, 'outer>(a: &'a Arena<'outer>, depth: usize) -> &'a Arith2<'a> {
        use Arith2::*;
        let n1 = a.alloc(|| N(1));
        let mut expr = n1;
        for i in 0..depth {
            if i % 2 == 0 {
                expr = a.alloc(|| Add(expr, n1));
            } else {
                expr = a.alloc(|| Sub(expr, n1));
            }
        }
        expr
    }

    #[bench]
    fn arith_box_1000(b: &mut Bencher) {
        b.iter(|| black_box(build_1(1000)))
    }

    #[bench]
    fn arith_arena_1000(b: &mut Bencher) {
        b.iter(|| {
            let a = Arena::with_size(1 << 12);
            black_box(build_2(&a, 1000));
        })
    }

    #[bench]
    fn arith_arena_cheat_1000(b: &mut Bencher) {
        b.iter(|| {
            let a = Arena::with_size(1 << 12);
            black_box(build_2_cheat(&a, 1000));
        })
    }
}
