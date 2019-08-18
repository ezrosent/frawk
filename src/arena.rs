//! Implement a simple arena-allocation mechanism around frozen vectors.
use crate::elsa::FrozenVec;
use crate::stable_deref_trait::StableDeref;

use std::alloc::{alloc, dealloc, Layout};
use std::cell::Cell;
use std::marker::PhantomData;
use std::mem;
use std::ptr;

/// The size of arena chunks. Allocating objects larger than this will fall back to the heap.
const CHUNK_SIZE: usize = 1024;

struct Chunk {
    len: Cell<usize>,
}

impl Chunk {
    fn layout() -> Layout {
        Layout::from_size_align(CHUNK_SIZE, mem::align_of::<Chunk>()).unwrap()
    }
    unsafe fn new() -> *mut Chunk {
        let chunk = alloc(Self::layout()) as *mut Chunk;
        ptr::write(
            chunk,
            Chunk {
                len: Cell::new(mem::size_of::<Chunk>()),
            },
        );
        chunk
    }
    unsafe fn dealloc(chunk: *mut Chunk) {
        dealloc(chunk as *mut u8, Self::layout());
    }
    unsafe fn get_raw(&self, off: usize) -> *mut u8 {
        let p = self as *const Chunk as *mut Chunk as *mut u8;
        p.offset(off as isize)
    }
    unsafe fn alloc_inner<T>(&self, n: usize) -> Option<*mut T> {
        let size = mem::size_of::<T>() * n;
        let align = mem::align_of::<T>();
        let len = self.len.get();
        let cap = CHUNK_SIZE;

        let extra = self.get_raw(len).align_offset(align);
        let start = len + extra;
        let new_len = start + size;
        if new_len > cap {
            return None;
        }
        // must set len before calling f, as f may call alloc recursively.
        self.len.set(new_len);
        Some(self.get_raw(start) as *mut T)
    }

    fn alloc<T>(&self, f: &impl Fn() -> T) -> Option<&T> {
        unsafe {
            let start = self.alloc_inner::<T>(1)?;
            ptr::write(start, f());
            Some(&*start)
        }
    }
}

struct ChunkPtr(*mut Chunk);

impl Drop for ChunkPtr {
    fn drop(&mut self) {
        unsafe { Chunk::dealloc(self.0) };
    }
}

impl Default for ChunkPtr {
    fn default() -> ChunkPtr {
        unsafe { ChunkPtr(Chunk::new()) }
    }
}

impl std::ops::Deref for ChunkPtr {
    type Target = Chunk;
    fn deref(&self) -> &Chunk {
        unsafe { &*self.0 }
    }
}

unsafe impl StableDeref for ChunkPtr {}

pub struct Arena<'outer> {
    data: FrozenVec<ChunkPtr>,
    drops: FrozenVec<Box<dyn Fn()>>,
    _marker: PhantomData<*const &'outer ()>,
}

impl<'outer> Drop for Arena<'outer> {
    fn drop(&mut self) {
        for f in mem::replace(&mut self.drops, Default::default()).into_iter() {
            f()
        }
    }
}

impl<'outer> Default for Arena<'outer> {
    fn default() -> Arena<'outer> {
        let res = Arena {
            data: Default::default(),
            drops: Default::default(),
            _marker: PhantomData,
        };
        res.data.push(Default::default());
        res
    }
}

impl<'outer> Arena<'outer> {
    fn head(&self) -> &Chunk {
        &self.data[self.data.len() - 1]
    }

    // TODO(ezr): implement alloc_many method for collection of our choice (smallvec?)

    pub fn alloc<T: 'outer>(&self, f: impl Fn() -> T) -> &T {
        if let Some(r) = self.head().alloc(&f) {
            if mem::needs_drop::<T>() {
                // Close over a *mut u8 instead of a *mut T. Why? Without this the borrow checker
                // complains that we are moving a T (with lifetime 'outer) into a Box (wwith
                // lifetime 'static). We enforce that the Box's contents will be dropped within
                // 'outer, so casting away this information is safe.
                let rr = r as *const _ as *mut u8;
                self.drops.push(Box::new(move || unsafe {
                    ptr::drop_in_place(rr as *mut T)
                }));
            }
            return r;
        }
        if mem::size_of::<T>() >= CHUNK_SIZE / 2 {
            let b = Box::new(f());
            let p = Box::into_raw(b);
            let r = p as *mut u8;
            self.drops.push(Box::new(move || unsafe {
                mem::drop(Box::from_raw(r as *mut T))
            }));
            unsafe { &*(p as *const T) }
        } else {
            self.data.push(ChunkPtr::default());
            self.alloc(f)
        }
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
        let a = Arena::default();
        let mut sum: usize = 0;
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
            let a = Arena::default();
            for _ in 0..CHUNK_SIZE {
                a.alloc(|| Dropper);
            }
        }
        assert_eq!(unsafe { N_DROPS }, CHUNK_SIZE);
    }

    #[test]
    fn drop_big() {
        static mut N_DROPS: usize = 0;
        #[derive(Default)]
        struct Dropper([[u64; 8]; 32]);
        impl Drop for Dropper {
            fn drop(&mut self) {
                unsafe {
                    N_DROPS += 1;
                }
            }
        }
        {
            let a = Arena::default();
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

    impl<'a> Arith2<'a> {
        fn eval(&self) -> i64 {
            use Arith2::*;
            match self {
                N(i) => *i,
                Add(n1, n2) => n1.eval() + n2.eval(),
                Sub(n1, n2) => n1.eval() - n2.eval(),
            }
        }
    }

    impl Arith1 {
        fn eval(&self) -> i64 {
            use Arith1::*;
            match self {
                N(i) => *i,
                Add(n1, n2) => n1.eval() + n2.eval(),
                Sub(n1, n2) => n1.eval() - n2.eval(),
            }
        }
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
            let a = Arena::default();
            black_box(build_2(&a, 1000));
        })
    }

    #[bench]
    fn arith_arena_cheat_1000(b: &mut Bencher) {
        b.iter(|| {
            let a = Arena::default();
            black_box(build_2_cheat(&a, 1000));
        })
    }

    #[bench]
    fn arith_box_eval_1000(b: &mut Bencher) {
        b.iter(|| black_box(build_1(1000).eval()))
    }

    #[bench]
    fn arith_arena_eval_1000(b: &mut Bencher) {
        let mut i = 0;
        b.iter(|| {
            let a = Arena::default();
            black_box(build_2(&a, 1000).eval());
            i += 1;
        })
    }

    #[bench]
    fn arith_arena_cheat_eval_1000(b: &mut Bencher) {
        let mut i = 0;
        b.iter(|| {
            let a = Arena::default();
            black_box(build_2_cheat(&a, 1000).eval());
            i += 1;
        })
    }

}
