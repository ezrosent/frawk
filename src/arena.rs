//! A basic arena allocator.
//!
//! This used to include a custom implementation based on frozen vectors that respected Drop calls
//! automatically (without using a Box type).  It has since been moved to a solution based on the
//! bumpalo crate, which is now the standard in the broader Rust ecosystem. We keep this as a
//! wrapper for a couple reasons:
//!
//! 1. bumpalo returns a mutable reference by default, but all of the uses of arena-allocated data
//!    in this project use immutable references.
//! 2. For byte slices and strings, frawk's runtime has special runtime requirements, so we use the
//!    extra wrapper to enforce those rather than passing them down to the user.
use std::ptr;

#[derive(Default)]
pub struct Arena(bumpalo::Bump);
pub type Vec<'a, T> = bumpalo::collections::Vec<'a, T>;

impl Arena {
    pub fn vec_with_capacity<T>(&self, capacity: usize) -> Vec<'_, T> {
        Vec::with_capacity_in(capacity, &self.0)
    }
    pub fn new_vec<T>(&self) -> Vec<'_, T> {
        Vec::new_in(&self.0)
    }
    pub fn new_vec_from_slice<'a, T: Clone>(&'a self, elts: &[T]) -> Vec<'a, T> {
        let mut res = Vec::with_capacity_in(elts.len(), &self.0);
        res.extend(elts.iter().cloned());
        res
    }
    pub fn alloc_str<'a>(&'a self, s: &str) -> &'a str {
        let bs = self.alloc_bytes(s.as_bytes());
        unsafe { std::str::from_utf8_unchecked(bs) }
    }

    // NB: do not use this to allocate a byte slice (will get assertion failures), use alloc_bytes instead
    pub fn alloc_slice<'a, T: Clone>(&'a self, t: &[T]) -> &'a [T] {
        self.0.alloc_slice_clone(t)
    }

    pub fn alloc_bytes<'a>(&'a self, bs: &[u8]) -> &'a [u8] {
        // We want all of these strings to be 8-byte aligned, due to how we represent string
        // contents at runtime in frawk.
        unsafe {
            let res_p = self
                .0
                .alloc_layout(std::alloc::Layout::from_size_align(bs.len(), 8).unwrap())
                .as_ptr();
            ptr::copy_nonoverlapping(bs.as_ptr(), res_p, bs.len());
            std::slice::from_raw_parts(res_p, bs.len())
        }
    }
    pub fn alloc<T>(&self, t: T) -> &T {
        self.0.alloc(t)
    }
}

#[cfg(all(feature = "unstable", test))]
mod bench {
    use super::*;
    extern crate test;
    use test::{black_box, Bencher};
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

    fn build_2<'a>(a: &'a Arena, depth: usize) -> &'a Arith2<'a> {
        use Arith2::*;
        let mut expr = a.alloc(N(1));
        for i in 0..depth {
            if i % 2 == 0 {
                expr = a.alloc(Add(expr, a.alloc(N(i as i64))));
            } else {
                expr = a.alloc(Sub(expr, a.alloc(N(i as i64))));
            }
        }
        expr
    }

    fn build_2_cheat<'a>(a: &'a Arena, depth: usize) -> &'a Arith2<'a> {
        use Arith2::*;
        let n1 = a.alloc(N(1));
        let mut expr = n1;
        for i in 0..depth {
            if i % 2 == 0 {
                expr = a.alloc(Add(expr, n1));
            } else {
                expr = a.alloc(Sub(expr, n1));
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
