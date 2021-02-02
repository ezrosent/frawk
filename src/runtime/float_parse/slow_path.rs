//! This module contains routines for converting strings to numbers.
//!
//! Right now, it includes several duplicate implementations that area here for benchmarking
//! purposes.

// TODO: explore if simd-json-style techniques could help here. Seems like we would have to change
// things a good amount to support the "partial parses" behavior.

use lazy_static::lazy_static;
use regex::bytes::Regex;
use smallvec::SmallVec;

#[allow(unused)]
pub(crate) fn strtoi(s: &[u8]) -> i64 {
    strtoi_libc(s)
}
pub(crate) fn strtod(s: &[u8]) -> f64 {
    strtod_libc(s)
}

fn strtoi_libc(s: &[u8]) -> i64 {
    let cstr = SmallCString::from_bytes(s);
    unsafe { libc::strtol(cstr.as_ptr(), std::ptr::null_mut(), 10) as i64 }
}

#[allow(unused)]
fn strtod_libc(s: &[u8]) -> f64 {
    let cstr = SmallCString::from_bytes(s);
    unsafe { libc::strtod(cstr.as_ptr(), std::ptr::null_mut()) as f64 }
}

// We keep the regex-based implementations around in case we want to use 100% safe rust, or there
// are portability issues with using libc.

#[allow(unused)]
fn strtoi_regex(s: &[u8]) -> i64 {
    lazy_static! {
        static ref INT_PATTERN: Regex = Regex::new(r"^[+-]?\d+").unwrap();
    };
    if let Some(num) = INT_PATTERN.captures(s).and_then(|c| c.get(0)) {
        std::str::from_utf8(num.as_bytes())
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    } else {
        0
    }
}

#[allow(unused)]
fn strtod_regex(s: &[u8]) -> f64 {
    lazy_static! {
        // Adapted from https://www.regular-expressions.info/floatingpoint.html
        static ref FLOAT_PATTERN: Regex = Regex::new(r"^[-+]?\d*\.?\d+([eE][-+]?\d+)?").unwrap();
    };
    if let Some(num) = FLOAT_PATTERN.captures(s).and_then(|c| c.get(0)) {
        std::str::from_utf8(num.as_bytes())
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0)
    } else {
        0.0
    }
}

/// SmallCString is a simple NUL-terminated string type used for passing to libc. We want to use
/// Rust strings elsewhere, meaning that we have to copy them before invoking C functions. Using
/// SmallVec ensures that these copies don't always require an additional allocation.
///
/// Because we never use the length of the string, some of these operations can be faster as well.
struct SmallCString(SmallVec<[u8; 32]>);
impl SmallCString {
    fn from_bytes(bs: &[u8]) -> SmallCString {
        let mut res = SmallCString(SmallVec::with_capacity(bs.len() + 1));
        res.0.extend_from_slice(bs);
        res.0.push(0);
        res
    }
    fn as_ptr(&self) -> *const std::os::raw::c_char {
        let unsigned: *const u8 = &self.0[0];
        unsigned as *const _
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_strtoi(mut f: impl FnMut(&[u8]) -> i64) {
        assert_eq!(f(b"0"), 0);
        assert_eq!(f(b"012345678910"), 12345678910);
        assert_eq!(f(b"12345678910"), 12345678910);
        assert_eq!(f(b"1.9245"), 1);
        assert_eq!(f(b"567hello there 8910"), 567);
        assert_eq!(f(b"+2037 there 8910"), 2037);
        assert_eq!(f(b"-99999 there 8910"), -99999);
        // NOTE these two have different behavior on {under,over}flow.
        // This shouldn't be that big of a deal for most cases, as most strings will be parsed as
        // floats first.
        //
        // // Rust behavior
        // assert_eq!(f("1180591620717411303424"), 0);
        // // libc behavior (also sets errno)
        // assert_eq!(f("1180591620717411303424"), i64::max_value());
    }

    fn test_strtod(mut f: impl FnMut(&[u8]) -> f64) {
        macro_rules! assert_near {
            ($x:expr, $y:expr) => {{
                let x: f64 = $x;
                let y: f64 = $y;
                let min = x * 0.99;
                let max = x * 1.01;
                let sn = x.signum();
                if sn == 1.0 {
                    assert!(y >= min && y <= max, "{} is not close enough to {}", x, y)
                } else if sn == -1.0 {
                    assert!(y <= min && y >= max, "{} is not close enough to {}", x, y)
                } else {
                    panic!("one of the options is NaN!")
                }
            }};
        }
        assert_eq!(f(b"0"), 0.0);
        assert_near!(f(b"1234"), 1234.0);
        assert_near!(f(b"12.34"), 12.34);
        assert_near!(f(b"12.34e9"), 12.34e9);
        assert_near!(f(b"12.34e9 hello there"), 12.34e9);
        assert_near!(f(b"-12.34e9 hello there"), -12.34e9);
        assert_near!(f(b"+92.34e9 hello there"), 92.34e9);
        assert_near!(f(b"-1."), -1.0);
    }

    #[test]
    fn test_strtoi_regex() {
        test_strtoi(strtoi_regex);
    }
    #[test]
    fn test_strtoi_libc() {
        test_strtoi(strtoi_libc);
    }
    #[test]
    fn test_strtoi_fast() {
        test_strtoi(super::super::strtoi);
    }

    #[test]
    fn test_strtod_regex() {
        test_strtod(strtod_regex);
    }
    #[test]
    fn test_strtod_libc() {
        test_strtod(strtod_libc);
    }
}

#[cfg(all(feature = "unstable", test))]
mod bench {
    use super::*;
    extern crate test;
    use test::{black_box, Bencher};
    fn bench_strtoi_long(b: &mut Bencher, mut f: impl FnMut(&[u8]) -> i64) {
        const INPUT: &[u8] = b"9514590998633183616833425126589570467868 some more data after that";
        b.iter(|| {
            black_box(f(INPUT));
        })
    }
    fn bench_strtoi_medium(b: &mut Bencher, mut f: impl FnMut(&[u8]) -> i64) {
        const INPUT: &[u8] = b"951459099863318361";
        b.iter(|| {
            black_box(f(INPUT));
        })
    }
    fn bench_strtoi_short(b: &mut Bencher, mut f: impl FnMut(&[u8]) -> i64) {
        const INPUT: &[u8] = b"19503609";
        b.iter(|| {
            black_box(f(INPUT));
        })
    }
    fn bench_strtod_long(b: &mut Bencher, mut f: impl FnMut(&[u8]) -> f64) {
        const INPUT: &[u8] =
            b"9514590.998633183616833425126589570467868E22 some more data after that";
        b.iter(|| {
            black_box(f(INPUT));
        })
    }
    fn bench_strtod_medium(b: &mut Bencher, mut f: impl FnMut(&[u8]) -> f64) {
        const INPUT: &[u8] = b"9514005346590.99863E10";
        b.iter(|| {
            black_box(f(INPUT));
        })
    }
    fn bench_strtod_short(b: &mut Bencher, mut f: impl FnMut(&[u8]) -> f64) {
        const INPUT: &[u8] = b"1950360.9";
        b.iter(|| {
            black_box(f(INPUT));
        })
    }

    #[bench]
    fn bench_strtoi_long_regex(b: &mut Bencher) {
        let _ = strtoi_regex(b"0");
        bench_strtoi_long(b, strtoi_regex)
    }
    #[bench]
    fn bench_strtoi_medium_regex(b: &mut Bencher) {
        let _ = strtoi_regex(b"0");
        bench_strtoi_medium(b, strtoi_regex)
    }
    #[bench]
    fn bench_strtoi_short_regex(b: &mut Bencher) {
        let _ = strtoi_regex(b"0");
        bench_strtoi_short(b, strtoi_regex)
    }
    #[bench]
    fn bench_strtod_long_regex(b: &mut Bencher) {
        let _ = strtod_regex(b"0");
        bench_strtod_long(b, strtod_regex)
    }
    #[bench]
    fn bench_strtod_medium_regex(b: &mut Bencher) {
        let _ = strtod_regex(b"0");
        bench_strtod_medium(b, strtod_regex)
    }
    #[bench]
    fn bench_strtod_short_regex(b: &mut Bencher) {
        let _ = strtod_regex(b"0");
        bench_strtod_short(b, strtod_regex)
    }

    #[bench]
    fn bench_strtoi_long_libc(b: &mut Bencher) {
        bench_strtoi_long(b, strtoi_libc)
    }
    #[bench]
    fn bench_strtoi_medium_libc(b: &mut Bencher) {
        bench_strtoi_medium(b, strtoi_libc)
    }
    #[bench]
    fn bench_strtoi_short_libc(b: &mut Bencher) {
        bench_strtoi_short(b, strtoi_libc)
    }
    #[bench]
    fn bench_strtod_long_libc(b: &mut Bencher) {
        bench_strtod_long(b, strtod_libc)
    }
    #[bench]
    fn bench_strtod_medium_libc(b: &mut Bencher) {
        bench_strtod_medium(b, strtod_libc)
    }
    #[bench]
    fn bench_strtod_short_libc(b: &mut Bencher) {
        bench_strtod_short(b, strtod_libc)
    }

    #[bench]
    fn bench_strtod_long_fast(b: &mut Bencher) {
        bench_strtod_long(b, super::super::strtod)
    }
    #[bench]
    fn bench_strtod_medium_fast(b: &mut Bencher) {
        bench_strtod_medium(b, super::super::strtod)
    }
    #[bench]
    fn bench_strtod_short_fast(b: &mut Bencher) {
        bench_strtod_short(b, super::super::strtod)
    }
    #[bench]
    fn bench_strtoi_long_fast(b: &mut Bencher) {
        bench_strtoi_long(b, super::super::strtoi)
    }
    #[bench]
    fn bench_strtoi_medium_fast(b: &mut Bencher) {
        bench_strtoi_medium(b, super::super::strtoi)
    }
    #[bench]
    fn bench_strtoi_short_fast(b: &mut Bencher) {
        bench_strtoi_short(b, super::super::strtoi)
    }

    fn ff_strtod(bs: &[u8]) -> f64 {
        if let Ok((f, _)) = fast_float::parse_partial(bs) {
            f
        } else {
            0.0f64
        }
    }

    #[bench]
    fn bench_strtod_long_ff(b: &mut Bencher) {
        bench_strtod_long(b, ff_strtod)
    }
    #[bench]
    fn bench_strtod_medium_ff(b: &mut Bencher) {
        bench_strtod_medium(b, ff_strtod)
    }
    #[bench]
    fn bench_strtod_short_ff(b: &mut Bencher) {
        bench_strtod_short(b, ff_strtod)
    }
}
