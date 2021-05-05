//! Implementation of substring searches.
//!
//! This is a tiny wrapper on top of `memmem::find` from the `memchr` crate.
use super::{Int, Str};
use memchr::memmem;

// 1-indexed, 0 on failure
pub fn index_substr<'a, 'b>(needle: &Str<'a>, haystack: &Str<'a>) -> Int {
    needle
        .with_bytes(|n| haystack.with_bytes(|h| memmem::find(h, n)))
        .map(|x| x as Int + 1)
        .unwrap_or(0)
}
