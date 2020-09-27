//! Implementation of substring searches.
//!
//! Back when frawk strings were guaranteed to be UTF8, we just used str::find from the standard
//! library. Now, we implement string searching in terms of memchr. For extremely long strings,
//! something like Teddy would probably work better, but this has fewer runtime requirements and
//! won't have much overhead for the common case use-caes of small strings.
use super::{Int, Str};
use memchr::{memchr, memchr_iter};

// 1-indexed, 0 on failure
pub fn index_substr<'a, 'b>(needle: &Str<'a>, haystack: &Str<'a>) -> Int {
    needle
        .with_bytes(|n| haystack.with_bytes(|h| index(n, h)))
        .map(|x| x as Int + 1)
        .unwrap_or(0)
}

fn index(needle: &[u8], haystack: &[u8]) -> Option<usize> {
    let (needle, first) = match needle.len() {
        0 => return Some(0),
        1 => return memchr(needle[0], haystack),
        _ => (&needle[1..], needle[0]),
    };
    for ix in memchr_iter(first, haystack) {
        let upto = needle.len() + ix + 1;
        if haystack.len() < upto {
            break;
        }
        if &haystack[ix + 1..upto] == needle {
            return Some(ix);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oracle(needle: &str, haystack: &str) {
        let expected = haystack.find(needle);
        let got = index(needle.as_bytes(), haystack.as_bytes());
        assert_eq!(
            got, expected,
            "searching for {:?} in {:?}, expected {:?}, got {:?}",
            needle, haystack, expected, got
        );
    }

    #[test]
    fn small_strings() {
        oracle("", "");
        oracle("", "any string");
        oracle("x", "pattern not found");
        oracle("x", "this string has an x in it");
        oracle("x", "this string has an x in it");
        oracle("ab", "asdfabcde as;dlfkjas ekdbd");
        oracle("ab", "ab");
        oracle("ab", "xxozoiuoba");
        oracle("jkl", "adfwqer;flkasjdfas;lk");
        oracle("jkl", "ajklqer;flkasjdfas;lk");
        oracle("jkl", "jkllqer;flkasjdfas;lk");
    }

    #[test]
    fn longer_strings() {
        oracle("hello", "I say there, hello and good day to you");
        oracle(
            "hello",
            "xadf;a asdfl;kasjdfasdlk;fjasdfas;alou ad/.,k'poqasewrq    ",
        );
        oracle(
            "hello",
            "helloa asdfl;kasjdfasdlk;fjasdfas;alou ad/.,k'poqasewrq    ",
        );
        oracle("medium length string", "medium length string");
        oracle("longer needle than", "haystack");
    }
}
