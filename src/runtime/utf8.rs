/// SIMD-accelerated UTF8 validation. A good clip faster in the ASCII fast-path,
/// and over 3x faster when validating non-ASCII UTF-8. Only x86 is supported for now, with
/// fallback to the standard libary string conversion routines if SSE2 is unavailable.
// TODO: support AVX2 or neon?
use std::str;

pub(crate) fn parse_utf8(bs: &[u8]) -> Option<&str> {
    if is_utf8(bs) {
        Some(unsafe { str::from_utf8_unchecked(bs) })
    } else {
        None
    }
}

fn validate_utf8_clipped(mut bs: &[u8]) -> Option<usize> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[inline]
        fn is_char_boundary(b: u8) -> bool {
            // Test if `b` is a character boundary, taken from the
            // str::is_char_boundary implementation in the standard
            // library.
            (b as i8) >= -0x40
        }
        if is_x86_feature_detected!("sse2") {
            // The SIMD implementation does not keep track of when a
            // string becomes invalid. That's important here because
            // `bs` could just be the prefix of a longer valid UTF8
            // string.  To allow for this we walk backwards through `bs`
            // to see if there's a potential incomplete character.
            let mut i = 0;
            for b in bs.iter().rev() {
                i += 1;
                if is_char_boundary(*b) {
                    break;
                }
                if i == 4 {
                    // We should have seen a char boundary after 4
                    // bytes, regardless of any clipping.
                    return None;
                }
            }

            if i > 0 && str::from_utf8(&bs[bs.len() - i..]).is_err() {
                bs = &bs[0..bs.len() - i];
            }

            let mut chunks = 0;
            const CHUNK_SIZE: usize = 1024;
            let valid = unsafe {
                // See comments in [is_utf8] for the strategy here re:
                // fast paths.
                if bs.len() >= 32 && x86::validate_ascii(&bs[0..32]) {
                    while bs.len() >= CHUNK_SIZE {
                        if !x86::validate_ascii(&bs[..CHUNK_SIZE]) {
                            break;
                        }
                        bs = &bs[CHUNK_SIZE..];
                        chunks += 1;
                    }
                }
                x86::validate_utf8(bs)
            };
            if valid {
                Some(chunks * CHUNK_SIZE + bs.len())
            } else {
                None
            }
        } else {
            validate_utf8_fallback(bs)
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        validate_utf8_fallback(bs)
    }
}

pub(crate) fn parse_utf8_clipped(bs: &[u8]) -> Option<&str> {
    validate_utf8_clipped(bs).map(|off| unsafe { str::from_utf8_unchecked(&bs[..off]) })
}

fn validate_utf8_fallback(bs: &[u8]) -> Option<usize> {
    match str::from_utf8(bs) {
        Ok(res) => Some(res.len()),
        Err(error) => {
            let last_valid = error.valid_up_to();
            if bs.len() - last_valid > 3 {
                None
            } else {
                Some(last_valid)
            }
        }
    }
}

pub(crate) fn is_utf8(mut bs: &[u8]) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("sse2") {
            unsafe {
                // We do a top-level fast path to speed up sequences with large all-ASCII prefixes
                // (in particular 100%-ASCII sequences), falling back to slower UTF8 validation if
                // ASCII validation fails. UTF8 validation itself has an ASCII fast path, so
                // sequences that are almost all ASCII will still see a relative speed-up.
                if bs.len() >= 32 && x86::validate_ascii(&bs[0..32]) {
                    const CHUNK_SIZE: usize = 1024;
                    while bs.len() >= CHUNK_SIZE {
                        if !x86::validate_ascii(&bs[..CHUNK_SIZE]) {
                            break;
                        }
                        bs = &bs[CHUNK_SIZE..];
                    }
                }
                x86::validate_utf8(bs)
            }
        } else {
            str::from_utf8(bs).is_ok()
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        str::from_utf8(bs).is_ok()
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use lazy_static::lazy_static;
    use test::{black_box, Bencher};

    fn parse_utf8_fallback(bs: &[u8]) -> Option<&str> {
        super::validate_utf8_fallback(bs)
            .map(|off| unsafe { std::str::from_utf8_unchecked(&bs[..off]) })
    }
    const LEN: usize = 50_000;

    lazy_static! {
        static ref ASCII: String = String::from_utf8(bytes(LEN, 0.0)).unwrap();
        static ref UTF8: String = String::from_utf8(bytes(LEN, 1.0)).unwrap();
    }

    #[test]
    fn test_partial() {
        let bs: Vec<_> = UTF8.as_bytes().iter().cloned().collect();
        let l = bs.len();
        let full = super::parse_utf8_clipped(&bs[..]).expect("full");
        assert_eq!(UTF8.as_str(), full);
        let partial = super::parse_utf8_clipped(&bs[..l - 1]).expect("partial (1)");
        let partial_fallback = parse_utf8_fallback(&bs[..l - 1]).expect("partial (2)");
        assert_eq!(partial, partial_fallback);
    }

    #[test]
    fn utf8_valid() {
        assert!(std::str::from_utf8(UTF8.as_bytes()).is_ok());
        assert!(super::is_utf8(UTF8.as_bytes()));
        // Corrupt it some
        let mut bs: Vec<_> = UTF8.as_bytes().iter().cloned().collect();
        let l = bs.len();
        assert!(std::str::from_utf8(&bs[..l - 1]).is_err());
        assert!(!super::is_utf8(&bs[..l - 1]));
        bs[l / 2] = 255;
        bs[l / 3] = 255;
        assert!(!super::is_utf8(&bs[..]));
        assert!(std::str::from_utf8(&bs[..]).is_err());
    }
    #[test]
    fn ascii_valid() {
        assert!(std::str::from_utf8(ASCII.as_bytes()).is_ok());
        assert!(super::is_utf8(ASCII.as_bytes()));
    }

    fn bytes(n: usize, utf8_pct: f64) -> Vec<u8> {
        let mut res = Vec::with_capacity(n);
        use rand::distributions::{Distribution, Uniform};
        let ascii = Uniform::new_inclusive(0u8, 127u8);
        let between = Uniform::new_inclusive(0.0, 1.0);
        let mut rng = rand::thread_rng();
        for _ in 0..n {
            if between.sample(&mut rng) < utf8_pct {
                let c = rand::random::<char>();
                let ix = res.len();
                for _ in 0..c.len_utf8() {
                    res.push(0);
                }
                c.encode_utf8(&mut res[ix..]);
            } else {
                res.push(ascii.sample(&mut rng))
            }
        }
        res
    }

    #[bench]
    fn parse_ascii_stdlib(b: &mut Bencher) {
        let bs = ASCII.as_bytes();
        b.iter(|| {
            black_box(std::str::from_utf8(bs).is_ok());
        })
    }

    #[bench]
    fn parse_ascii_simd(b: &mut Bencher) {
        let bs = ASCII.as_bytes();
        b.iter(|| {
            black_box(super::is_utf8(bs));
        })
    }

    #[bench]
    fn parse_100_utf8_simd(b: &mut Bencher) {
        let bs = bytes(LEN, 1.0);
        b.iter(|| {
            black_box(super::is_utf8(&bs[..]));
        })
    }

    #[bench]
    fn parse_100_utf8_stdlib(b: &mut Bencher) {
        let bs = bytes(LEN, 1.0);
        b.iter(|| {
            black_box(std::str::from_utf8(&bs[..]).is_ok());
        })
    }

    #[bench]
    fn parse_50_utf8_simd(b: &mut Bencher) {
        let bs = bytes(LEN, 0.5);
        b.iter(|| {
            black_box(super::is_utf8(&bs[..]));
        })
    }

    #[bench]
    fn parse_50_utf8_stdlib(b: &mut Bencher) {
        let bs = bytes(LEN, 0.5);
        b.iter(|| {
            black_box(std::str::from_utf8(&bs[..]).is_ok());
        })
    }

    #[bench]
    fn parse_10_utf8_simd(b: &mut Bencher) {
        let bs = bytes(LEN, 0.1);
        b.iter(|| {
            black_box(super::is_utf8(&bs[..]));
        })
    }

    #[bench]
    fn parse_10_utf8_stdlib(b: &mut Bencher) {
        let bs = bytes(LEN, 0.1);
        b.iter(|| {
            black_box(std::str::from_utf8(&bs[..]).is_ok());
        })
    }

    #[bench]
    fn parse_1_utf8_simd(b: &mut Bencher) {
        let bs = bytes(LEN, 0.01);
        b.iter(|| {
            black_box(super::is_utf8(&bs[..]));
        })
    }

    #[bench]
    fn parse_1_utf8_stdlib(b: &mut Bencher) {
        let bs = bytes(LEN, 0.01);
        b.iter(|| {
            black_box(std::str::from_utf8(&bs[..]).is_ok());
        })
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    // Most of this is a line-by-line translation of
    // https://github.com/lemire/fastvalidate-utf-8/blob/master/include/simdutf8check.h. But with
    // added notes gleaned from the code, as well as the simdjson paper:
    // https://arxiv.org/pdf/1902.08318.pdf
    //
    // TODO: add support for AVX2, using the same references.
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[inline]
    unsafe fn check_smaller_than_0xf4(current_bytes: __m128i, has_error: &mut __m128i) {
        // Intel doesn't define a byte-wise comparison instruction. We could load these byte values
        // into two vectors and use _mm_cmplt_epi16 to compare the two vectors against 0xf4, and
        // then merge those results back together, but a more efficient strategy is to use
        // byte-wise (unsigned) saturated subtraction (where (n-(n+k)) = 0). Subtracting 0xf4 and
        // or-ing the results will set the error vector iff a given byte is >= 0xf4.
        *has_error = _mm_or_si128(
            *has_error,
            // There's no _mm_set1_epu8, so we do the nested as business here.
            _mm_subs_epu8(current_bytes, _mm_set1_epi8(0xf4u8 as i8)),
        );
    }

    #[inline]
    unsafe fn continuation_lengths(high_nibbles: __m128i) -> __m128i {
        // The entries in high_nibbles are guaranteed to be in [0,15], so we can use the shuffle
        // instruction here as a lookup table. Nonzero entries indicate that the byte is at the
        // start of a character boundary with a character that requires that many bytes to
        // represent. Zero entries indicate nibbles that are in the middle of such a byte sequence.
        _mm_shuffle_epi8(
            _mm_setr_epi8(
                1, 1, 1, 1, 1, 1, 1, 1, // ASCII characters only require a single byte.
                0, 0, 0, 0, // Middle of a character.
                2, 2, 3, 4, // The start of 2,3, or 4-byte characters.
            ),
            high_nibbles,
        )
    }

    #[inline]
    unsafe fn carry_continuations(initial_lengths: __m128i, previous_carries: __m128i) -> __m128i {
        // This is an intermediate computation used to check if the byte sequence respects the
        // lengths computed in [continuation_lengths]. For example, if a particular character has
        // length 2, then the next character should have length 0, and the character after that
        // should have a nonzero length.
        //
        // This computation helps check this by shifting the lengths to the right by 1 and
        // subtracting 1, then 2/by 2, then 3/by 3 and summing all 4 of the vectors. The last two can
        // be done in one step by first summing the inital vector and the vector shifted by 1, and
        // then shifting that intermediate sum by 2.
        //
        //     initial = [4 0 0 0 3 0 0 1], previous = [2 1 2 1 4 3 2 1]
        // Logical:
        //     right1  = [0 3 0 0 0 2 0 0] (shift k, subtract by k)
        //     right2  = [0 0 2 0 0 0 1 0]
        //     right3  = [0 0 0 1 0 0 0 0]
        //     sum     = [4 3 2 1 3 2 1 1] (initial+right{1,2,3})
        // Actual:
        //     right1  = [0 3 0 0 0 2 0 0] (shift initial by 1, subtract 1)
        //     sum0    = [4 3 0 0 3 2 0 1] (initial + right1)
        //     right2  = [0 0 2 1 0 0 1 0] (shift sum0 by 2, subtract 2)
        //     sum     = [4 3 2 1 3 2 1 1] (sum0 + right2)
        //
        // The sum value is then fed to a verification computation, which checks that all of the
        // zeros have become nonzero, and that none of the nonzero entries have increased in size
        // (see [check_continuations]).
        let right1 = _mm_subs_epu8(
            _mm_alignr_epi8(initial_lengths, previous_carries, 16 - 1),
            _mm_set1_epi8(1),
        );
        let sum = _mm_add_epi8(initial_lengths, right1);
        let right2 = _mm_subs_epu8(
            _mm_alignr_epi8(sum, previous_carries, 16 - 2),
            _mm_set1_epi8(2),
        );
        _mm_add_epi8(sum, right2)
    }

    #[inline]
    unsafe fn check_continuations(
        initial_lengths: __m128i,
        carries: __m128i,
        has_error: &mut __m128i,
    ) {
        // We want to check that the sum returned in [carry_continuations] does not exceed the
        // input lengths, except where the original lengths were zero. This verifies that no new
        // characters "started too early". We also want to check that there are no zeros, otherwise
        // there would have been too many continuation tokens. We can do this in 3 comparisons by
        // checking testing that the carries only exceed the original lengths when the original
        // lengths were 0. The code inverts this (because we are setting an error flag):
        //
        // has_error ||= carries > length == lengths > 0
        let overunder = _mm_cmpeq_epi8(
            _mm_cmpgt_epi8(carries, initial_lengths),
            _mm_cmpgt_epi8(initial_lengths, _mm_setzero_si128()),
        );
        *has_error = _mm_or_si128(*has_error, overunder);
    }

    #[inline]
    unsafe fn check_first_continuation_max(
        current_bytes: __m128i,
        off1_current_bytes: __m128i,
        has_error: &mut __m128i,
    ) {
        // In UTF-8, 0xED cannot be followed by a byte larger than 0x9F. Similarly, 0xF4 cannot be
        // followed by a byte larger than 0x8f. We check for both of these by computing masks for
        // which bytes are 0xED(F4), and then ensuring all following indexes where this mask is
        // true are less than 0x9F(8F).
        let mask_ed = _mm_cmpeq_epi8(off1_current_bytes, _mm_set1_epi8(0xEDu8 as i8));
        let mask_f4 = _mm_cmpeq_epi8(off1_current_bytes, _mm_set1_epi8(0xF4u8 as i8));

        let bad_follow_ed = _mm_and_si128(
            _mm_cmpgt_epi8(current_bytes, _mm_set1_epi8(0x9Fu8 as i8)),
            mask_ed,
        );
        let bad_follow_f4 = _mm_and_si128(
            _mm_cmpgt_epi8(current_bytes, _mm_set1_epi8(0x8Fu8 as i8)),
            mask_f4,
        );

        *has_error = _mm_or_si128(*has_error, _mm_or_si128(bad_follow_ed, bad_follow_f4));
    }

    #[inline]
    unsafe fn check_overlong(
        current_bytes: __m128i,
        off1_current_bytes: __m128i,
        hibits: __m128i,
        previous_hibits: __m128i,
        has_error: &mut __m128i,
    ) {
        // This function checks a few more constraints on byte values.
        // * 0xC0 and 0xC1 are banned
        // * When a byte value is 0xE0, the next byte must be larger than 0xA0.
        // * When a byte value is 0xF0, the next byte must be at least 0x90.
        // Inverting these checks gives us the following table from the original code about the
        // relationshiop between the current high nibble, the high nibbles offset by 1, and the
        // current bytes:
        //
        // (table copied from the original code)
        // hibits     off1    cur
        // C       => < C2 && true
        // E       => < E1 && < A0
        // F       => < F1 && < 90
        // else      false && false
        //
        // Where the constraint being true means we have an error. To determine the position in the
        // first column (C,E,F,else) we use a similar lookup table to [continuation_lengths]
        // indexed by the high nibbles. This time, the contents of the table will be used in
        // comparisons, so hard-coded true and false values are given i8::max and i8::min,
        // respectively.

        let off1_hibits = _mm_alignr_epi8(hibits, previous_hibits, 16 - 1);
        const MIN: i8 = -128;
        const MAX: i8 = 127;
        let initial_mins = _mm_shuffle_epi8(
            _mm_setr_epi8(
                // 0 up to B have no constraints
                MIN,
                MIN,
                MIN,
                MIN,
                MIN,
                MIN,
                MIN,
                MIN,
                MIN,
                MIN,
                MIN,
                MIN,
                0xC2u8 as i8,
                // D has no constraints
                MIN,
                0xE1u8 as i8,
                0xF1u8 as i8,
            ),
            off1_hibits,
        );

        // Check if the current bytes shifted by 1 are <= the lower bounds we have.
        let initial_under = _mm_cmpgt_epi8(initial_mins, off1_current_bytes);

        // Now get lower bounds for the last ("cur") column:
        let second_mins = _mm_shuffle_epi8(
            _mm_setr_epi8(
                MIN,
                MIN,
                MIN,
                MIN,
                MIN,
                MIN,
                MIN,
                MIN,
                MIN,
                MIN,
                MIN,
                MIN,
                MAX,
                MAX,
                0xA0u8 as i8,
                0x90u8 as i8,
            ),
            off1_hibits,
        );
        let second_under = _mm_cmpgt_epi8(second_mins, current_bytes);
        // And the two masks together to get the errors.
        *has_error = _mm_or_si128(*has_error, _mm_and_si128(initial_under, second_under));
    }

    #[derive(Copy, Clone)]
    struct ProcessedUTF8Bytes {
        rawbytes: __m128i,
        high_nibbles: __m128i,
        carried_continuations: __m128i,
    }

    #[inline]
    unsafe fn check_utf8_bytes(
        current_bytes: __m128i,
        previous: &ProcessedUTF8Bytes,
        has_error: &mut __m128i,
    ) -> ProcessedUTF8Bytes {
        if _mm_testz_si128(current_bytes, _mm_set1_epi8(0x80u8 as i8)) != 0 {
            // This vector is all ASCII. Let's check to make sure there aren't any stray
            // continuations that went unfinished, and then reuse previous.
            //
            // The overhead of performing this check seems minimal (a few %) for data that is all
            // non-ASCII, but the gains are substantial for almost-all-ASCII inputs. For data that
            // is 100% ASCII there is additional short-circuiting done at the top level.
            *has_error = _mm_or_si128(
                _mm_cmpgt_epi8(
                    previous.carried_continuations,
                    _mm_setr_epi8(9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1),
                ),
                *has_error,
            );
            return *previous;
        }
        // We just want to shift all bytes right by 4, but there is no _mm_srli_epi8, so we emulate
        // it by shifting the 16-bit integers right and masking off the low nibble.
        let high_nibbles = _mm_and_si128(_mm_srli_epi16(current_bytes, 4), _mm_set1_epi8(0x0F));
        check_smaller_than_0xf4(current_bytes, has_error);
        let initial_lengths = continuation_lengths(high_nibbles);
        let carried_continuations =
            carry_continuations(initial_lengths, previous.carried_continuations);
        check_continuations(initial_lengths, carried_continuations, has_error);
        let off1_current_bytes = _mm_alignr_epi8(current_bytes, previous.rawbytes, 16 - 1);
        check_first_continuation_max(current_bytes, off1_current_bytes, has_error);
        check_overlong(
            current_bytes,
            off1_current_bytes,
            high_nibbles,
            previous.high_nibbles,
            has_error,
        );
        ProcessedUTF8Bytes {
            rawbytes: current_bytes,
            high_nibbles,
            carried_continuations,
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse2")]
    pub(crate) unsafe fn validate_utf8(src: &[u8]) -> bool {
        let base = src.as_ptr();
        let len = src.len() as isize;
        let mut has_error = _mm_setzero_si128();
        let mut previous = ProcessedUTF8Bytes {
            rawbytes: _mm_setzero_si128(),
            high_nibbles: _mm_setzero_si128(),
            carried_continuations: _mm_setzero_si128(),
        };
        let mut i = 0;
        // Loop over input in chunks of 16 bytes.
        if len >= 16 {
            loop {
                let current_bytes = _mm_loadu_si128(base.offset(i) as *const __m128i);
                previous = check_utf8_bytes(current_bytes, &previous, &mut has_error);
                i += 16;
                if i > len - 16 {
                    break;
                }
            }
        }

        if i < len {
            // Handle any leftovers by reading them into a stack-allocated buffer padded with
            // zeros.
            let mut buffer = [0u8; 16];
            std::ptr::copy_nonoverlapping(base.offset(i), buffer.as_mut_ptr(), (len - i) as usize);
            let current_bytes = _mm_loadu_si128(buffer.as_ptr() as *const __m128i);
            let _ = check_utf8_bytes(current_bytes, &previous, &mut has_error);
        } else {
            has_error = _mm_or_si128(
                has_error,
                // We need to make sure that the last carried continuation is okay. If the last
                // byte was the start of a two-byte character sequence, check_continuations would
                // not catch it (though it would in the next iteration). This one sets the error
                // vector if it was >1 (i.e it was not the last in a sequence). Note that we do not
                // need to do the same above because the padded zeros are ASCII, and there is at
                // least one byte of padding.
                _mm_cmpgt_epi8(
                    previous.carried_continuations,
                    _mm_setr_epi8(9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1),
                ),
            );
        }
        _mm_testz_si128(has_error, has_error) == 1
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse2")]
    pub(crate) unsafe fn validate_ascii(src: &[u8]) -> bool {
        // ASCII is much simpler to validate. This code simply ORs together all
        // of the bytes and checks if the MSB ever gets set.
        let base = src.as_ptr();
        let len = src.len() as isize;
        let mut has_error = _mm_setzero_si128();
        let mut i = 0;
        if len >= 16 {
            loop {
                let current_bytes = _mm_loadu_si128(base.offset(i) as *const __m128i);
                has_error = _mm_or_si128(has_error, current_bytes);
                i += 16;
                if i > len - 16 {
                    break;
                }
            }
        }
        let mut error_mask = _mm_movemask_epi8(has_error);
        let mut tail_has_error = 0u8;
        while i < len {
            tail_has_error |= *base.offset(i);
            i += 1;
        }
        error_mask |= (tail_has_error & 0x80) as i32;
        error_mask == 0
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn test_utf8() {
            unsafe {
                if is_x86_feature_detected!("sse2") {
                    assert!(validate_utf8(&[]));
                    assert!(validate_utf8("short ascii".as_bytes()));
                    const VIRGIL: &'static str = r"
Arms, and the man I sing, who, forc'd by fate,
And haughty Juno's unrelenting hate,
Expell'd and exil'd, left the Trojan shore.
Long labors, both by sea and land, he bore,
And in the doubtful war, before he won
The Latian realm, and built the destin'd town;
His banish'd gods restor'd to rites divine,
And settled sure succession in his line,
From whence the race of Alban fathers come,
And the long glories of majestic Rome.
O Muse! the causes and the crimes relate;
What goddess was provok'd, and whence her hate;
For what offense the Queen of Heav'n began
To persecute so brave, so just a man;
Involv'd his anxious life in endless cares,
Expos'd to wants, and hurried into wars!
Can heav'nly minds such high resentment show,
Or exercise their spite in human woe?";
                    assert!(validate_utf8(VIRGIL.as_bytes()));
                    assert!(validate_ascii(VIRGIL.as_bytes()));
                    // Selection from the Analects quoted from the Chinese Text Project
                    // https://ctext.org/analects/wei-zheng
                    assert!(validate_utf8( " 子張學干祿。子曰：「多聞闕疑，慎言其餘，則寡尤；多見闕殆，慎行其餘，則寡悔。言寡尤，行寡悔，祿在其中矣".as_bytes()));
                    assert!(!validate_utf8(&[64, 74, 255, 255, 255]));
                }
            }
        }
    }
}
