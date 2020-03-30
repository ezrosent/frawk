/// Fast float parser based on github.com/lemire/fast_double_parser, but adopted to support AWK
/// semantics (no failures, just 0s and stopping early).
use std::intrinsics::unlikely;
use std::mem;
mod slow_path;

// The simdjson repo has more optimizations to add for int parsing, but this is a big win over libc
// for the time being, if only because we do not have to copy `s` into a NUL-terminated
// representation.
pub fn strtoi(s: &str) -> i64 {
    let bs = s.as_bytes();
    if bs.len() == 0 {
        return 0;
    }
    let neg = bs[0] == '-' as u8;
    let off = if neg || bs[0] == '+' as u8 { 1 } else { 0 };
    let mut i = 0i64;
    for b in bs[off..].iter().cloned().take_while(|b| is_integer(*b)) {
        let digit = (b - '0' as u8) as i64;
        i = if let Some(i) = i.checked_mul(10).and_then(|i| i.checked_add(digit)) {
            i
        } else {
            // overflow
            return 0;
        }
    }
    if neg {
        -i
    } else {
        i
    }
}

// And now, for floats

#[derive(Copy, Clone)]
struct Explicit {
    mantissa: u64,
    exp: i32,
}

// We bail out for floats that are too small or too large. These numbers are powers of 10.
const FASTFLOAT_SMALLEST_POWER: i64 = -325;
const FASTFLOAT_LARGEST_POWER: i64 = 308;

const POWERS_OF_TEN: &[f64] = &[
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

// Makes a good effort at computing i * 10^power exactly. If it fails, it returns None.
// Assumes that power is within [FASTFLOAT_SMALLEST_POWER, FASTFLOAT_LARGEST_POWER], and that i is
// nonzero.
#[inline(always)]
fn compute_float_64(power: i64, mut i: u64, negative: bool) -> Option<f64> {
    debug_assert!(power >= FASTFLOAT_SMALLEST_POWER);
    debug_assert!(power <= FASTFLOAT_LARGEST_POWER);
    // Very fast path: we can fit in 53 bits. We can just dump i in an f64 and multiply.
    const MAX_MANTISSA: u64 = (1 << 53) - 1;
    if -22 <= power && power <= 22 && i <= MAX_MANTISSA {
        let mut d = i as f64;
        if power < 0 {
            d = d / unsafe { POWERS_OF_TEN.get_unchecked((-power) as usize) };
        } else {
            d = d * unsafe { POWERS_OF_TEN.get_unchecked(power as usize) };
        }
        if negative {
            d = -d;
        }
        return Some(d);
    }

    // Okay, so we weren't so lucky. The basic goal here is to take i and multiply it by 10^power
    // (keeping track of how many bits we've shifted off to add to the exponent later). If we can
    // perform this multiplication exactly, we can _usually_ just read off the bits we need for the
    // mantissa. We try this twice: once with 128 bits and once with 192 bits. If we still overflow
    // after using 192 bits we bail out. We also bail out if we detect that we are in "round to
    // even" territory.

    // The exponent is always rounded down with a leading 1.
    let Explicit {
        mantissa: factor_mantissa,
        exp: factor_exp,
    } = unsafe {
        // This is where we use the first two assumptions.
        *POWER_OF_TEN_COMPONENTS.get_unchecked((power - FASTFLOAT_SMALLEST_POWER) as usize)
    };
    // We want the MSB to be 1 here, because leading zeros get added to the exponent.
    let mut lz = i.leading_zeros();
    i = i.wrapping_shl(lz);

    // Now, compute the 128-bit product of the two mantissas, grab the most significant bits.
    let mut lower = i.wrapping_mul(factor_mantissa);
    let mut upper = ((i as u128 * factor_mantissa as u128) >> 64) as u64;

    // NB i and mantissa both have a leading 1, so upper is at least as big as (2^126)/(2^64)=2^62:
    // that means `upper` has at least 1 leading zero.
    // XXX not sure if that reasoning makes sense: couldn't it be larger?
    //
    // Check if the leading 55 bits are exact by looking for any 1s in the first 9 bits (plus part
    // of one bit after that).
    if unlikely((upper & 0x1FF) == 0x1FF && (lower.wrapping_add(i) < lower)) {
        // This should be unlikely: recompute these values with the full mantissa, using 192 bits.
        let factor_mantissa_low =
            *unsafe { MANTISSA_128.get_unchecked((power - FASTFLOAT_SMALLEST_POWER) as usize) };
        let product_low = i * factor_mantissa_low;
        let product_middle2 = ((i as u128).wrapping_mul(factor_mantissa_low as u128) >> 64) as u64;
        let product_middle1 = lower;
        let mut product_high = upper;
        let product_middle = product_middle1.wrapping_add(product_middle2);
        if product_middle < product_middle1 {
            // product_middle overflowed, add one to product_high.
            product_high = product_high.wrapping_add(1);
        }
        // After all that, if a similar overflow check sill fails (original source points to
        // mantissa * i + i), we bail out.
        if product_middle.wrapping_add(1) == 0
            && product_high & 0x1FF == 0x1FF
            && product_low.wrapping_add(i) < product_low
        {
            return None;
        }
        upper = product_high;
        lower = product_middle;
    }

    // Okay, we eventually want the mantissa to be 53 bits with a leading 1. Let's shift it to 54
    // bits with a leading 1 and figure out what to round:
    let upperbit = upper >> 63;
    let mut mantissa = upper >> (upperbit + 9);
    lz += 1 ^ (upperbit as u32);

    // We want to round "to even": which breaks ties in favor of even numbers if we are halfway
    // between two integers. This check bails out if it detects that we may need to do this: the
    // check here just looks for some trailing zeros, and a 1 that "goes away" after we multiply.
    if unlikely(lower == 0 && upper & 0x1FF == 0 && mantissa & 1 == 1) {
        return None;
    }
    // And now for the final bit
    mantissa += mantissa & 1;
    mantissa >>= 1;
    // Let's check for overflow
    if mantissa >= 1 << 53 {
        // add one to the exponent.
        mantissa = 1 << 52;
        lz -= 1;
    }
    // Now mask off the high bits: we'll mask in an exponent then return it.
    mantissa &= !(1 << 52);

    let real_exponent = factor_exp - lz as i32;
    // Check that the the real exponent is valid: values at the end of the range have special
    // meaning re: subnormals, NaNs, infinity.
    if unlikely(real_exponent < 1 || real_exponent > 2046) {
        return None;
    }
    mantissa |= (real_exponent as u64) << 52;
    mantissa |= (if negative { 1 } else { 0 }) << 63;
    Some(unsafe { mem::transmute::<u64, f64>(mantissa) })
}

#[allow(unreachable_code)]
pub fn strtod(s: &str) -> f64 {
    if s.len() == 0 {
        return 0.0;
    }
    let bs = s.as_bytes();
    let mut cur = bs;
    // We iterate differently, because our strings are not null-terminated. That means we check for
    // length more often. There are some more sophisticated unrolling techniques we could try, but
    // this code is pretty scary as it is!
    macro_rules! advance_or {
        ($e:expr) => {
            if cur.len() <= 1 {
                return $e;
            }
            cur = &cur[1..];
        };
    }
    macro_rules! cur {
        () => {{
            debug_assert!(cur.len() > 0);
            *cur.as_ptr()
        }};
    }

    #[inline(always)]
    fn may_overflow(mut start_digits: &[u8], mut digits: usize) -> bool {
        if unlikely(digits >= 19) {
            // We may overflow. The only way this doesn't happen is if we have a lot of
            // 0.0000000... action going on, (which floats are pretty good at representing).
            loop {
                if start_digits.len() > 0
                    && (start_digits[0] == '0' as u8 || start_digits[0] == '.' as u8)
                {
                    start_digits = &start_digits[1..];
                    digits = digits.saturating_sub(1);
                    continue;
                }
                break;
            }
            digits >= 19
        } else {
            false
        }
    }

    fn return_int_handle_overflow(
        u: u64,
        neg: bool,
        digits: usize,
        orig: &str,
        start_digits: &[u8],
    ) -> f64 {
        if may_overflow(start_digits, digits) {
            slow_path::strtod(orig)
        } else {
            let res = u as f64;
            if neg {
                -res
            } else {
                res
            }
        }
    }
    unsafe {
        let negative = cur!() == '-' as u8;
        if negative || cur!() == '+' as u8 {
            advance_or!(0.0);
        }
        if !is_integer(cur!()) {
            return 0.0;
        }
        let start_digits = cur;

        // The original code has a check for leading zeros at this point: we actually let these
        // pass as it is common for leading zeros to not affect the output in AWK implementations.
        let mut i = 0u64;
        loop {
            if !is_integer(cur!()) {
                break;
            }
            let digit = cur!() - '0' as u8;
            // This may overflow, but we will check for that later.
            i = i.wrapping_mul(10).wrapping_add(digit as u64);
            advance_or!(return_int_handle_overflow(
                i,
                negative,
                start_digits.len() - cur.len(),
                s,
                start_digits
            ));
        }
        let mut exponent = 0i64;
        if cur!() == '.' as u8 {
            advance_or!(return_int_handle_overflow(
                i,
                negative,
                start_digits.len() - cur.len() - 1,
                s,
                start_digits
            ));
            loop {
                if !is_integer(cur!()) {
                    break;
                }
                let digit = cur!() - '0' as u8;
                i = i.wrapping_mul(10).wrapping_add(digit as u64);
                exponent -= 1;
                advance_or!(break);
            }
        }
        let digit_count = start_digits.len() - cur.len() - 1;
        let mut exp_number = 0i64;

        if cur!() == 'e' as u8 || cur!() == 'E' as u8 {
            loop {
                advance_or!(break);
                let mut neg_exp = false;
                if cur!() == '-' as u8 {
                    neg_exp = true;
                    advance_or!(break);
                } else if cur!() == '+' as u8 {
                    advance_or!(break);
                }
                loop {
                    if !is_integer(cur!()) {
                        break;
                    }

                    let digit = cur!() - '0' as u8;
                    exp_number = exp_number.wrapping_mul(10).wrapping_add(digit as i64);
                    if exp_number > 0x100000000 {
                        // Yikes! That's a big exponent. Let's just defer to the slow path.
                        return slow_path::strtod(s);
                    }
                    advance_or!(break);
                }
                exponent += if neg_exp { -exp_number } else { exp_number };
                break;
            }
        }
        if unlikely(may_overflow(start_digits, digit_count)) {
            return slow_path::strtod(s);
        }
        if unlikely(exponent < FASTFLOAT_SMALLEST_POWER || exponent > FASTFLOAT_LARGEST_POWER) {
            return slow_path::strtod(s);
        }
        match compute_float_64(exponent, i, negative) {
            Some(f) => f,
            None => slow_path::strtod(s),
        }
    }
}

fn is_integer(c: u8) -> bool {
    (c >= '0' as u8) && (c <= '9' as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basic_behavior() {
        assert_eq!(strtod("1.234"), 1.234);
        assert_eq!(strtod("1.234hello"), 1.234);
        assert_eq!(strtod("1.234E70hello"), 1.234E70);
        assert_eq!(strtod("752834029324532"), 752834029324532.0);
        assert_eq!(strtod(""), 0.0);
        let imax = format!("{}", i64::max_value());
        let imin = format!("{}", i64::min_value());
        assert_eq!(strtod(imax.as_str()), i64::max_value() as f64);
        assert_eq!(strtod(imin.as_str()), i64::min_value() as f64);
    }
}

// What follows is a bunch of precomputed values for powers of 10; it's unlikely to be very
// interesting.

// Precomputed explicit representations for the powers of 10 going from 10^FASTFLOAT_SMALLEST_POWER
// to 10^FASTFLOAT_LARGEST_POWER. These basically correspond to subnormal values, which we do not
// try and parse (instead we fall back on `slow_path`). The mantissa here is truncated (not
// rounded).
#[rustfmt::skip]
const POWER_OF_TEN_COMPONENTS: &[Explicit] = &[
    Explicit { mantissa: 0xa5ced43b7e3e9188, exp: 7},
    Explicit { mantissa: 0xcf42894a5dce35ea, exp: 10},
    Explicit { mantissa: 0x818995ce7aa0e1b2, exp: 14},
    Explicit { mantissa: 0xa1ebfb4219491a1f, exp: 17},
    Explicit { mantissa: 0xca66fa129f9b60a6, exp: 20},
    Explicit { mantissa: 0xfd00b897478238d0, exp: 23},
    Explicit { mantissa: 0x9e20735e8cb16382, exp: 27},
    Explicit { mantissa: 0xc5a890362fddbc62, exp: 30},
    Explicit { mantissa: 0xf712b443bbd52b7b, exp: 33},
    Explicit { mantissa: 0x9a6bb0aa55653b2d, exp: 37},
    Explicit { mantissa: 0xc1069cd4eabe89f8, exp: 40},
    Explicit { mantissa: 0xf148440a256e2c76, exp: 43},
    Explicit { mantissa: 0x96cd2a865764dbca, exp: 47},
    Explicit { mantissa: 0xbc807527ed3e12bc, exp: 50},
    Explicit { mantissa: 0xeba09271e88d976b, exp: 53},
    Explicit { mantissa: 0x93445b8731587ea3, exp: 57},
    Explicit { mantissa: 0xb8157268fdae9e4c, exp: 60},
    Explicit { mantissa: 0xe61acf033d1a45df, exp: 63},
    Explicit { mantissa: 0x8fd0c16206306bab, exp: 67},
    Explicit { mantissa: 0xb3c4f1ba87bc8696, exp: 70},
    Explicit { mantissa: 0xe0b62e2929aba83c, exp: 73},
    Explicit { mantissa: 0x8c71dcd9ba0b4925, exp: 77},
    Explicit { mantissa: 0xaf8e5410288e1b6f, exp: 80},
    Explicit { mantissa: 0xdb71e91432b1a24a, exp: 83},
    Explicit { mantissa: 0x892731ac9faf056e, exp: 87},
    Explicit { mantissa: 0xab70fe17c79ac6ca, exp: 90},
    Explicit { mantissa: 0xd64d3d9db981787d, exp: 93},
    Explicit { mantissa: 0x85f0468293f0eb4e, exp: 97},
    Explicit { mantissa: 0xa76c582338ed2621, exp: 100},
    Explicit { mantissa: 0xd1476e2c07286faa, exp: 103},
    Explicit { mantissa: 0x82cca4db847945ca, exp: 107},
    Explicit { mantissa: 0xa37fce126597973c, exp: 110},
    Explicit { mantissa: 0xcc5fc196fefd7d0c, exp: 113},
    Explicit { mantissa: 0xff77b1fcbebcdc4f, exp: 116},
    Explicit { mantissa: 0x9faacf3df73609b1, exp: 120},
    Explicit { mantissa: 0xc795830d75038c1d, exp: 123},
    Explicit { mantissa: 0xf97ae3d0d2446f25, exp: 126},
    Explicit { mantissa: 0x9becce62836ac577, exp: 130},
    Explicit { mantissa: 0xc2e801fb244576d5, exp: 133},
    Explicit { mantissa: 0xf3a20279ed56d48a, exp: 136},
    Explicit { mantissa: 0x9845418c345644d6, exp: 140},
    Explicit { mantissa: 0xbe5691ef416bd60c, exp: 143},
    Explicit { mantissa: 0xedec366b11c6cb8f, exp: 146},
    Explicit { mantissa: 0x94b3a202eb1c3f39, exp: 150},
    Explicit { mantissa: 0xb9e08a83a5e34f07, exp: 153},
    Explicit { mantissa: 0xe858ad248f5c22c9, exp: 156},
    Explicit { mantissa: 0x91376c36d99995be, exp: 160},
    Explicit { mantissa: 0xb58547448ffffb2d, exp: 163},
    Explicit { mantissa: 0xe2e69915b3fff9f9, exp: 166},
    Explicit { mantissa: 0x8dd01fad907ffc3b, exp: 170},
    Explicit { mantissa: 0xb1442798f49ffb4a, exp: 173},
    Explicit { mantissa: 0xdd95317f31c7fa1d, exp: 176},
    Explicit { mantissa: 0x8a7d3eef7f1cfc52, exp: 180},
    Explicit { mantissa: 0xad1c8eab5ee43b66, exp: 183},
    Explicit { mantissa: 0xd863b256369d4a40, exp: 186},
    Explicit { mantissa: 0x873e4f75e2224e68, exp: 190},
    Explicit { mantissa: 0xa90de3535aaae202, exp: 193},
    Explicit { mantissa: 0xd3515c2831559a83, exp: 196},
    Explicit { mantissa: 0x8412d9991ed58091, exp: 200},
    Explicit { mantissa: 0xa5178fff668ae0b6, exp: 203},
    Explicit { mantissa: 0xce5d73ff402d98e3, exp: 206},
    Explicit { mantissa: 0x80fa687f881c7f8e, exp: 210},
    Explicit { mantissa: 0xa139029f6a239f72, exp: 213},
    Explicit { mantissa: 0xc987434744ac874e, exp: 216},
    Explicit { mantissa: 0xfbe9141915d7a922, exp: 219},
    Explicit { mantissa: 0x9d71ac8fada6c9b5, exp: 223},
    Explicit { mantissa: 0xc4ce17b399107c22, exp: 226},
    Explicit { mantissa: 0xf6019da07f549b2b, exp: 229},
    Explicit { mantissa: 0x99c102844f94e0fb, exp: 233},
    Explicit { mantissa: 0xc0314325637a1939, exp: 236},
    Explicit { mantissa: 0xf03d93eebc589f88, exp: 239},
    Explicit { mantissa: 0x96267c7535b763b5, exp: 243},
    Explicit { mantissa: 0xbbb01b9283253ca2, exp: 246},
    Explicit { mantissa: 0xea9c227723ee8bcb, exp: 249},
    Explicit { mantissa: 0x92a1958a7675175f, exp: 253},
    Explicit { mantissa: 0xb749faed14125d36, exp: 256},
    Explicit { mantissa: 0xe51c79a85916f484, exp: 259},
    Explicit { mantissa: 0x8f31cc0937ae58d2, exp: 263},
    Explicit { mantissa: 0xb2fe3f0b8599ef07, exp: 266},
    Explicit { mantissa: 0xdfbdcece67006ac9, exp: 269},
    Explicit { mantissa: 0x8bd6a141006042bd, exp: 273},
    Explicit { mantissa: 0xaecc49914078536d, exp: 276},
    Explicit { mantissa: 0xda7f5bf590966848, exp: 279},
    Explicit { mantissa: 0x888f99797a5e012d, exp: 283},
    Explicit { mantissa: 0xaab37fd7d8f58178, exp: 286},
    Explicit { mantissa: 0xd5605fcdcf32e1d6, exp: 289},
    Explicit { mantissa: 0x855c3be0a17fcd26, exp: 293},
    Explicit { mantissa: 0xa6b34ad8c9dfc06f, exp: 296},
    Explicit { mantissa: 0xd0601d8efc57b08b, exp: 299},
    Explicit { mantissa: 0x823c12795db6ce57, exp: 303},
    Explicit { mantissa: 0xa2cb1717b52481ed, exp: 306},
    Explicit { mantissa: 0xcb7ddcdda26da268, exp: 309},
    Explicit { mantissa: 0xfe5d54150b090b02, exp: 312},
    Explicit { mantissa: 0x9efa548d26e5a6e1, exp: 316},
    Explicit { mantissa: 0xc6b8e9b0709f109a, exp: 319},
    Explicit { mantissa: 0xf867241c8cc6d4c0, exp: 322},
    Explicit { mantissa: 0x9b407691d7fc44f8, exp: 326},
    Explicit { mantissa: 0xc21094364dfb5636, exp: 329},
    Explicit { mantissa: 0xf294b943e17a2bc4, exp: 332},
    Explicit { mantissa: 0x979cf3ca6cec5b5a, exp: 336},
    Explicit { mantissa: 0xbd8430bd08277231, exp: 339},
    Explicit { mantissa: 0xece53cec4a314ebd, exp: 342},
    Explicit { mantissa: 0x940f4613ae5ed136, exp: 346},
    Explicit { mantissa: 0xb913179899f68584, exp: 349},
    Explicit { mantissa: 0xe757dd7ec07426e5, exp: 352},
    Explicit { mantissa: 0x9096ea6f3848984f, exp: 356},
    Explicit { mantissa: 0xb4bca50b065abe63, exp: 359},
    Explicit { mantissa: 0xe1ebce4dc7f16dfb, exp: 362},
    Explicit { mantissa: 0x8d3360f09cf6e4bd, exp: 366},
    Explicit { mantissa: 0xb080392cc4349dec, exp: 369},
    Explicit { mantissa: 0xdca04777f541c567, exp: 372},
    Explicit { mantissa: 0x89e42caaf9491b60, exp: 376},
    Explicit { mantissa: 0xac5d37d5b79b6239, exp: 379},
    Explicit { mantissa: 0xd77485cb25823ac7, exp: 382},
    Explicit { mantissa: 0x86a8d39ef77164bc, exp: 386},
    Explicit { mantissa: 0xa8530886b54dbdeb, exp: 389},
    Explicit { mantissa: 0xd267caa862a12d66, exp: 392},
    Explicit { mantissa: 0x8380dea93da4bc60, exp: 396},
    Explicit { mantissa: 0xa46116538d0deb78, exp: 399},
    Explicit { mantissa: 0xcd795be870516656, exp: 402},
    Explicit { mantissa: 0x806bd9714632dff6, exp: 406},
    Explicit { mantissa: 0xa086cfcd97bf97f3, exp: 409},
    Explicit { mantissa: 0xc8a883c0fdaf7df0, exp: 412},
    Explicit { mantissa: 0xfad2a4b13d1b5d6c, exp: 415},
    Explicit { mantissa: 0x9cc3a6eec6311a63, exp: 419},
    Explicit { mantissa: 0xc3f490aa77bd60fc, exp: 422},
    Explicit { mantissa: 0xf4f1b4d515acb93b, exp: 425},
    Explicit { mantissa: 0x991711052d8bf3c5, exp: 429},
    Explicit { mantissa: 0xbf5cd54678eef0b6, exp: 432},
    Explicit { mantissa: 0xef340a98172aace4, exp: 435},
    Explicit { mantissa: 0x9580869f0e7aac0e, exp: 439},
    Explicit { mantissa: 0xbae0a846d2195712, exp: 442},
    Explicit { mantissa: 0xe998d258869facd7, exp: 445},
    Explicit { mantissa: 0x91ff83775423cc06, exp: 449},
    Explicit { mantissa: 0xb67f6455292cbf08, exp: 452},
    Explicit { mantissa: 0xe41f3d6a7377eeca, exp: 455},
    Explicit { mantissa: 0x8e938662882af53e, exp: 459},
    Explicit { mantissa: 0xb23867fb2a35b28d, exp: 462},
    Explicit { mantissa: 0xdec681f9f4c31f31, exp: 465},
    Explicit { mantissa: 0x8b3c113c38f9f37e, exp: 469},
    Explicit { mantissa: 0xae0b158b4738705e, exp: 472},
    Explicit { mantissa: 0xd98ddaee19068c76, exp: 475},
    Explicit { mantissa: 0x87f8a8d4cfa417c9, exp: 479},
    Explicit { mantissa: 0xa9f6d30a038d1dbc, exp: 482},
    Explicit { mantissa: 0xd47487cc8470652b, exp: 485},
    Explicit { mantissa: 0x84c8d4dfd2c63f3b, exp: 489},
    Explicit { mantissa: 0xa5fb0a17c777cf09, exp: 492},
    Explicit { mantissa: 0xcf79cc9db955c2cc, exp: 495},
    Explicit { mantissa: 0x81ac1fe293d599bf, exp: 499},
    Explicit { mantissa: 0xa21727db38cb002f, exp: 502},
    Explicit { mantissa: 0xca9cf1d206fdc03b, exp: 505},
    Explicit { mantissa: 0xfd442e4688bd304a, exp: 508},
    Explicit { mantissa: 0x9e4a9cec15763e2e, exp: 512},
    Explicit { mantissa: 0xc5dd44271ad3cdba, exp: 515},
    Explicit { mantissa: 0xf7549530e188c128, exp: 518},
    Explicit { mantissa: 0x9a94dd3e8cf578b9, exp: 522},
    Explicit { mantissa: 0xc13a148e3032d6e7, exp: 525},
    Explicit { mantissa: 0xf18899b1bc3f8ca1, exp: 528},
    Explicit { mantissa: 0x96f5600f15a7b7e5, exp: 532},
    Explicit { mantissa: 0xbcb2b812db11a5de, exp: 535},
    Explicit { mantissa: 0xebdf661791d60f56, exp: 538},
    Explicit { mantissa: 0x936b9fcebb25c995, exp: 542},
    Explicit { mantissa: 0xb84687c269ef3bfb, exp: 545},
    Explicit { mantissa: 0xe65829b3046b0afa, exp: 548},
    Explicit { mantissa: 0x8ff71a0fe2c2e6dc, exp: 552},
    Explicit { mantissa: 0xb3f4e093db73a093, exp: 555},
    Explicit { mantissa: 0xe0f218b8d25088b8, exp: 558},
    Explicit { mantissa: 0x8c974f7383725573, exp: 562},
    Explicit { mantissa: 0xafbd2350644eeacf, exp: 565},
    Explicit { mantissa: 0xdbac6c247d62a583, exp: 568},
    Explicit { mantissa: 0x894bc396ce5da772, exp: 572},
    Explicit { mantissa: 0xab9eb47c81f5114f, exp: 575},
    Explicit { mantissa: 0xd686619ba27255a2, exp: 578},
    Explicit { mantissa: 0x8613fd0145877585, exp: 582},
    Explicit { mantissa: 0xa798fc4196e952e7, exp: 585},
    Explicit { mantissa: 0xd17f3b51fca3a7a0, exp: 588},
    Explicit { mantissa: 0x82ef85133de648c4, exp: 592},
    Explicit { mantissa: 0xa3ab66580d5fdaf5, exp: 595},
    Explicit { mantissa: 0xcc963fee10b7d1b3, exp: 598},
    Explicit { mantissa: 0xffbbcfe994e5c61f, exp: 601},
    Explicit { mantissa: 0x9fd561f1fd0f9bd3, exp: 605},
    Explicit { mantissa: 0xc7caba6e7c5382c8, exp: 608},
    Explicit { mantissa: 0xf9bd690a1b68637b, exp: 611},
    Explicit { mantissa: 0x9c1661a651213e2d, exp: 615},
    Explicit { mantissa: 0xc31bfa0fe5698db8, exp: 618},
    Explicit { mantissa: 0xf3e2f893dec3f126, exp: 621},
    Explicit { mantissa: 0x986ddb5c6b3a76b7, exp: 625},
    Explicit { mantissa: 0xbe89523386091465, exp: 628},
    Explicit { mantissa: 0xee2ba6c0678b597f, exp: 631},
    Explicit { mantissa: 0x94db483840b717ef, exp: 635},
    Explicit { mantissa: 0xba121a4650e4ddeb, exp: 638},
    Explicit { mantissa: 0xe896a0d7e51e1566, exp: 641},
    Explicit { mantissa: 0x915e2486ef32cd60, exp: 645},
    Explicit { mantissa: 0xb5b5ada8aaff80b8, exp: 648},
    Explicit { mantissa: 0xe3231912d5bf60e6, exp: 651},
    Explicit { mantissa: 0x8df5efabc5979c8f, exp: 655},
    Explicit { mantissa: 0xb1736b96b6fd83b3, exp: 658},
    Explicit { mantissa: 0xddd0467c64bce4a0, exp: 661},
    Explicit { mantissa: 0x8aa22c0dbef60ee4, exp: 665},
    Explicit { mantissa: 0xad4ab7112eb3929d, exp: 668},
    Explicit { mantissa: 0xd89d64d57a607744, exp: 671},
    Explicit { mantissa: 0x87625f056c7c4a8b, exp: 675},
    Explicit { mantissa: 0xa93af6c6c79b5d2d, exp: 678},
    Explicit { mantissa: 0xd389b47879823479, exp: 681},
    Explicit { mantissa: 0x843610cb4bf160cb, exp: 685},
    Explicit { mantissa: 0xa54394fe1eedb8fe, exp: 688},
    Explicit { mantissa: 0xce947a3da6a9273e, exp: 691},
    Explicit { mantissa: 0x811ccc668829b887, exp: 695},
    Explicit { mantissa: 0xa163ff802a3426a8, exp: 698},
    Explicit { mantissa: 0xc9bcff6034c13052, exp: 701},
    Explicit { mantissa: 0xfc2c3f3841f17c67, exp: 704},
    Explicit { mantissa: 0x9d9ba7832936edc0, exp: 708},
    Explicit { mantissa: 0xc5029163f384a931, exp: 711},
    Explicit { mantissa: 0xf64335bcf065d37d, exp: 714},
    Explicit { mantissa: 0x99ea0196163fa42e, exp: 718},
    Explicit { mantissa: 0xc06481fb9bcf8d39, exp: 721},
    Explicit { mantissa: 0xf07da27a82c37088, exp: 724},
    Explicit { mantissa: 0x964e858c91ba2655, exp: 728},
    Explicit { mantissa: 0xbbe226efb628afea, exp: 731},
    Explicit { mantissa: 0xeadab0aba3b2dbe5, exp: 734},
    Explicit { mantissa: 0x92c8ae6b464fc96f, exp: 738},
    Explicit { mantissa: 0xb77ada0617e3bbcb, exp: 741},
    Explicit { mantissa: 0xe55990879ddcaabd, exp: 744},
    Explicit { mantissa: 0x8f57fa54c2a9eab6, exp: 748},
    Explicit { mantissa: 0xb32df8e9f3546564, exp: 751},
    Explicit { mantissa: 0xdff9772470297ebd, exp: 754},
    Explicit { mantissa: 0x8bfbea76c619ef36, exp: 758},
    Explicit { mantissa: 0xaefae51477a06b03, exp: 761},
    Explicit { mantissa: 0xdab99e59958885c4, exp: 764},
    Explicit { mantissa: 0x88b402f7fd75539b, exp: 768},
    Explicit { mantissa: 0xaae103b5fcd2a881, exp: 771},
    Explicit { mantissa: 0xd59944a37c0752a2, exp: 774},
    Explicit { mantissa: 0x857fcae62d8493a5, exp: 778},
    Explicit { mantissa: 0xa6dfbd9fb8e5b88e, exp: 781},
    Explicit { mantissa: 0xd097ad07a71f26b2, exp: 784},
    Explicit { mantissa: 0x825ecc24c873782f, exp: 788},
    Explicit { mantissa: 0xa2f67f2dfa90563b, exp: 791},
    Explicit { mantissa: 0xcbb41ef979346bca, exp: 794},
    Explicit { mantissa: 0xfea126b7d78186bc, exp: 797},
    Explicit { mantissa: 0x9f24b832e6b0f436, exp: 801},
    Explicit { mantissa: 0xc6ede63fa05d3143, exp: 804},
    Explicit { mantissa: 0xf8a95fcf88747d94, exp: 807},
    Explicit { mantissa: 0x9b69dbe1b548ce7c, exp: 811},
    Explicit { mantissa: 0xc24452da229b021b, exp: 814},
    Explicit { mantissa: 0xf2d56790ab41c2a2, exp: 817},
    Explicit { mantissa: 0x97c560ba6b0919a5, exp: 821},
    Explicit { mantissa: 0xbdb6b8e905cb600f, exp: 824},
    Explicit { mantissa: 0xed246723473e3813, exp: 827},
    Explicit { mantissa: 0x9436c0760c86e30b, exp: 831},
    Explicit { mantissa: 0xb94470938fa89bce, exp: 834},
    Explicit { mantissa: 0xe7958cb87392c2c2, exp: 837},
    Explicit { mantissa: 0x90bd77f3483bb9b9, exp: 841},
    Explicit { mantissa: 0xb4ecd5f01a4aa828, exp: 844},
    Explicit { mantissa: 0xe2280b6c20dd5232, exp: 847},
    Explicit { mantissa: 0x8d590723948a535f, exp: 851},
    Explicit { mantissa: 0xb0af48ec79ace837, exp: 854},
    Explicit { mantissa: 0xdcdb1b2798182244, exp: 857},
    Explicit { mantissa: 0x8a08f0f8bf0f156b, exp: 861},
    Explicit { mantissa: 0xac8b2d36eed2dac5, exp: 864},
    Explicit { mantissa: 0xd7adf884aa879177, exp: 867},
    Explicit { mantissa: 0x86ccbb52ea94baea, exp: 871},
    Explicit { mantissa: 0xa87fea27a539e9a5, exp: 874},
    Explicit { mantissa: 0xd29fe4b18e88640e, exp: 877},
    Explicit { mantissa: 0x83a3eeeef9153e89, exp: 881},
    Explicit { mantissa: 0xa48ceaaab75a8e2b, exp: 884},
    Explicit { mantissa: 0xcdb02555653131b6, exp: 887},
    Explicit { mantissa: 0x808e17555f3ebf11, exp: 891},
    Explicit { mantissa: 0xa0b19d2ab70e6ed6, exp: 894},
    Explicit { mantissa: 0xc8de047564d20a8b, exp: 897},
    Explicit { mantissa: 0xfb158592be068d2e, exp: 900},
    Explicit { mantissa: 0x9ced737bb6c4183d, exp: 904},
    Explicit { mantissa: 0xc428d05aa4751e4c, exp: 907},
    Explicit { mantissa: 0xf53304714d9265df, exp: 910},
    Explicit { mantissa: 0x993fe2c6d07b7fab, exp: 914},
    Explicit { mantissa: 0xbf8fdb78849a5f96, exp: 917},
    Explicit { mantissa: 0xef73d256a5c0f77c, exp: 920},
    Explicit { mantissa: 0x95a8637627989aad, exp: 924},
    Explicit { mantissa: 0xbb127c53b17ec159, exp: 927},
    Explicit { mantissa: 0xe9d71b689dde71af, exp: 930},
    Explicit { mantissa: 0x9226712162ab070d, exp: 934},
    Explicit { mantissa: 0xb6b00d69bb55c8d1, exp: 937},
    Explicit { mantissa: 0xe45c10c42a2b3b05, exp: 940},
    Explicit { mantissa: 0x8eb98a7a9a5b04e3, exp: 944},
    Explicit { mantissa: 0xb267ed1940f1c61c, exp: 947},
    Explicit { mantissa: 0xdf01e85f912e37a3, exp: 950},
    Explicit { mantissa: 0x8b61313bbabce2c6, exp: 954},
    Explicit { mantissa: 0xae397d8aa96c1b77, exp: 957},
    Explicit { mantissa: 0xd9c7dced53c72255, exp: 960},
    Explicit { mantissa: 0x881cea14545c7575, exp: 964},
    Explicit { mantissa: 0xaa242499697392d2, exp: 967},
    Explicit { mantissa: 0xd4ad2dbfc3d07787, exp: 970},
    Explicit { mantissa: 0x84ec3c97da624ab4, exp: 974},
    Explicit { mantissa: 0xa6274bbdd0fadd61, exp: 977},
    Explicit { mantissa: 0xcfb11ead453994ba, exp: 980},
    Explicit { mantissa: 0x81ceb32c4b43fcf4, exp: 984},
    Explicit { mantissa: 0xa2425ff75e14fc31, exp: 987},
    Explicit { mantissa: 0xcad2f7f5359a3b3e, exp: 990},
    Explicit { mantissa: 0xfd87b5f28300ca0d, exp: 993},
    Explicit { mantissa: 0x9e74d1b791e07e48, exp: 997},
    Explicit { mantissa: 0xc612062576589dda, exp: 1000},
    Explicit { mantissa: 0xf79687aed3eec551, exp: 1003},
    Explicit { mantissa: 0x9abe14cd44753b52, exp: 1007},
    Explicit { mantissa: 0xc16d9a0095928a27, exp: 1010},
    Explicit { mantissa: 0xf1c90080baf72cb1, exp: 1013},
    Explicit { mantissa: 0x971da05074da7bee, exp: 1017},
    Explicit { mantissa: 0xbce5086492111aea, exp: 1020},
    Explicit { mantissa: 0xec1e4a7db69561a5, exp: 1023},
    Explicit { mantissa: 0x9392ee8e921d5d07, exp: 1027},
    Explicit { mantissa: 0xb877aa3236a4b449, exp: 1030},
    Explicit { mantissa: 0xe69594bec44de15b, exp: 1033},
    Explicit { mantissa: 0x901d7cf73ab0acd9, exp: 1037},
    Explicit { mantissa: 0xb424dc35095cd80f, exp: 1040},
    Explicit { mantissa: 0xe12e13424bb40e13, exp: 1043},
    Explicit { mantissa: 0x8cbccc096f5088cb, exp: 1047},
    Explicit { mantissa: 0xafebff0bcb24aafe, exp: 1050},
    Explicit { mantissa: 0xdbe6fecebdedd5be, exp: 1053},
    Explicit { mantissa: 0x89705f4136b4a597, exp: 1057},
    Explicit { mantissa: 0xabcc77118461cefc, exp: 1060},
    Explicit { mantissa: 0xd6bf94d5e57a42bc, exp: 1063},
    Explicit { mantissa: 0x8637bd05af6c69b5, exp: 1067},
    Explicit { mantissa: 0xa7c5ac471b478423, exp: 1070},
    Explicit { mantissa: 0xd1b71758e219652b, exp: 1073},
    Explicit { mantissa: 0x83126e978d4fdf3b, exp: 1077},
    Explicit { mantissa: 0xa3d70a3d70a3d70a, exp: 1080},
    Explicit { mantissa: 0xcccccccccccccccc, exp: 1083},
    Explicit { mantissa: 0x8000000000000000, exp: 1087},
    Explicit { mantissa: 0xa000000000000000, exp: 1090},
    Explicit { mantissa: 0xc800000000000000, exp: 1093},
    Explicit { mantissa: 0xfa00000000000000, exp: 1096},
    Explicit { mantissa: 0x9c40000000000000, exp: 1100},
    Explicit { mantissa: 0xc350000000000000, exp: 1103},
    Explicit { mantissa: 0xf424000000000000, exp: 1106},
    Explicit { mantissa: 0x9896800000000000, exp: 1110},
    Explicit { mantissa: 0xbebc200000000000, exp: 1113},
    Explicit { mantissa: 0xee6b280000000000, exp: 1116},
    Explicit { mantissa: 0x9502f90000000000, exp: 1120},
    Explicit { mantissa: 0xba43b74000000000, exp: 1123},
    Explicit { mantissa: 0xe8d4a51000000000, exp: 1126},
    Explicit { mantissa: 0x9184e72a00000000, exp: 1130},
    Explicit { mantissa: 0xb5e620f480000000, exp: 1133},
    Explicit { mantissa: 0xe35fa931a0000000, exp: 1136},
    Explicit { mantissa: 0x8e1bc9bf04000000, exp: 1140},
    Explicit { mantissa: 0xb1a2bc2ec5000000, exp: 1143},
    Explicit { mantissa: 0xde0b6b3a76400000, exp: 1146},
    Explicit { mantissa: 0x8ac7230489e80000, exp: 1150},
    Explicit { mantissa: 0xad78ebc5ac620000, exp: 1153},
    Explicit { mantissa: 0xd8d726b7177a8000, exp: 1156},
    Explicit { mantissa: 0x878678326eac9000, exp: 1160},
    Explicit { mantissa: 0xa968163f0a57b400, exp: 1163},
    Explicit { mantissa: 0xd3c21bcecceda100, exp: 1166},
    Explicit { mantissa: 0x84595161401484a0, exp: 1170},
    Explicit { mantissa: 0xa56fa5b99019a5c8, exp: 1173},
    Explicit { mantissa: 0xcecb8f27f4200f3a, exp: 1176},
    Explicit { mantissa: 0x813f3978f8940984, exp: 1180},
    Explicit { mantissa: 0xa18f07d736b90be5, exp: 1183},
    Explicit { mantissa: 0xc9f2c9cd04674ede, exp: 1186},
    Explicit { mantissa: 0xfc6f7c4045812296, exp: 1189},
    Explicit { mantissa: 0x9dc5ada82b70b59d, exp: 1193},
    Explicit { mantissa: 0xc5371912364ce305, exp: 1196},
    Explicit { mantissa: 0xf684df56c3e01bc6, exp: 1199},
    Explicit { mantissa: 0x9a130b963a6c115c, exp: 1203},
    Explicit { mantissa: 0xc097ce7bc90715b3, exp: 1206},
    Explicit { mantissa: 0xf0bdc21abb48db20, exp: 1209},
    Explicit { mantissa: 0x96769950b50d88f4, exp: 1213},
    Explicit { mantissa: 0xbc143fa4e250eb31, exp: 1216},
    Explicit { mantissa: 0xeb194f8e1ae525fd, exp: 1219},
    Explicit { mantissa: 0x92efd1b8d0cf37be, exp: 1223},
    Explicit { mantissa: 0xb7abc627050305ad, exp: 1226},
    Explicit { mantissa: 0xe596b7b0c643c719, exp: 1229},
    Explicit { mantissa: 0x8f7e32ce7bea5c6f, exp: 1233},
    Explicit { mantissa: 0xb35dbf821ae4f38b, exp: 1236},
    Explicit { mantissa: 0xe0352f62a19e306e, exp: 1239},
    Explicit { mantissa: 0x8c213d9da502de45, exp: 1243},
    Explicit { mantissa: 0xaf298d050e4395d6, exp: 1246},
    Explicit { mantissa: 0xdaf3f04651d47b4c, exp: 1249},
    Explicit { mantissa: 0x88d8762bf324cd0f, exp: 1253},
    Explicit { mantissa: 0xab0e93b6efee0053, exp: 1256},
    Explicit { mantissa: 0xd5d238a4abe98068, exp: 1259},
    Explicit { mantissa: 0x85a36366eb71f041, exp: 1263},
    Explicit { mantissa: 0xa70c3c40a64e6c51, exp: 1266},
    Explicit { mantissa: 0xd0cf4b50cfe20765, exp: 1269},
    Explicit { mantissa: 0x82818f1281ed449f, exp: 1273},
    Explicit { mantissa: 0xa321f2d7226895c7, exp: 1276},
    Explicit { mantissa: 0xcbea6f8ceb02bb39, exp: 1279},
    Explicit { mantissa: 0xfee50b7025c36a08, exp: 1282},
    Explicit { mantissa: 0x9f4f2726179a2245, exp: 1286},
    Explicit { mantissa: 0xc722f0ef9d80aad6, exp: 1289},
    Explicit { mantissa: 0xf8ebad2b84e0d58b, exp: 1292},
    Explicit { mantissa: 0x9b934c3b330c8577, exp: 1296},
    Explicit { mantissa: 0xc2781f49ffcfa6d5, exp: 1299},
    Explicit { mantissa: 0xf316271c7fc3908a, exp: 1302},
    Explicit { mantissa: 0x97edd871cfda3a56, exp: 1306},
    Explicit { mantissa: 0xbde94e8e43d0c8ec, exp: 1309},
    Explicit { mantissa: 0xed63a231d4c4fb27, exp: 1312},
    Explicit { mantissa: 0x945e455f24fb1cf8, exp: 1316},
    Explicit { mantissa: 0xb975d6b6ee39e436, exp: 1319},
    Explicit { mantissa: 0xe7d34c64a9c85d44, exp: 1322},
    Explicit { mantissa: 0x90e40fbeea1d3a4a, exp: 1326},
    Explicit { mantissa: 0xb51d13aea4a488dd, exp: 1329},
    Explicit { mantissa: 0xe264589a4dcdab14, exp: 1332},
    Explicit { mantissa: 0x8d7eb76070a08aec, exp: 1336},
    Explicit { mantissa: 0xb0de65388cc8ada8, exp: 1339},
    Explicit { mantissa: 0xdd15fe86affad912, exp: 1342},
    Explicit { mantissa: 0x8a2dbf142dfcc7ab, exp: 1346},
    Explicit { mantissa: 0xacb92ed9397bf996, exp: 1349},
    Explicit { mantissa: 0xd7e77a8f87daf7fb, exp: 1352},
    Explicit { mantissa: 0x86f0ac99b4e8dafd, exp: 1356},
    Explicit { mantissa: 0xa8acd7c0222311bc, exp: 1359},
    Explicit { mantissa: 0xd2d80db02aabd62b, exp: 1362},
    Explicit { mantissa: 0x83c7088e1aab65db, exp: 1366},
    Explicit { mantissa: 0xa4b8cab1a1563f52, exp: 1369},
    Explicit { mantissa: 0xcde6fd5e09abcf26, exp: 1372},
    Explicit { mantissa: 0x80b05e5ac60b6178, exp: 1376},
    Explicit { mantissa: 0xa0dc75f1778e39d6, exp: 1379},
    Explicit { mantissa: 0xc913936dd571c84c, exp: 1382},
    Explicit { mantissa: 0xfb5878494ace3a5f, exp: 1385},
    Explicit { mantissa: 0x9d174b2dcec0e47b, exp: 1389},
    Explicit { mantissa: 0xc45d1df942711d9a, exp: 1392},
    Explicit { mantissa: 0xf5746577930d6500, exp: 1395},
    Explicit { mantissa: 0x9968bf6abbe85f20, exp: 1399},
    Explicit { mantissa: 0xbfc2ef456ae276e8, exp: 1402},
    Explicit { mantissa: 0xefb3ab16c59b14a2, exp: 1405},
    Explicit { mantissa: 0x95d04aee3b80ece5, exp: 1409},
    Explicit { mantissa: 0xbb445da9ca61281f, exp: 1412},
    Explicit { mantissa: 0xea1575143cf97226, exp: 1415},
    Explicit { mantissa: 0x924d692ca61be758, exp: 1419},
    Explicit { mantissa: 0xb6e0c377cfa2e12e, exp: 1422},
    Explicit { mantissa: 0xe498f455c38b997a, exp: 1425},
    Explicit { mantissa: 0x8edf98b59a373fec, exp: 1429},
    Explicit { mantissa: 0xb2977ee300c50fe7, exp: 1432},
    Explicit { mantissa: 0xdf3d5e9bc0f653e1, exp: 1435},
    Explicit { mantissa: 0x8b865b215899f46c, exp: 1439},
    Explicit { mantissa: 0xae67f1e9aec07187, exp: 1442},
    Explicit { mantissa: 0xda01ee641a708de9, exp: 1445},
    Explicit { mantissa: 0x884134fe908658b2, exp: 1449},
    Explicit { mantissa: 0xaa51823e34a7eede, exp: 1452},
    Explicit { mantissa: 0xd4e5e2cdc1d1ea96, exp: 1455},
    Explicit { mantissa: 0x850fadc09923329e, exp: 1459},
    Explicit { mantissa: 0xa6539930bf6bff45, exp: 1462},
    Explicit { mantissa: 0xcfe87f7cef46ff16, exp: 1465},
    Explicit { mantissa: 0x81f14fae158c5f6e, exp: 1469},
    Explicit { mantissa: 0xa26da3999aef7749, exp: 1472},
    Explicit { mantissa: 0xcb090c8001ab551c, exp: 1475},
    Explicit { mantissa: 0xfdcb4fa002162a63, exp: 1478},
    Explicit { mantissa: 0x9e9f11c4014dda7e, exp: 1482},
    Explicit { mantissa: 0xc646d63501a1511d, exp: 1485},
    Explicit { mantissa: 0xf7d88bc24209a565, exp: 1488},
    Explicit { mantissa: 0x9ae757596946075f, exp: 1492},
    Explicit { mantissa: 0xc1a12d2fc3978937, exp: 1495},
    Explicit { mantissa: 0xf209787bb47d6b84, exp: 1498},
    Explicit { mantissa: 0x9745eb4d50ce6332, exp: 1502},
    Explicit { mantissa: 0xbd176620a501fbff, exp: 1505},
    Explicit { mantissa: 0xec5d3fa8ce427aff, exp: 1508},
    Explicit { mantissa: 0x93ba47c980e98cdf, exp: 1512},
    Explicit { mantissa: 0xb8a8d9bbe123f017, exp: 1515},
    Explicit { mantissa: 0xe6d3102ad96cec1d, exp: 1518},
    Explicit { mantissa: 0x9043ea1ac7e41392, exp: 1522},
    Explicit { mantissa: 0xb454e4a179dd1877, exp: 1525},
    Explicit { mantissa: 0xe16a1dc9d8545e94, exp: 1528},
    Explicit { mantissa: 0x8ce2529e2734bb1d, exp: 1532},
    Explicit { mantissa: 0xb01ae745b101e9e4, exp: 1535},
    Explicit { mantissa: 0xdc21a1171d42645d, exp: 1538},
    Explicit { mantissa: 0x899504ae72497eba, exp: 1542},
    Explicit { mantissa: 0xabfa45da0edbde69, exp: 1545},
    Explicit { mantissa: 0xd6f8d7509292d603, exp: 1548},
    Explicit { mantissa: 0x865b86925b9bc5c2, exp: 1552},
    Explicit { mantissa: 0xa7f26836f282b732, exp: 1555},
    Explicit { mantissa: 0xd1ef0244af2364ff, exp: 1558},
    Explicit { mantissa: 0x8335616aed761f1f, exp: 1562},
    Explicit { mantissa: 0xa402b9c5a8d3a6e7, exp: 1565},
    Explicit { mantissa: 0xcd036837130890a1, exp: 1568},
    Explicit { mantissa: 0x802221226be55a64, exp: 1572},
    Explicit { mantissa: 0xa02aa96b06deb0fd, exp: 1575},
    Explicit { mantissa: 0xc83553c5c8965d3d, exp: 1578},
    Explicit { mantissa: 0xfa42a8b73abbf48c, exp: 1581},
    Explicit { mantissa: 0x9c69a97284b578d7, exp: 1585},
    Explicit { mantissa: 0xc38413cf25e2d70d, exp: 1588},
    Explicit { mantissa: 0xf46518c2ef5b8cd1, exp: 1591},
    Explicit { mantissa: 0x98bf2f79d5993802, exp: 1595},
    Explicit { mantissa: 0xbeeefb584aff8603, exp: 1598},
    Explicit { mantissa: 0xeeaaba2e5dbf6784, exp: 1601},
    Explicit { mantissa: 0x952ab45cfa97a0b2, exp: 1605},
    Explicit { mantissa: 0xba756174393d88df, exp: 1608},
    Explicit { mantissa: 0xe912b9d1478ceb17, exp: 1611},
    Explicit { mantissa: 0x91abb422ccb812ee, exp: 1615},
    Explicit { mantissa: 0xb616a12b7fe617aa, exp: 1618},
    Explicit { mantissa: 0xe39c49765fdf9d94, exp: 1621},
    Explicit { mantissa: 0x8e41ade9fbebc27d, exp: 1625},
    Explicit { mantissa: 0xb1d219647ae6b31c, exp: 1628},
    Explicit { mantissa: 0xde469fbd99a05fe3, exp: 1631},
    Explicit { mantissa: 0x8aec23d680043bee, exp: 1635},
    Explicit { mantissa: 0xada72ccc20054ae9, exp: 1638},
    Explicit { mantissa: 0xd910f7ff28069da4, exp: 1641},
    Explicit { mantissa: 0x87aa9aff79042286, exp: 1645},
    Explicit { mantissa: 0xa99541bf57452b28, exp: 1648},
    Explicit { mantissa: 0xd3fa922f2d1675f2, exp: 1651},
    Explicit { mantissa: 0x847c9b5d7c2e09b7, exp: 1655},
    Explicit { mantissa: 0xa59bc234db398c25, exp: 1658},
    Explicit { mantissa: 0xcf02b2c21207ef2e, exp: 1661},
    Explicit { mantissa: 0x8161afb94b44f57d, exp: 1665},
    Explicit { mantissa: 0xa1ba1ba79e1632dc, exp: 1668},
    Explicit { mantissa: 0xca28a291859bbf93, exp: 1671},
    Explicit { mantissa: 0xfcb2cb35e702af78, exp: 1674},
    Explicit { mantissa: 0x9defbf01b061adab, exp: 1678},
    Explicit { mantissa: 0xc56baec21c7a1916, exp: 1681},
    Explicit { mantissa: 0xf6c69a72a3989f5b, exp: 1684},
    Explicit { mantissa: 0x9a3c2087a63f6399, exp: 1688},
    Explicit { mantissa: 0xc0cb28a98fcf3c7f, exp: 1691},
    Explicit { mantissa: 0xf0fdf2d3f3c30b9f, exp: 1694},
    Explicit { mantissa: 0x969eb7c47859e743, exp: 1698},
    Explicit { mantissa: 0xbc4665b596706114, exp: 1701},
    Explicit { mantissa: 0xeb57ff22fc0c7959, exp: 1704},
    Explicit { mantissa: 0x9316ff75dd87cbd8, exp: 1708},
    Explicit { mantissa: 0xb7dcbf5354e9bece, exp: 1711},
    Explicit { mantissa: 0xe5d3ef282a242e81, exp: 1714},
    Explicit { mantissa: 0x8fa475791a569d10, exp: 1718},
    Explicit { mantissa: 0xb38d92d760ec4455, exp: 1721},
    Explicit { mantissa: 0xe070f78d3927556a, exp: 1724},
    Explicit { mantissa: 0x8c469ab843b89562, exp: 1728},
    Explicit { mantissa: 0xaf58416654a6babb, exp: 1731},
    Explicit { mantissa: 0xdb2e51bfe9d0696a, exp: 1734},
    Explicit { mantissa: 0x88fcf317f22241e2, exp: 1738},
    Explicit { mantissa: 0xab3c2fddeeaad25a, exp: 1741},
    Explicit { mantissa: 0xd60b3bd56a5586f1, exp: 1744},
    Explicit { mantissa: 0x85c7056562757456, exp: 1748},
    Explicit { mantissa: 0xa738c6bebb12d16c, exp: 1751},
    Explicit { mantissa: 0xd106f86e69d785c7, exp: 1754},
    Explicit { mantissa: 0x82a45b450226b39c, exp: 1758},
    Explicit { mantissa: 0xa34d721642b06084, exp: 1761},
    Explicit { mantissa: 0xcc20ce9bd35c78a5, exp: 1764},
    Explicit { mantissa: 0xff290242c83396ce, exp: 1767},
    Explicit { mantissa: 0x9f79a169bd203e41, exp: 1771},
    Explicit { mantissa: 0xc75809c42c684dd1, exp: 1774},
    Explicit { mantissa: 0xf92e0c3537826145, exp: 1777},
    Explicit { mantissa: 0x9bbcc7a142b17ccb, exp: 1781},
    Explicit { mantissa: 0xc2abf989935ddbfe, exp: 1784},
    Explicit { mantissa: 0xf356f7ebf83552fe, exp: 1787},
    Explicit { mantissa: 0x98165af37b2153de, exp: 1791},
    Explicit { mantissa: 0xbe1bf1b059e9a8d6, exp: 1794},
    Explicit { mantissa: 0xeda2ee1c7064130c, exp: 1797},
    Explicit { mantissa: 0x9485d4d1c63e8be7, exp: 1801},
    Explicit { mantissa: 0xb9a74a0637ce2ee1, exp: 1804},
    Explicit { mantissa: 0xe8111c87c5c1ba99, exp: 1807},
    Explicit { mantissa: 0x910ab1d4db9914a0, exp: 1811},
    Explicit { mantissa: 0xb54d5e4a127f59c8, exp: 1814},
    Explicit { mantissa: 0xe2a0b5dc971f303a, exp: 1817},
    Explicit { mantissa: 0x8da471a9de737e24, exp: 1821},
    Explicit { mantissa: 0xb10d8e1456105dad, exp: 1824},
    Explicit { mantissa: 0xdd50f1996b947518, exp: 1827},
    Explicit { mantissa: 0x8a5296ffe33cc92f, exp: 1831},
    Explicit { mantissa: 0xace73cbfdc0bfb7b, exp: 1834},
    Explicit { mantissa: 0xd8210befd30efa5a, exp: 1837},
    Explicit { mantissa: 0x8714a775e3e95c78, exp: 1841},
    Explicit { mantissa: 0xa8d9d1535ce3b396, exp: 1844},
    Explicit { mantissa: 0xd31045a8341ca07c, exp: 1847},
    Explicit { mantissa: 0x83ea2b892091e44d, exp: 1851},
    Explicit { mantissa: 0xa4e4b66b68b65d60, exp: 1854},
    Explicit { mantissa: 0xce1de40642e3f4b9, exp: 1857},
    Explicit { mantissa: 0x80d2ae83e9ce78f3, exp: 1861},
    Explicit { mantissa: 0xa1075a24e4421730, exp: 1864},
    Explicit { mantissa: 0xc94930ae1d529cfc, exp: 1867},
    Explicit { mantissa: 0xfb9b7cd9a4a7443c, exp: 1870},
    Explicit { mantissa: 0x9d412e0806e88aa5, exp: 1874},
    Explicit { mantissa: 0xc491798a08a2ad4e, exp: 1877},
    Explicit { mantissa: 0xf5b5d7ec8acb58a2, exp: 1880},
    Explicit { mantissa: 0x9991a6f3d6bf1765, exp: 1884},
    Explicit { mantissa: 0xbff610b0cc6edd3f, exp: 1887},
    Explicit { mantissa: 0xeff394dcff8a948e, exp: 1890},
    Explicit { mantissa: 0x95f83d0a1fb69cd9, exp: 1894},
    Explicit { mantissa: 0xbb764c4ca7a4440f, exp: 1897},
    Explicit { mantissa: 0xea53df5fd18d5513, exp: 1900},
    Explicit { mantissa: 0x92746b9be2f8552c, exp: 1904},
    Explicit { mantissa: 0xb7118682dbb66a77, exp: 1907},
    Explicit { mantissa: 0xe4d5e82392a40515, exp: 1910},
    Explicit { mantissa: 0x8f05b1163ba6832d, exp: 1914},
    Explicit { mantissa: 0xb2c71d5bca9023f8, exp: 1917},
    Explicit { mantissa: 0xdf78e4b2bd342cf6, exp: 1920},
    Explicit { mantissa: 0x8bab8eefb6409c1a, exp: 1924},
    Explicit { mantissa: 0xae9672aba3d0c320, exp: 1927},
    Explicit { mantissa: 0xda3c0f568cc4f3e8, exp: 1930},
    Explicit { mantissa: 0x8865899617fb1871, exp: 1934},
    Explicit { mantissa: 0xaa7eebfb9df9de8d, exp: 1937},
    Explicit { mantissa: 0xd51ea6fa85785631, exp: 1940},
    Explicit { mantissa: 0x8533285c936b35de, exp: 1944},
    Explicit { mantissa: 0xa67ff273b8460356, exp: 1947},
    Explicit { mantissa: 0xd01fef10a657842c, exp: 1950},
    Explicit { mantissa: 0x8213f56a67f6b29b, exp: 1954},
    Explicit { mantissa: 0xa298f2c501f45f42, exp: 1957},
    Explicit { mantissa: 0xcb3f2f7642717713, exp: 1960},
    Explicit { mantissa: 0xfe0efb53d30dd4d7, exp: 1963},
    Explicit { mantissa: 0x9ec95d1463e8a506, exp: 1967},
    Explicit { mantissa: 0xc67bb4597ce2ce48, exp: 1970},
    Explicit { mantissa: 0xf81aa16fdc1b81da, exp: 1973},
    Explicit { mantissa: 0x9b10a4e5e9913128, exp: 1977},
    Explicit { mantissa: 0xc1d4ce1f63f57d72, exp: 1980},
    Explicit { mantissa: 0xf24a01a73cf2dccf, exp: 1983},
    Explicit { mantissa: 0x976e41088617ca01, exp: 1987},
    Explicit { mantissa: 0xbd49d14aa79dbc82, exp: 1990},
    Explicit { mantissa: 0xec9c459d51852ba2, exp: 1993},
    Explicit { mantissa: 0x93e1ab8252f33b45, exp: 1997},
    Explicit { mantissa: 0xb8da1662e7b00a17, exp: 2000},
    Explicit { mantissa: 0xe7109bfba19c0c9d, exp: 2003},
    Explicit { mantissa: 0x906a617d450187e2, exp: 2007},
    Explicit { mantissa: 0xb484f9dc9641e9da, exp: 2010},
    Explicit { mantissa: 0xe1a63853bbd26451, exp: 2013},
    Explicit { mantissa: 0x8d07e33455637eb2, exp: 2017},
    Explicit { mantissa: 0xb049dc016abc5e5f, exp: 2020},
    Explicit { mantissa: 0xdc5c5301c56b75f7, exp: 2023},
    Explicit { mantissa: 0x89b9b3e11b6329ba, exp: 2027},
    Explicit { mantissa: 0xac2820d9623bf429, exp: 2030},
    Explicit { mantissa: 0xd732290fbacaf133, exp: 2033},
    Explicit { mantissa: 0x867f59a9d4bed6c0, exp: 2037},
    Explicit { mantissa: 0xa81f301449ee8c70, exp: 2040},
    Explicit { mantissa: 0xd226fc195c6a2f8c, exp: 2043},
    Explicit { mantissa: 0x83585d8fd9c25db7, exp: 2047},
    Explicit { mantissa: 0xa42e74f3d032f525, exp: 2050},
    Explicit { mantissa: 0xcd3a1230c43fb26f, exp: 2053},
    Explicit { mantissa: 0x80444b5e7aa7cf85, exp: 2057},
    Explicit { mantissa: 0xa0555e361951c366, exp: 2060},
    Explicit { mantissa: 0xc86ab5c39fa63440, exp: 2063},
    Explicit { mantissa: 0xfa856334878fc150, exp: 2066},
    Explicit { mantissa: 0x9c935e00d4b9d8d2, exp: 2070},
    Explicit { mantissa: 0xc3b8358109e84f07, exp: 2073},
    Explicit { mantissa: 0xf4a642e14c6262c8, exp: 2076},
    Explicit { mantissa: 0x98e7e9cccfbd7dbd, exp: 2080},
    Explicit { mantissa: 0xbf21e44003acdd2c, exp: 2083},
    Explicit { mantissa: 0xeeea5d5004981478, exp: 2086},
    Explicit { mantissa: 0x95527a5202df0ccb, exp: 2090},
    Explicit { mantissa: 0xbaa718e68396cffd, exp: 2093},
    Explicit { mantissa: 0xe950df20247c83fd, exp: 2096},
    Explicit { mantissa: 0x91d28b7416cdd27e, exp: 2100},
    Explicit { mantissa: 0xb6472e511c81471d, exp: 2103},
    Explicit { mantissa: 0xe3d8f9e563a198e5, exp: 2106},
    Explicit { mantissa: 0x8e679c2f5e44ff8f, exp: 2110},
];

// Un-truncated mantissas for the relevant powers of 10.
#[rustfmt::skip]
const MANTISSA_128: &[u64] = &[
    0x419ea3bd35385e2d, 0x52064cac828675b9, 0x7343efebd1940993, 0x1014ebe6c5f90bf8,
    0xd41a26e077774ef6, 0x8920b098955522b4, 0x55b46e5f5d5535b0, 0xeb2189f734aa831d,
    0xa5e9ec7501d523e4, 0x47b233c92125366e, 0x999ec0bb696e840a, 0xc00670ea43ca250d,
    0x380406926a5e5728, 0xc605083704f5ecf2, 0xf7864a44c633682e, 0x7ab3ee6afbe0211d,
    0x5960ea05bad82964, 0x6fb92487298e33bd, 0xa5d3b6d479f8e056, 0x8f48a4899877186c,
    0x331acdabfe94de87, 0x9ff0c08b7f1d0b14, 0x7ecf0ae5ee44dd9, 0xc9e82cd9f69d6150,
    0xbe311c083a225cd2, 0x6dbd630a48aaf406, 0x92cbbccdad5b108, 0x25bbf56008c58ea5,
    0xaf2af2b80af6f24e, 0x1af5af660db4aee1, 0x50d98d9fc890ed4d, 0xe50ff107bab528a0,
    0x1e53ed49a96272c8, 0x25e8e89c13bb0f7a, 0x77b191618c54e9ac, 0xd59df5b9ef6a2417,
    0x4b0573286b44ad1d, 0x4ee367f9430aec32, 0x229c41f793cda73f, 0x6b43527578c1110f,
    0x830a13896b78aaa9, 0x23cc986bc656d553, 0x2cbfbe86b7ec8aa8, 0x7bf7d71432f3d6a9,
    0xdaf5ccd93fb0cc53, 0xd1b3400f8f9cff68, 0x23100809b9c21fa1, 0xabd40a0c2832a78a,
    0x16c90c8f323f516c, 0xae3da7d97f6792e3, 0x99cd11cfdf41779c, 0x40405643d711d583,
    0x482835ea666b2572, 0xda3243650005eecf, 0x90bed43e40076a82, 0x5a7744a6e804a291,
    0x711515d0a205cb36, 0xd5a5b44ca873e03, 0xe858790afe9486c2, 0x626e974dbe39a872,
    0xfb0a3d212dc8128f, 0x7ce66634bc9d0b99, 0x1c1fffc1ebc44e80, 0xa327ffb266b56220,
    0x4bf1ff9f0062baa8, 0x6f773fc3603db4a9, 0xcb550fb4384d21d3, 0x7e2a53a146606a48,
    0x2eda7444cbfc426d, 0xfa911155fefb5308, 0x793555ab7eba27ca, 0x4bc1558b2f3458de,
    0x9eb1aaedfb016f16, 0x465e15a979c1cadc, 0xbfacd89ec191ec9, 0xcef980ec671f667b,
    0x82b7e12780e7401a, 0xd1b2ecb8b0908810, 0x861fa7e6dcb4aa15, 0x67a791e093e1d49a,
    0xe0c8bb2c5c6d24e0, 0x58fae9f773886e18, 0xaf39a475506a899e, 0x6d8406c952429603,
    0xc8e5087ba6d33b83, 0xfb1e4a9a90880a64, 0x5cf2eea09a55067f, 0xf42faa48c0ea481e,
    0xf13b94daf124da26, 0x76c53d08d6b70858, 0x54768c4b0c64ca6e, 0xa9942f5dcf7dfd09,
    0xd3f93b35435d7c4c, 0xc47bc5014a1a6daf, 0x359ab6419ca1091b, 0xc30163d203c94b62,
    0x79e0de63425dcf1d, 0x985915fc12f542e4, 0x3e6f5b7b17b2939d, 0xa705992ceecf9c42,
    0x50c6ff782a838353, 0xa4f8bf5635246428, 0x871b7795e136be99, 0x28e2557b59846e3f,
    0x331aeada2fe589cf, 0x3ff0d2c85def7621, 0xfed077a756b53a9, 0xd3e8495912c62894,
    0x64712dd7abbbd95c, 0xbd8d794d96aacfb3, 0xecf0d7a0fc5583a0, 0xf41686c49db57244,
    0x311c2875c522ced5, 0x7d633293366b828b, 0xae5dff9c02033197, 0xd9f57f830283fdfc,
    0xd072df63c324fd7b, 0x4247cb9e59f71e6d, 0x52d9be85f074e608, 0x67902e276c921f8b,
    0xba1cd8a3db53b6, 0x80e8a40eccd228a4, 0x6122cd128006b2cd, 0x796b805720085f81,
    0xcbe3303674053bb0, 0xbedbfc4411068a9c, 0xee92fb5515482d44, 0x751bdd152d4d1c4a,
    0xd262d45a78a0635d, 0x86fb897116c87c34, 0xd45d35e6ae3d4da0, 0x8974836059cca109,
    0x2bd1a438703fc94b, 0x7b6306a34627ddcf, 0x1a3bc84c17b1d542, 0x20caba5f1d9e4a93,
    0x547eb47b7282ee9c, 0xe99e619a4f23aa43, 0x6405fa00e2ec94d4, 0xde83bc408dd3dd04,
    0x9624ab50b148d445, 0x3badd624dd9b0957, 0xe54ca5d70a80e5d6, 0x5e9fcf4ccd211f4c,
    0x7647c3200069671f, 0x29ecd9f40041e073, 0xf468107100525890, 0x7182148d4066eeb4,
    0xc6f14cd848405530, 0xb8ada00e5a506a7c, 0xa6d90811f0e4851c, 0x908f4a166d1da663,
    0x9a598e4e043287fe, 0x40eff1e1853f29fd, 0xd12bee59e68ef47c, 0x82bb74f8301958ce,
    0xe36a52363c1faf01, 0xdc44e6c3cb279ac1, 0x29ab103a5ef8c0b9, 0x7415d448f6b6f0e7,
    0x111b495b3464ad21, 0xcab10dd900beec34, 0x3d5d514f40eea742, 0xcb4a5a3112a5112,
    0x47f0e785eaba72ab, 0x59ed216765690f56, 0x306869c13ec3532c, 0x1e414218c73a13fb,
    0xe5d1929ef90898fa, 0xdf45f746b74abf39, 0x6b8bba8c328eb783, 0x66ea92f3f326564,
    0xc80a537b0efefebd, 0xbd06742ce95f5f36, 0x2c48113823b73704, 0xf75a15862ca504c5,
    0x9a984d73dbe722fb, 0xc13e60d0d2e0ebba, 0x318df905079926a8, 0xfdf17746497f7052,
    0xfeb6ea8bedefa633, 0xfe64a52ee96b8fc0, 0x3dfdce7aa3c673b0, 0x6bea10ca65c084e,
    0x486e494fcff30a62, 0x5a89dba3c3efccfa, 0xf89629465a75e01c, 0xf6bbb397f1135823,
    0x746aa07ded582e2c, 0xa8c2a44eb4571cdc, 0x92f34d62616ce413, 0x77b020baf9c81d17,
    0xace1474dc1d122e, 0xd819992132456ba, 0x10e1fff697ed6c69, 0xca8d3ffa1ef463c1,
    0xbd308ff8a6b17cb2, 0xac7cb3f6d05ddbde, 0x6bcdf07a423aa96b, 0x86c16c98d2c953c6,
    0xe871c7bf077ba8b7, 0x11471cd764ad4972, 0xd598e40d3dd89bcf, 0x4aff1d108d4ec2c3,
    0xcedf722a585139ba, 0xc2974eb4ee658828, 0x733d226229feea32, 0x806357d5a3f525f,
    0xca07c2dcb0cf26f7, 0xfc89b393dd02f0b5, 0xbbac2078d443ace2, 0xd54b944b84aa4c0d,
    0xa9e795e65d4df11, 0x4d4617b5ff4a16d5, 0x504bced1bf8e4e45, 0xe45ec2862f71e1d6,
    0x5d767327bb4e5a4c, 0x3a6a07f8d510f86f, 0x890489f70a55368b, 0x2b45ac74ccea842e,
    0x3b0b8bc90012929d, 0x9ce6ebb40173744, 0xcc420a6a101d0515, 0x9fa946824a12232d,
    0x47939822dc96abf9, 0x59787e2b93bc56f7, 0x57eb4edb3c55b65a, 0xede622920b6b23f1,
    0xe95fab368e45eced, 0x11dbcb0218ebb414, 0xd652bdc29f26a119, 0x4be76d3346f0495f,
    0x6f70a4400c562ddb, 0xcb4ccd500f6bb952, 0x7e2000a41346a7a7, 0x8ed400668c0c28c8,
    0x728900802f0f32fa, 0x4f2b40a03ad2ffb9, 0xe2f610c84987bfa8, 0xdd9ca7d2df4d7c9,
    0x91503d1c79720dbb, 0x75a44c6397ce912a, 0xc986afbe3ee11aba, 0xfbe85badce996168,
    0xfae27299423fb9c3, 0xdccd879fc967d41a, 0x5400e987bbc1c920, 0x290123e9aab23b68,
    0xf9a0b6720aaf6521, 0xf808e40e8d5b3e69, 0xb60b1d1230b20e04, 0xb1c6f22b5e6f48c2,
    0x1e38aeb6360b1af3, 0x25c6da63c38de1b0, 0x579c487e5a38ad0e, 0x2d835a9df0c6d851,
    0xf8e431456cf88e65, 0x1b8e9ecb641b58ff, 0xe272467e3d222f3f, 0x5b0ed81dcc6abb0f,
    0x98e947129fc2b4e9, 0x3f2398d747b36224, 0x8eec7f0d19a03aad, 0x1953cf68300424ac,
    0x5fa8c3423c052dd7, 0x3792f412cb06794d, 0xe2bbd88bbee40bd0, 0x5b6aceaeae9d0ec4,
    0xf245825a5a445275, 0xeed6e2f0f0d56712, 0x55464dd69685606b, 0xaa97e14c3c26b886,
    0xd53dd99f4b3066a8, 0xe546a8038efe4029, 0xde98520472bdd033, 0x963e66858f6d4440,
    0xdde7001379a44aa8, 0x5560c018580d5d52, 0xaab8f01e6e10b4a6, 0xcab3961304ca70e8,
    0x3d607b97c5fd0d22, 0x8cb89a7db77c506a, 0x77f3608e92adb242, 0x55f038b237591ed3,
    0x6b6c46dec52f6688, 0x2323ac4b3b3da015, 0xabec975e0a0d081a, 0x96e7bd358c904a21,
    0x7e50d64177da2e54, 0xdde50bd1d5d0b9e9, 0x955e4ec64b44e864, 0xbd5af13bef0b113e,
    0xecb1ad8aeacdd58e, 0x67de18eda5814af2, 0x80eacf948770ced7, 0xa1258379a94d028d,
    0x96ee45813a04330, 0x8bca9d6e188853fc, 0x775ea264cf55347d, 0x95364afe032a819d,
    0x3a83ddbd83f52204, 0xc4926a9672793542, 0x75b7053c0f178293, 0x5324c68b12dd6338,
    0xd3f6fc16ebca5e03, 0x88f4bb1ca6bcf584, 0x2b31e9e3d06c32e5, 0x3aff322e62439fcf,
    0x9befeb9fad487c2, 0x4c2ebe687989a9b3, 0xf9d37014bf60a10, 0x538484c19ef38c94,
    0x2865a5f206b06fb9, 0xf93f87b7442e45d3, 0xf78f69a51539d748, 0xb573440e5a884d1b,
    0x31680a88f8953030, 0xfdc20d2b36ba7c3d, 0x3d32907604691b4c, 0xa63f9a49c2c1b10f,
    0xfcf80dc33721d53, 0xd3c36113404ea4a8, 0x645a1cac083126e9, 0x3d70a3d70a3d70a3,
    0xcccccccccccccccc, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x4000000000000000,
    0x5000000000000000, 0xa400000000000000, 0x4d00000000000000, 0xf020000000000000,
    0x6c28000000000000, 0xc732000000000000, 0x3c7f400000000000, 0x4b9f100000000000,
    0x1e86d40000000000, 0x1314448000000000, 0x17d955a000000000, 0x5dcfab0800000000,
    0x5aa1cae500000000, 0xf14a3d9e40000000, 0x6d9ccd05d0000000, 0xe4820023a2000000,
    0xdda2802c8a800000, 0xd50b2037ad200000, 0x4526f422cc340000, 0x9670b12b7f410000,
    0x3c0cdd765f114000, 0xa5880a69fb6ac800, 0x8eea0d047a457a00, 0x72a4904598d6d880,
    0x47a6da2b7f864750, 0x999090b65f67d924, 0xfff4b4e3f741cf6d, 0xbff8f10e7a8921a4,
    0xaff72d52192b6a0d, 0x9bf4f8a69f764490, 0x2f236d04753d5b4, 0x1d762422c946590,
    0x424d3ad2b7b97ef5, 0xd2e0898765a7deb2, 0x63cc55f49f88eb2f, 0x3cbf6b71c76b25fb,
    0x8bef464e3945ef7a, 0x97758bf0e3cbb5ac, 0x3d52eeed1cbea317, 0x4ca7aaa863ee4bdd,
    0x8fe8caa93e74ef6a, 0xb3e2fd538e122b44, 0x60dbbca87196b616, 0xbc8955e946fe31cd,
    0x6babab6398bdbe41, 0xc696963c7eed2dd1, 0xfc1e1de5cf543ca2, 0x3b25a55f43294bcb,
    0x49ef0eb713f39ebe, 0x6e3569326c784337, 0x49c2c37f07965404, 0xdc33745ec97be906,
    0x69a028bb3ded71a3, 0xc40832ea0d68ce0c, 0xf50a3fa490c30190, 0x792667c6da79e0fa,
    0x577001b891185938, 0xed4c0226b55e6f86, 0x544f8158315b05b4, 0x696361ae3db1c721,
    0x3bc3a19cd1e38e9, 0x4ab48a04065c723, 0x62eb0d64283f9c76, 0x3ba5d0bd324f8394,
    0xca8f44ec7ee36479, 0x7e998b13cf4e1ecb, 0x9e3fedd8c321a67e, 0xc5cfe94ef3ea101e,
    0xbba1f1d158724a12, 0x2a8a6e45ae8edc97, 0xf52d09d71a3293bd, 0x593c2626705f9c56,
    0x6f8b2fb00c77836c, 0xb6dfb9c0f956447, 0x4724bd4189bd5eac, 0x58edec91ec2cb657,
    0x2f2967b66737e3ed, 0xbd79e0d20082ee74, 0xecd8590680a3aa11, 0xe80e6f4820cc9495,
    0x3109058d147fdcdd, 0xbd4b46f0599fd415, 0x6c9e18ac7007c91a, 0x3e2cf6bc604ddb0,
    0x84db8346b786151c, 0xe612641865679a63, 0x4fcb7e8f3f60c07e, 0xe3be5e330f38f09d,
    0x5cadf5bfd3072cc5, 0x73d9732fc7c8f7f6, 0x2867e7fddcdd9afa, 0xb281e1fd541501b8,
    0x1f225a7ca91a4226, 0x3375788de9b06958, 0x52d6b1641c83ae, 0xc0678c5dbd23a49a,
    0xf840b7ba963646e0, 0xb650e5a93bc3d898, 0xa3e51f138ab4cebe, 0xc66f336c36b10137,
    0xb80b0047445d4184, 0xa60dc059157491e5, 0x87c89837ad68db2f, 0x29babe4598c311fb,
    0xf4296dd6fef3d67a, 0x1899e4a65f58660c, 0x5ec05dcff72e7f8f, 0x76707543f4fa1f73,
    0x6a06494a791c53a8, 0x487db9d17636892, 0x45a9d2845d3c42b6, 0xb8a2392ba45a9b2,
    0x8e6cac7768d7141e, 0x3207d795430cd926, 0x7f44e6bd49e807b8, 0x5f16206c9c6209a6,
    0x36dba887c37a8c0f, 0xc2494954da2c9789, 0xf2db9baa10b7bd6c, 0x6f92829494e5acc7,
    0xcb772339ba1f17f9, 0xff2a760414536efb, 0xfef5138519684aba, 0x7eb258665fc25d69,
    0xef2f773ffbd97a61, 0xaafb550ffacfd8fa, 0x95ba2a53f983cf38, 0xdd945a747bf26183,
    0x94f971119aeef9e4, 0x7a37cd5601aab85d, 0xac62e055c10ab33a, 0x577b986b314d6009,
    0xed5a7e85fda0b80b, 0x14588f13be847307, 0x596eb2d8ae258fc8, 0x6fca5f8ed9aef3bb,
    0x25de7bb9480d5854, 0xaf561aa79a10ae6a, 0x1b2ba1518094da04, 0x90fb44d2f05d0842,
    0x353a1607ac744a53, 0x42889b8997915ce8, 0x69956135febada11, 0x43fab9837e699095,
    0x94f967e45e03f4bb, 0x1d1be0eebac278f5, 0x6462d92a69731732, 0x7d7b8f7503cfdcfe,
    0x5cda735244c3d43e, 0x3a0888136afa64a7, 0x88aaa1845b8fdd0, 0x8aad549e57273d45,
    0x36ac54e2f678864b, 0x84576a1bb416a7dd, 0x656d44a2a11c51d5, 0x9f644ae5a4b1b325,
    0x873d5d9f0dde1fee, 0xa90cb506d155a7ea, 0x9a7f12442d588f2, 0xc11ed6d538aeb2f,
    0x8f1668c8a86da5fa, 0xf96e017d694487bc, 0x37c981dcc395a9ac, 0x85bbe253f47b1417,
    0x93956d7478ccec8e, 0x387ac8d1970027b2, 0x6997b05fcc0319e, 0x441fece3bdf81f03,
    0xd527e81cad7626c3, 0x8a71e223d8d3b074, 0xf6872d5667844e49, 0xb428f8ac016561db,
    0xe13336d701beba52, 0xecc0024661173473, 0x27f002d7f95d0190, 0x31ec038df7b441f4,
    0x7e67047175a15271, 0xf0062c6e984d386, 0x52c07b78a3e60868, 0xa7709a56ccdf8a82,
    0x88a66076400bb691, 0x6acff893d00ea435, 0x583f6b8c4124d43, 0xc3727a337a8b704a,
    0x744f18c0592e4c5c, 0x1162def06f79df73, 0x8addcb5645ac2ba8, 0x6d953e2bd7173692,
    0xc8fa8db6ccdd0437, 0x1d9c9892400a22a2, 0x2503beb6d00cab4b, 0x2e44ae64840fd61d,
    0x5ceaecfed289e5d2, 0x7425a83e872c5f47, 0xd12f124e28f77719, 0x82bd6b70d99aaa6f,
    0x636cc64d1001550b, 0x3c47f7e05401aa4e, 0x65acfaec34810a71, 0x7f1839a741a14d0d,
    0x1ede48111209a050, 0x934aed0aab460432, 0xf81da84d5617853f, 0x36251260ab9d668e,
    0xc1d72b7c6b426019, 0xb24cf65b8612f81f, 0xdee033f26797b627, 0x169840ef017da3b1,
    0x8e1f289560ee864e, 0xf1a6f2bab92a27e2, 0xae10af696774b1db, 0xacca6da1e0a8ef29,
    0x17fd090a58d32af3, 0xddfc4b4cef07f5b0, 0x4abdaf101564f98e, 0x9d6d1ad41abe37f1,
    0x84c86189216dc5ed, 0x32fd3cf5b4e49bb4, 0x3fbc8c33221dc2a1, 0xfabaf3feaa5334a,
    0x29cb4d87f2a7400e, 0x743e20e9ef511012, 0x914da9246b255416, 0x1ad089b6c2f7548e,
    0xa184ac2473b529b1, 0xc9e5d72d90a2741e, 0x7e2fa67c7a658892, 0xddbb901b98feeab7,
    0x552a74227f3ea565, 0xd53a88958f87275f, 0x8a892abaf368f137, 0x2d2b7569b0432d85,
    0x9c3b29620e29fc73, 0x8349f3ba91b47b8f, 0x241c70a936219a73, 0xed238cd383aa0110,
    0xf4363804324a40aa, 0xb143c6053edcd0d5, 0xdd94b7868e94050a, 0xca7cf2b4191c8326,
    0xfd1c2f611f63a3f0, 0xbc633b39673c8cec, 0xd5be0503e085d813, 0x4b2d8644d8a74e18,
    0xddf8e7d60ed1219e, 0xcabb90e5c942b503, 0x3d6a751f3b936243, 0xcc512670a783ad4,
    0x27fb2b80668b24c5, 0xb1f9f660802dedf6, 0x5e7873f8a0396973, 0xdb0b487b6423e1e8,
    0x91ce1a9a3d2cda62, 0x7641a140cc7810fb, 0xa9e904c87fcb0a9d, 0x546345fa9fbdcd44,
    0xa97c177947ad4095, 0x49ed8eabcccc485d, 0x5c68f256bfff5a74, 0x73832eec6fff3111,
    0xc831fd53c5ff7eab, 0xba3e7ca8b77f5e55, 0x28ce1bd2e55f35eb, 0x7980d163cf5b81b3,
    0xd7e105bcc332621f, 0x8dd9472bf3fefaa7, 0xb14f98f6f0feb951, 0x6ed1bf9a569f33d3,
    0xa862f80ec4700c8, 0xcd27bb612758c0fa, 0x8038d51cb897789c, 0xe0470a63e6bd56c3,
    0x1858ccfce06cac74, 0xf37801e0c43ebc8, 0xd30560258f54e6ba, 0x47c6b82ef32a2069,
    0x4cdc331d57fa5441, 0xe0133fe4adf8e952, 0x58180fddd97723a6, 0x570f09eaa7ea7648,
];
