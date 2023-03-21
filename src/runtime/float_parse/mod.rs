//! Fast float parser based on github.com/lemire/fast_double_parser, but adopted to support AWK
//! semantics (no failures, just 0s and stopping early). Mistakes are surely my own.

fn is_integer(c: u8) -> bool {
    c.is_ascii_digit()
}

/// The simdjson repo has more optimizations to add for int parsing, but this is a big win over libc
/// for the time being, if only because we do not have to copy `s` into a NUL-terminated
/// representation.
pub fn strtoi(bs: &[u8]) -> i64 {
    if bs.is_empty() {
        return 0;
    }
    let neg = bs[0] == b'-';
    let off = if neg || bs[0] == b'+' { 1 } else { 0 };
    let mut i = 0i64;
    for b in bs[off..].iter().cloned().take_while(|b| is_integer(*b)) {
        let digit = (b - b'0') as i64;
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

/// Simple hexadecimal integer parser, similar in spirit to the strtoi implementation here.
pub fn hextoi(mut bs: &[u8]) -> i64 {
    let mut neg = false;
    if bs.is_empty() {
        return 0;
    }
    if bs[0] == b'-' {
        neg = true;
        bs = &bs[1..]
    }
    if bs.len() >= 2 && bs[0..2] == [b'0', b'x'] || bs[0..2] == [b'0', b'X'] {
        bs = &bs[2..]
    }
    let mut i = 0i64;
    for b in bs.iter().cloned() {
        let digit = match b {
            b'A'..=b'F' => (b - b'A') as i64 + 10,
            b'a'..=b'f' => (b - b'a') as i64 + 10,
            b'0'..=b'9' => (b - b'0') as i64,
            _ => break,
        };
        i = if let Some(i) = i.checked_mul(16).and_then(|i| i.checked_add(digit)) {
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

/// Parse a floating-poing number from `bs`, returning 0 if one isn't there.
pub fn strtod(bs: &[u8]) -> f64 {
    if let Ok((f, _)) = fast_float::parse_partial(bs) {
        f
    } else {
        0.0f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basic_behavior() {
        assert_eq!(strtod(b"1.234"), 1.234);
        assert_eq!(strtod(b"1.234hello"), 1.234);
        assert_eq!(strtod(b"1.234E70hello"), 1.234E70);
        assert_eq!(strtod(b"752834029324532"), 752834029324532.0);
        assert_eq!(strtod(b"-3.463682231963e-01"), -3.463682231963e-01);
        assert_eq!(strtod(b""), 0.0);
        let imax = format!("{}", i64::max_value());
        let imin = format!("{}", i64::min_value());
        assert_eq!(strtod(imax.as_bytes()), i64::max_value() as f64);
        assert_eq!(strtod(imin.as_bytes()), i64::min_value() as f64);
    }
}
