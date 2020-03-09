/// This module implements much of printf in awk.
///
/// We lean heavily on ryu and the std::fmt machinery; as such, most of the work is parsing
/// awk-style format strings and translating them to individual calls to write!.
use crate::common::Result;
use crate::runtime::{convert, strton::strtoi, Float, Int, Str};

use std::convert::TryFrom;
use std::fmt;
use std::io::Write;
use std::str;

type SmallVec<T> = smallvec::SmallVec<[T; 32]>;

#[derive(Default)]
struct StackWriter(pub SmallVec<u8>);

impl StackWriter {
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl Write for StackWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
#[derive(Clone)]
pub(crate) enum FormatArg<'a> {
    S(Str<'a>),
    F(Float),
    I(Int),
}

impl<'a> From<Str<'a>> for FormatArg<'a> {
    fn from(s: Str<'a>) -> FormatArg<'a> {
        FormatArg::S(s)
    }
}

impl<'a> From<&'a str> for FormatArg<'a> {
    fn from(s: &'a str) -> FormatArg<'a> {
        FormatArg::S(s.into())
    }
}

impl<'a> From<Int> for FormatArg<'a> {
    fn from(i: Int) -> FormatArg<'a> {
        FormatArg::I(i)
    }
}

impl<'a> From<Float> for FormatArg<'a> {
    fn from(f: Float) -> FormatArg<'a> {
        FormatArg::F(f)
    }
}

impl<'a> FormatArg<'a> {
    fn to_float(&self) -> f64 {
        use FormatArg::*;
        match self {
            S(s) => convert::<_, f64>(s),
            F(f) => *f,
            I(i) => convert::<_, f64>(*i),
        }
    }
    fn to_int(&self) -> i64 {
        use FormatArg::*;
        match self {
            S(s) => convert::<_, i64>(s),
            F(f) => convert::<_, i64>(*f),
            I(i) => *i,
        }
    }
    fn with_str<R>(&self, f: impl FnOnce(&str) -> R) -> R {
        use FormatArg::*;
        let s: Str<'a> = match self {
            S(s) => s.clone(),
            F(f) => convert::<_, Str>(*f),
            I(i) => convert::<_, Str>(*i),
        };
        s.with_str(f)
    }
}

#[derive(Copy, Clone)]
struct FormatSpec {
    // leading '-' ? -- left justification.
    minus: bool,
    // number to the left of '.', if any
    leading_zeros: bool,
    // padding
    lnum: usize,
    // maximum string width, or floating point precision.
    rnum: usize,
    // format specifier: e.g. c, d, s, x.
    spec: u8,
}

impl Default for FormatSpec {
    fn default() -> FormatSpec {
        FormatSpec {
            minus: false,
            leading_zeros: false,
            lnum: 0,
            rnum: usize::max_value(),
            spec: 'z' as u8, /* invalid */
        }
    }
}

fn is_spec(c: char) -> bool {
    match c {
        'f' | 'c' | 'd' | 'e' | 'g' | 'o' | 's' | 'x' => true,
        _ => false,
    }
}

fn process_spec(mut w: impl Write, fspec: &mut FormatSpec, arg: &FormatArg) -> Result<()> {
    macro_rules! match_for_spec {
        ($s:expr, $arg:expr) => {
            match (
                fspec.minus,
                fspec.leading_zeros,
                fspec.lnum,
                fspec.rnum == usize::max_value(),
            ) {
                (true, true, lnum, true) => write!(w, concat!("{:0<l$", $s, "}"), $arg, l = lnum),
                (true, false, lnum, true) => write!(w, concat!("{:<l$", $s, "}"), $arg, l = lnum),
                (true, true, lnum, false) => write!(
                    w,
                    concat!("{:0<l$.r$", $s, "}"),
                    $arg,
                    l = lnum,
                    r = fspec.rnum
                ),
                (true, false, lnum, false) => write!(
                    w,
                    concat!("{:<l$.r$", $s, "}"),
                    $arg,
                    l = lnum,
                    r = fspec.rnum
                ),
                (false, true, lnum, true) => write!(w, concat!("{:0>l$", $s, "}"), $arg, l = lnum),
                (false, false, lnum, true) => write!(w, concat!("{:>l$", $s, "}"), $arg, l = lnum),
                (false, true, lnum, false) => write!(
                    w,
                    concat!("{:0>l$.r$", $s, "}"),
                    $arg,
                    l = lnum,
                    r = fspec.rnum
                ),
                (false, false, lnum, false) => write!(
                    w,
                    concat!("{:>l$.r$", $s, "}"),
                    $arg,
                    l = lnum,
                    r = fspec.rnum
                ),
            }
        };
    }
    let res = match fspec.spec as char {
        'f' => {
            if !fspec.leading_zeros && fspec.lnum == 0 && fspec.rnum == usize::max_value() {
                // Fast path: use Ryu, which today is more efficient than the standard library.
                // NB Ryu prints some things a bit differently than most awk implementations.
                // `write!(w, "{}", arg.to_float())` is a bit closer.
                let mut buf = ryu::Buffer::new();
                write!(w, "{}", buf.format(arg.to_float()))
            } else {
                match_for_spec!("", arg.to_float())
            }
        }
        'e' => match_for_spec!("e", arg.to_float()),
        'g' => {
            let mut buf = StackWriter::default();
            // %g means "pick the shorter of standard and scientific notation". We do the obvious
            // thing of computing both and writing out the smaller one.
            fspec.spec = 'f' as u8;
            process_spec(&mut buf, fspec, arg)?;
            let l1 = buf.len();
            fspec.spec = 'e' as u8;
            process_spec(&mut buf, fspec, arg)?;
            let l2 = buf.len() - l1;
            if l1 < l2 {
                write!(w, "{}", unsafe { str::from_utf8_unchecked(&buf.0[0..l1]) })
            } else {
                write!(w, "{}", unsafe {
                    str::from_utf8_unchecked(&buf.0[l1..(l1 + l2)])
                })
            }
        }
        'd' => match_for_spec!("", arg.to_int()),
        'o' => match_for_spec!("o", arg.to_int()),
        'x' => match_for_spec!("x", arg.to_int()),
        'c' => match char::try_from(arg.to_int() as u32) {
            Ok(ch) if ch.is_ascii() => match_for_spec!("", ch),
            _ => arg.with_str(|s| {
                if let Some(ch) = s.chars().next() {
                    match_for_spec!("", ch)
                } else {
                    Ok(())
                }
            }),
        },
        's' => arg.with_str(|s| match_for_spec!("", s)),
        x => return err!("unsupported format specifier: {}", x),
    };
    wrap_result(res)
}

fn wrap_result(r: std::result::Result<(), impl fmt::Display>) -> Result<()> {
    match r {
        Ok(()) => Ok(()),
        Err(e) => err!("formatter: {}", e),
    }
}

fn write_str(mut w: impl Write, s: &str) -> Result<()> {
    wrap_result(write!(w, "{}", s))
}

pub(crate) fn printf(mut w: impl Write, spec: &str, mut args: &[FormatArg]) -> Result<()> {
    #[derive(Copy, Clone)]
    enum State {
        // Byte index of start of string
        Raw(usize),
        // Byte index of percent sign
        Format(usize),
    }

    use State::*;
    let mut iter = spec.char_indices();
    macro_rules! next_state {
        ($e:expr) => {
            match $e {
                Some((_, '%')) => Format(0),
                Some(_) => Raw(0),
                None => return Ok(()),
            }
        };
    }
    let mut state = next_state!(iter.next());
    let default = FormatArg::S(Default::default());
    let mut next_arg = || {
        if args.len() == 0 {
            &default
        } else {
            let res = &args[0];
            args = &args[1..];
            res
        }
    };
    let mut buf = SmallVec::new();
    'outer: loop {
        match state {
            Raw(start) => {
                while let Some((ix, ch)) = iter.next() {
                    if ch == '%' {
                        write_str(&mut w, &spec[start..ix])?;
                        state = Format(ix);
                        continue 'outer;
                    }
                }
                write_str(&mut w, &spec[start..])?;
                break 'outer;
            }
            Format(start) => {
                let mut fs = FormatSpec::default();
                #[derive(Copy, Clone)]
                enum Stage {
                    Begin,
                    Lnum,
                    Rnum,
                }
                use Stage::*;
                let mut stage = Begin;
                let mut next = iter.next();
                // AWK is, as usual, rather permissive when it comes to invalid format specifiers:
                // If something is formatted incorrectly, it is simply treated like a normal
                // string. We implement by checking for error conditions and `break`ing out of the
                // inner loop, which will change state to Raw(start).
                while let Some((ix, ch)) = next {
                    if !ch.is_ascii() {
                        // We cast characters to bytes in what follows.
                        break;
                    }
                    match (ch, stage) {
                        ('%', Begin) => {
                            fs.spec = '%' as u8;
                            process_spec(&mut w, &mut fs, next_arg())?;
                            state = Raw(ix + 1);
                            continue 'outer;
                        }
                        (ch, _) if is_spec(ch) => {
                            fs.spec = ch as u8;
                            process_spec(&mut w, &mut fs, next_arg())?;
                            state = Raw(ix + 1);
                            continue 'outer;
                        }
                        ('-', Begin) => {
                            stage = Lnum;
                            fs.minus = true;
                        }
                        ('-', _) | ('%', _) => break,
                        (ch, Lnum) | (ch, Begin) => {
                            if fs.lnum != 0 {
                                break;
                            }
                            buf.clear();
                            if ch == '0' {
                                fs.leading_zeros = true;
                            } else {
                                buf.push(ch as u8);
                            };
                            next = None;
                            while let Some((ix, ch)) = iter.next() {
                                if !ch.is_digit(/*radix=*/ 10) {
                                    next = Some((ix, ch));
                                    break;
                                }
                                buf.push(ch as u8);
                            }
                            let num = strtoi(str::from_utf8(&buf[..]).unwrap());
                            if num < 0 {
                                break;
                            }
                            fs.lnum = num as usize;
                            stage = Rnum;
                            continue;
                        }
                        (ch, Rnum) => {
                            if fs.rnum != usize::max_value() {
                                break;
                            }
                            if ch != '.' {
                                break;
                            }
                            buf.clear();
                            next = None;
                            while let Some((ix, ch)) = iter.next() {
                                if !ch.is_digit(/*radix=*/ 10) {
                                    next = Some((ix, ch));
                                    break;
                                }
                                buf.push(ch as u8);
                            }
                            let buf_str = str::from_utf8(&buf[..]).unwrap();
                            let num = strtoi(buf_str);
                            if num < 0 {
                                break;
                            }
                            fs.rnum = num as usize;
                            continue;
                        }
                    };
                    next = iter.next();
                }
                // We do not have a complete format specifier, and we have exhausted the string.
                // Just print it out.
                state = Raw(start);
                continue 'outer;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::Cursor;

    macro_rules! sprintf {
        ($fmt:expr $(, $e:expr)*) => {{
            let mut v = Vec::<u8>::new();
            let w = Cursor::new(&mut v);
            printf(w, $fmt, &[$( $e.into() ),*]).expect("printf failure");
            String::from_utf8(v).expect("printf should produce valid utf8")
        }}
    }

    #[test]
    fn basic_printf() {
        use FormatArg::*;
        let mut v = Vec::<u8>::new();
        let w = Cursor::new(&mut v);
        // We don't use the macro here to test the truncation semantics here.
        printf(
            w,
            "Hi %s, to my %d friends %f percent of the time: %g!",
            &[S("there".into()), F(2.5), I(1), F(1.25369E23)],
        )
        .expect("printf failed");
        let s = str::from_utf8(&v[..]).unwrap();
        assert_eq!(
            s,
            "Hi there, to my 2 friends 1.0 percent of the time: 1.25369e23!"
        );

        let s2 = sprintf!("%e %d ~~ %s", 12535, 3, "hi");
        assert_eq!(s2.as_str(), "1.2535e4 3 ~~ hi");
    }

    #[test]
    fn truncation_padding() {
        let s1 = sprintf!("%06o |%-10.3s|", 98, "February");
        assert_eq!(s1.as_str(), "000142 |Feb       |");
        let s2 = sprintf!("|%-10.");
        assert_eq!(s2.as_str(), "|%-10.");
    }

    #[test]
    fn float_rounding() {
        let s1 = sprintf!("%02.2f", 2.375);
        assert_eq!(s1.as_str(), "2.38");
    }
}
