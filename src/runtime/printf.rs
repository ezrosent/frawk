/// This module implements much of printf in AWK, except for some semantics around the rounding of
/// floating point numbers (they are currently treated the same as integers are).
///
/// We currently lean on std::fmt to do the heavy lifting. Most of the code here just parses format
/// strings.
///
/// TODO: handle proper semantics for floats. The main Ryu repo includes support for variable
/// precision and scientific notation, we should read the paper and implement it using that.
use crate::common::Result;
use crate::runtime::{convert, strton::strtoi, Float, Int, Str};

use std::convert::TryFrom;
use std::fmt;
use std::io::{self, Write};
use std::iter::repeat;
use std::str;

type SmallVec<T> = smallvec::SmallVec<[T; 32]>;

#[derive(Default)]
struct StackWriter(pub SmallVec<u8>);

impl StackWriter {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl io::Write for StackWriter {
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

struct FormatSpec {
    // leading '-' ? -- left justification.
    minus: bool,
    // number to the left of '.', if any
    leading_zeros: bool,
    // padding
    // TODO: only zero-pad with a finite float
    lnum: usize,
    // maximum string width
    // TODO: implement rounding.
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

fn spec_base(mut w: impl io::Write, spec: u8, arg: &FormatArg) -> Result<()> {
    let mut buf = StackWriter::default();
    match spec as char {
        'f' => {
            // Ryu prints some things a bit differently than most awk implementations.
            // `write!(w, "{}", arg.to_float())` is a bit closer.
            let mut buf = ryu::Buffer::new();
            wrap_result(write!(w, "{}", buf.format(arg.to_float())))
        }
        'c' => {
            let res = match char::try_from(arg.to_int() as u32) {
                Ok(ch) => ch,
                Err(e) => return err!("invalid character:  {}", e),
            };
            wrap_result(write!(w, "{}", res))
        }
        'd' => wrap_result(write!(w, "{}", arg.to_int())),
        'e' => wrap_result(write!(w, "{:e}", arg.to_float())),
        'g' => {
            // %g means "pick the shorter of standard and scientific notation". We do the obvious
            // thing of computing both and writing out the smaller one.
            let f = arg.to_float();
            wrap_result(write!(&mut buf, "{}", f))?;
            let l1 = buf.len();
            wrap_result(write!(&mut buf, "{:e}", f))?;
            let l2 = buf.len() - l1;
            if l1 < l2 {
                wrap_result(write!(w, "{}", unsafe {
                    str::from_utf8_unchecked(&buf.0[0..l1])
                }))
            } else {
                wrap_result(write!(w, "{}", unsafe {
                    str::from_utf8_unchecked(&buf.0[l1..(l1 + l2)])
                }))
            }
        }
        'o' => wrap_result(write!(w, "{:o}", arg.to_int())),
        's' => arg.with_str(|s| wrap_result(write!(w, "{}", s))),
        'x' => wrap_result(write!(w, "{:x}", arg.to_int())),
        _ => return err!("invalid format spec: {}", spec),
    }
}

fn process_spec(
    mut w: impl io::Write,
    FormatSpec {
        minus,
        leading_zeros,
        lnum,
        rnum,
        spec,
    }: FormatSpec,
    arg: &FormatArg,
) -> Result<()> {
    let mut padding = StackWriter::default();
    let mut buf = StackWriter::default();
    spec_base(&mut buf, spec, arg)?;
    if rnum < buf.len() {
        buf.0.truncate(rnum);
    }
    let padding_bytes = lnum.saturating_sub(buf.len());
    if padding_bytes > 0 {
        padding
            .0
            .extend(repeat(if leading_zeros { '0' } else { ' ' } as u8).take(padding_bytes));
    }
    let padding_str = unsafe { str::from_utf8_unchecked(&padding.0[..]) };
    let buf_str = unsafe { str::from_utf8_unchecked(&buf.0[..]) };
    if minus {
        // minus => left justified. Padding comes second
        wrap_result(write!(&mut w, "{}", buf_str))?;
        wrap_result(write!(&mut w, "{}", padding_str))?;
    } else {
        wrap_result(write!(&mut w, "{}", padding_str))?;
        wrap_result(write!(&mut w, "{}", buf_str))?;
    }
    Ok(())
}

fn wrap_result(r: std::result::Result<(), impl fmt::Display>) -> Result<()> {
    match r {
        Ok(()) => Ok(()),
        Err(e) => err!("formatter: {}", e),
    }
}

fn write_str(mut w: impl io::Write, s: &str) -> Result<()> {
    wrap_result(write!(w, "{}", s))
}

pub(crate) fn printf(mut w: impl io::Write, spec: &str, mut args: &[FormatArg]) -> Result<()> {
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
                            process_spec(&mut w, fs, next_arg())?;
                            state = Raw(ix + 1);
                            continue 'outer;
                        }
                        (ch, _) if is_spec(ch) => {
                            fs.spec = ch as u8;
                            process_spec(&mut w, fs, next_arg())?;
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
}
