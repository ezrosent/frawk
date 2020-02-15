use crate::common::Result;

use std::fmt;
use std::io;
// Create an enum
// Ocatal(Leading(Optional<str>)),
// Float(pre(optional(int)), post(optional(int))),
// decimal, hex, etc.
// Const(str),
//
// Evaluate it in sequence by write! ing it witht he existing fmt library.
//
// Or just do it in place.
// Do we need a "growablechunk?"
trait Formatable {
    fn to_float(&self) -> f64;
    fn to_int(&self) -> i64;
    fn to_str(&self) -> &str;
}

struct FormatSpec {
    // leading '-' ?
    minus: bool,
    // number to the left of '.', if any
    lnum: Option<usize>,
    // number to the right of '.', if any
    rnum: Option<usize>,
    // format specifier: e.g. c, d, s, x.
    spec: char,
}

// TODO: consider just using an enum instead of dynamic dispatch; we'll only ever pass 3 things in
// here.

fn process_format(w: impl io::Write, spec: &str, arg: &dyn Formatable) -> Result<()> {
    unimplemented!()
}

fn write_str(mut w: impl io::Write, s: &str) -> Result<()> {
    match write!(w, "{}", s) {
        Ok(()) => Ok(()),
        Err(e) => err!("formatter: {}", e),
    }
}

fn printf(mut w: impl io::Write, spec: &str, mut args: &[&dyn Formatable]) -> Result<()> {
    #[derive(Copy, Clone)]
    enum State {
        // Byte index of start of string
        Raw(usize),
        // Byte index of percent sign
        Format(usize),
        Formatted(usize /* start */, usize /* one-past-the-end */),
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
    let mut next_arg = || {
        if args.len() == 0 {
            panic!("nyi")
        } else {
            let res = args[0];
            args = &args[1..];
            res
        }
    };
    loop {
        let next = iter.next();
        match (state, next) {
            (Raw(start), None) => return write_str(w, &spec[start..]),
            (Raw(start), Some((ix, '%'))) => {
                write_str(&mut w, &spec[start..ix])?;
                state = Format(ix);
            }
            (Raw(_), Some(_)) => continue,
            (Format(start), None) => return process_format(w, &spec[start..], next_arg()),
            (Format(start), Some((ix, '%'))) if ix != start + 1 => {
                process_format(&mut w, &spec[start..ix], next_arg())?;
                state = Format(ix);
            }
            (Format(start), Some((ix, ch))) => {
                if match ch {
                    'f' | 'c' | 'd' | 'e' | 'g' | 'o' | 's' | 'x' | '%' => true,
                    _ => false,
                } {
                    state = Formatted(start, ix + 1);
                }
            }
            (Formatted(start, end), next) => {
                process_format(&mut w, &spec[start..end], next_arg())?;
                state = next_state!(next);
            }
        }
    }
}
