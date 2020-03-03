#![feature(core_intrinsics)]
#![feature(test)]
#[macro_use]
pub mod common;
pub mod arena;
pub mod ast;
pub mod builtins;
pub mod bytecode;
pub mod cfg;
pub mod compile;
mod display;
pub mod dom;
pub mod harness;
pub mod interp;
pub mod lexer;
pub mod llvm;
#[allow(unused_parens)]
pub mod parsing;
pub mod runtime;
#[cfg(test)]
mod test_string_constants;
pub mod types;
extern crate clap;
extern crate elsa;
extern crate hashbrown;
extern crate jemallocator;
extern crate lalrpop_util;
extern crate lazy_static;
extern crate libc;
extern crate llvm_sys;
extern crate petgraph;
extern crate rand;
extern crate regex;
extern crate ryu;
extern crate smallvec;
extern crate stable_deref_trait;
extern crate unicode_xid;

use clap::Clap;

use arena::Arena;
use std::fs::File;
use std::io::{self, BufReader, Write};

// TODO: put jemalloc behind a feature flag
// #[global_allocator]
// static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[derive(Clap, Debug)]
struct Opts {
    program: Option<String>,
    input_files: Vec<String>,
    #[clap(short = "f", long = "f")]
    program_file: Option<String>,
    #[clap(short = "O", long = "opt-level", default_value = "-1")]
    opt_level: i32,
    #[clap(short = "o", long = "out-file")]
    out_file: Option<String>,
}
macro_rules! fail {
    ($($t:tt)*) => {{
        eprintln!($($t)*);
        std::process::exit(1)
    }}
}
fn open_file_read(f: &str) -> io::BufReader<File> {
    match File::open(f) {
        Ok(f) => BufReader::new(f),
        Err(e) => fail!("failed to open file {}: {}", f, e),
    }
}

fn get_context<'a>(prog: &str, a: &'a Arena) -> cfg::ProgramContext<'a, &'a str> {
    let prog = a.alloc_str(prog);
    let lexer = lexer::Tokenizer::new(prog);
    let mut buf = Vec::new();
    let parser = parsing::syntax::ProgParser::new();
    let stmt = match parser.parse(a, &mut buf, lexer) {
        Ok(program) => a.alloc_v(program),
        Err(e) => {
            let mut ix = 0;
            let mut msg: Vec<u8> = "failed to parse program:\n======\n".as_bytes().into();
            for line in prog.lines() {
                write!(&mut msg, "[{:3}] {}\n", ix, line).unwrap();
                ix += line.len() + 1;
            }
            write!(&mut msg, "=====\nError: {:?}\n", e).unwrap();
            fail!("{}", String::from_utf8(msg).unwrap())
        }
    };
    match cfg::ProgramContext::from_prog(a, stmt) {
        Ok(ctx) => ctx,
        Err(e) => fail!("failed to create program context: {}", e),
    }
}

fn run_interp(prog: &str, stdin: impl io::Read + 'static, stdout: impl io::Write + 'static) {
    let a = Arena::default();
    let mut ctx = get_context(prog, &a);
    let mut interp = match compile::bytecode(&mut ctx, stdin, stdout) {
        Ok(ctx) => ctx,
        Err(e) => fail!("bytecode compilation failure: {}", e),
    };
    if let Err(e) = interp.run() {
        fail!("fatal error during execution: {}", e);
    }
}

fn run_llvm(
    prog: &str,
    stdin: impl io::Read + 'static,
    stdout: impl io::Write + 'static,
    cfg: llvm::Config,
) {
    let a = Arena::default();
    let mut ctx = get_context(prog, &a);
    if let Err(e) = compile::run_llvm(&mut ctx, stdin, stdout, cfg) {
        fail!("error compiling llvm: {}", e)
    }
}

fn main() {
    let opts: Opts = Opts::parse();
    let program_string = {
        if let Some(p) = &opts.program {
            p.clone()
        } else if let Some(pfile) = &opts.program_file {
            match std::fs::read_to_string(pfile) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("failed to read program from {}: {}", pfile, e);
                    std::process::exit(1)
                }
            }
        } else {
            eprintln!("must specify program at command line, or in a file via -f");
            std::process::exit(1)
        }
    };
    macro_rules! with_inp {
        ($inp:ident, $body:expr) => {
            match opts.input_files.len() {
                0 => {
                    let $inp = std::io::stdin();
                    $body
                }
                1 => {
                    let $inp = open_file_read(opts.input_files[0].as_str());
                    $body
                }
                _ => {
                    eprintln!("multiple files not yet implemented");
                    std::process::exit(1);
                }
            }
        };
    }
    macro_rules! with_io {
        (|$inp:ident, $out:ident| $body:expr) => {
            match opts.out_file {
                Some(oup) => {
                    let $out = io::BufWriter::new(
                        File::open(oup.as_str())
                            .unwrap_or_else(|e| fail!("failed to open {}: {}", oup.as_str(), e)),
                    );
                    with_inp!($inp, $body);
                }
                None => {
                    let $out = std::io::stdout();
                    with_inp!($inp, $body);
                }
            }
        };
    }

    if opts.opt_level < 0 {
        with_io!(|inp, oup| run_interp(program_string.as_str(), inp, oup));
    } else if opts.opt_level > 3 {
        fail!("opt levels can only be negative, or in the range [0, 3]");
    } else {
        with_io!(|inp, oup| run_llvm(
            program_string.as_str(),
            inp,
            oup,
            llvm::Config {
                opt_level: opts.opt_level as usize
            }
        ));
    }
}
