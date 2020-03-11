#![recursion_limit = "256"]
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
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

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
    #[clap(long = "dump-llvm")]
    dump_llvm: bool,
    #[clap(long = "dump-bytecode")]
    dump_bytecode: bool,
    #[clap(long = "dump-cfg")]
    dump_cfg: bool,
    #[clap(
        default_value = "32768",
        long = "out-buffer-size",
        help = "output to --out-file is buffered; this flag determines buffer size"
    )]
    out_file_bufsize: usize,
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

fn dump_llvm(prog: &str, cfg: llvm::Config) -> String {
    let a = Arena::default();
    let mut ctx = get_context(prog, &a);
    match compile::dump_llvm(&mut ctx, cfg) {
        Ok(s) => s,
        Err(e) => fail!("error compiling llvm: {}", e),
    }
}

fn dump_bytecode(prog: &str) -> String {
    use std::io::Cursor;
    let a = Arena::default();
    let mut ctx = get_context(prog, &a);
    let fake_io = Cursor::new(vec![]);
    let interp = match compile::bytecode(&mut ctx, fake_io.clone(), fake_io) {
        Ok(ctx) => ctx,
        Err(e) => fail!("bytecode compilation failure: {}", e),
    };
    let mut v = Vec::<u8>::new();
    for (i, func) in interp.instrs().iter().enumerate() {
        write!(&mut v, "function {} {{\n", i).unwrap();
        for (j, inst) in func.iter().enumerate() {
            write!(&mut v, "\t[{:2}] {:?}\n", j, inst).unwrap();
        }
        write!(&mut v, "}}\n").unwrap();
    }
    String::from_utf8(v).unwrap()
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
    if opts.opt_level > 3 {
        fail!("opt levels can only be negative, or in the range [0, 3]");
    }
    let skip_output = opts.dump_llvm || opts.dump_bytecode || opts.dump_cfg;
    if opts.dump_llvm {
        let config = llvm::Config {
            opt_level: if opts.opt_level < 0 {
                3
            } else {
                opts.opt_level as usize
            },
        };
        let _ = write!(
            std::io::stdout(),
            "{}",
            dump_llvm(program_string.as_str(), config)
        );
    }
    if opts.dump_bytecode {
        let _ = write!(
            std::io::stdout(),
            "{}",
            dump_bytecode(program_string.as_str()),
        );
    }
    if opts.dump_cfg {
        let a = Arena::default();
        let ctx = get_context(program_string.as_str(), &a);
        let mut stdout = std::io::stdout();
        let _ = ctx.dbg_print(&mut stdout);
    }
    if skip_output {
        return;
    }
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
                    let $out = io::BufWriter::with_capacity(
                        opts.out_file_bufsize,
                        File::create(oup.as_str())
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
