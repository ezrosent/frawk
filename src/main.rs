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
use llvm::IntoRuntime;
use runtime::{
    splitter::{CSVReader, RegexSplitter},
    ChainedReader, LineReader, CHUNK_SIZE,
};
use std::fs::File;
use std::io::{self, BufReader, Write};

// TODO: put jemalloc behind a feature flag
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[derive(Clap, Debug)]
struct Opts {
    #[clap(
        short = "f",
        long = "f",
        help = "a file containing a valid frawk program"
    )]
    program_file: Option<String>,
    #[clap(
        short = "O",
        long = "opt-level",
        default_value = "3",
        help = "the optimization level for the program. Levels 0 through 3 are passed \
to LLVM. To force bytecode interpretation pass level -1. The bytecode interpreter is \
good for debugging, and it will execute much faster for small scripts."
    )]
    opt_level: i32,
    #[clap(short = "o", long = "out-file")]
    out_file: Option<String>,
    #[clap(long = "dump-llvm", help = "dump llvm-ir for input program")]
    dump_llvm: bool,
    #[clap(long = "dump-bytecode", help = "dump typed bytecode for input program")]
    dump_bytecode: bool,
    #[clap(long = "dump-cfg", help = "dump untyped SSA form for input program")]
    dump_cfg: bool,
    #[clap(
        default_value = "32768",
        long = "out-buffer-size",
        help = "output to --out-file is buffered; this flag determines buffer size"
    )]
    out_file_bufsize: usize,
    #[clap(
        long = "csv",
        short = "c",
        help = "input is split according to the rules of csv. \
$0 contains the unescaped line, assigning to any columns including $0 does \
nothing. To drive home the experimental nature of this feature, CSV requires \
AVX2 support on the current CPU"
    )]
    csv: bool,
    #[clap(
        long = "var",
        short = "v",
        multiple = true,
        help = "Has the form <identifier> = <expr>"
    )]
    var: Vec<String>,
    #[clap(
        short = "F",
        help = "Field separator for frawk program. This is interpreted as a regular expression"
    )]
    field_sep: Option<String>,
    program: Option<String>,
    input_files: Vec<String>,
}
macro_rules! fail {
    ($($t:tt)*) => {{
        eprintln!($($t)*);
        std::process::exit(1)
    }}
}

struct Prelude<'a> {
    var_decs: Vec<(&'a str, &'a ast::Expr<'a, 'a, &'a str>)>,
    field_sep: Option<&'a str>,
}

fn open_file_read(f: &str) -> io::BufReader<File> {
    match File::open(f) {
        Ok(f) => BufReader::new(f),
        Err(e) => fail!("failed to open file {}: {}", f, e),
    }
}

fn chained<LR: LineReader>(lr: LR) -> ChainedReader<LR> {
    ChainedReader::new(std::iter::once(lr))
}

fn csv_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    const IS_X64: bool = true;
    #[cfg(not(target_arch = "x86_64"))]
    const IS_X64: bool = false;
    IS_X64 && is_x86_feature_detected!("avx2")
}

fn get_vars<'a, 'b>(
    vars: impl Iterator<Item = &'b str>,
    a: &'a Arena,
) -> Vec<(&'a str, &'a ast::Expr<'a, 'a, &'a str>)> {
    let mut buf = Vec::new();
    let mut stmts = Vec::new();
    for (i, var) in vars.enumerate() {
        buf.clear();
        let var = a.alloc_str(var);
        let lexer = lexer::Tokenizer::new(var);
        let parser = parsing::syntax::VarDefParser::new();
        match parser.parse(a, &mut buf, lexer) {
            Ok(stmt) => stmts.push(stmt),
            Err(e) => fail!(
                "failed to parse var at index {}:\n{}\nerror:{:?}",
                i + 1,
                var,
                e
            ),
        }
    }
    stmts
}

fn get_prelude<'a>(
    a: &'a Arena,
    field_sep: &Option<String>,
    var_decs: &Vec<String>,
) -> Prelude<'a> {
    Prelude {
        field_sep: field_sep.as_ref().map(|s| a.alloc_str(s.as_str())),
        var_decs: get_vars(var_decs.iter().map(|s| s.as_str()), a),
    }
}

fn get_context<'a>(
    prog: &str,
    a: &'a Arena,
    prelude: Prelude<'a>,
) -> cfg::ProgramContext<'a, &'a str> {
    let prog = a.alloc_str(prog);
    let lexer = lexer::Tokenizer::new(prog);
    let mut buf = Vec::new();
    let parser = parsing::syntax::ProgParser::new();
    let stmt = match parser.parse(a, &mut buf, lexer) {
        Ok(mut program) => {
            program.field_sep = prelude.field_sep;
            program.prelude_vardecs = prelude.var_decs;
            a.alloc_v(program)
        }
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

fn run_interp(
    prog: &str,
    stdin: impl LineReader,
    stdout: impl io::Write + 'static,
    field_sep: &Option<String>,
    var_decs: &Vec<String>,
) {
    let a = Arena::default();
    let mut ctx = get_context(prog, &a, get_prelude(&a, field_sep, var_decs));
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
    stdin: impl IntoRuntime,
    stdout: impl io::Write + 'static,
    cfg: llvm::Config,
    field_sep: &Option<String>,
    var_decs: &Vec<String>,
) {
    let a = Arena::default();
    let mut ctx = get_context(prog, &a, get_prelude(&a, field_sep, var_decs));
    if let Err(e) = compile::run_llvm(&mut ctx, stdin, stdout, cfg) {
        fail!("error compiling llvm: {}", e)
    }
}

fn dump_llvm(
    prog: &str,
    cfg: llvm::Config,
    field_sep: &Option<String>,
    var_decs: &Vec<String>,
) -> String {
    let a = Arena::default();
    let mut ctx = get_context(prog, &a, get_prelude(&a, field_sep, var_decs));
    match compile::dump_llvm(&mut ctx, cfg) {
        Ok(s) => s,
        Err(e) => fail!("error compiling llvm: {}", e),
    }
}

fn dump_bytecode(prog: &str, field_sep: &Option<String>, var_decs: &Vec<String>) -> String {
    use std::io::Cursor;
    let a = Arena::default();
    let mut ctx = get_context(prog, &a, get_prelude(&a, field_sep, var_decs));
    let fake_inp: Box<dyn io::Read> = Box::new(Cursor::new(vec![]));
    let fake_out: Box<dyn io::Write> = Box::new(Cursor::new(vec![]));
    let interp = match compile::bytecode(
        &mut ctx,
        chained(CSVReader::new(fake_inp, "unuse")),
        fake_out,
    ) {
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
    if opts.csv && !csv_supported() {
        fail!("CSV requires an x86 processor with AVX2 support");
    }
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
            dump_llvm(program_string.as_str(), config, &opts.field_sep, &opts.var)
        );
    }
    if opts.dump_bytecode {
        let _ = write!(
            std::io::stdout(),
            "{}",
            dump_bytecode(program_string.as_str(), &opts.field_sep, &opts.var),
        );
    }
    if opts.dump_cfg {
        let a = Arena::default();
        let ctx = get_context(
            program_string.as_str(),
            &a,
            get_prelude(&a, &opts.field_sep, &opts.var),
        );
        let mut stdout = std::io::stdout();
        let _ = ctx.dbg_print(&mut stdout);
    }
    if skip_output {
        return;
    }
    macro_rules! with_inp {
        ($inp:ident, $body:expr) => {
            if opts.input_files.len() == 0 {
                let _reader: Box<dyn io::Read> = Box::new(io::stdin());
                if opts.csv {
                    let $inp = chained(CSVReader::new(_reader, "-"));
                    $body
                } else {
                    let $inp = chained(RegexSplitter::new(_reader, CHUNK_SIZE, "-"));
                    $body
                }
            } else if opts.csv {
                let iter = opts.input_files.iter().cloned().map(|file| {
                    let reader: Box<dyn io::Read> = Box::new(open_file_read(file.as_str()));
                    CSVReader::new(reader, file)
                });
                let $inp = ChainedReader::new(iter);
                $body
            } else {
                let iter = opts.input_files.iter().cloned().map(|file| {
                    let reader: Box<dyn io::Read> = Box::new(open_file_read(file.as_str()));
                    RegexSplitter::new(reader, CHUNK_SIZE, file)
                });
                let $inp = ChainedReader::new(iter);
                $body
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
        with_io!(|inp, oup| run_interp(
            program_string.as_str(),
            inp,
            oup,
            &opts.field_sep,
            &opts.var,
        ));
    } else {
        with_io!(|inp, oup| run_llvm(
            program_string.as_str(),
            inp,
            oup,
            llvm::Config {
                opt_level: opts.opt_level as usize
            },
            &opts.field_sep,
            &opts.var,
        ));
    }
}
