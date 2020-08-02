#![recursion_limit = "256"]
#![feature(core_intrinsics)]
#![feature(test)]
#![feature(write_all_vectored)]
#[macro_use]
pub mod common;
pub mod arena;
pub mod ast;
pub mod builtins;
pub mod bytecode;
pub mod cfg;
pub mod compile;
pub mod cross_stage;
mod display;
pub mod dom;
#[cfg(test)]
pub mod harness;
pub mod interp;
pub mod lexer;
pub mod llvm;
#[allow(unused_parens)] // Warnings appear in generated code
pub mod parsing;
pub mod pushdown;
pub mod runtime;
#[cfg(test)]
mod test_string_constants;
pub mod types;
extern crate clap;
extern crate crossbeam_channel;
extern crate elsa;
extern crate hashbrown;
#[cfg(feature = "use_jemalloc")]
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
use cfg::Escaper;
use llvm::IntoRuntime;
use runtime::{
    splitter::{
        batch::{bytereader_supported, ByteReader, CSVReader, InputFormat},
        regex::RegexSplitter,
        DefaultSplitter,
    },
    ChainedReader, LineReader, CHUNK_SIZE,
};
use std::fs::File;
use std::io::{self, BufReader, Write};

#[cfg(feature = "use_jemalloc")]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[derive(Clap, Debug)]
struct Opts {
    #[clap(
        short = 'f',
        long = "f",
        about = "a file containing a valid frawk program"
    )]
    program_file: Option<String>,
    #[clap(
        short = 'O',
        long = "opt-level",
        default_value = "3",
        about = "the optimization level for the program. Levels 0 through 3 are passed \
to LLVM. To force bytecode interpretation pass level -1. The bytecode interpreter is \
good for debugging, and it will execute much faster for small scripts."
    )]
    opt_level: i32,
    #[clap(long = "out-file")]
    out_file: Option<String>,
    #[clap(long = "dump-llvm", about = "dump llvm-ir for input program")]
    dump_llvm: bool,
    #[clap(
        long = "dump-bytecode",
        about = "dump typed bytecode for input program"
    )]
    dump_bytecode: bool,
    #[clap(long = "dump-cfg", about = "dump untyped SSA form for input program")]
    dump_cfg: bool,
    #[clap(
        long = "input-format",
        short = 'i',
        about = "Legal values are csv, tsv. Input is split according to the rules of (csv|tsv). \
$0 contains the unescaped line, assigning to any columns including $0 does \
nothing."
    )]
    input_format: Option<String>,
    #[clap(
        long = "var",
        short = 'v',
        multiple = true,
        about = "Has the form <identifier> = <expr>"
    )]
    var: Vec<String>,
    #[clap(
        short = 'F',
        about = "Field separator for frawk program. This is interpreted as a regular expression"
    )]
    field_sep: Option<String>,
    #[clap(
        short = 'b',
        long = "bytecode",
        about = "Execute the program with the bytecode interpreter."
    )]
    bytecode: bool,
    #[clap(
        short = 'o',
        long = "output-format",
        about = "Legal values are csv, tsv. If set, records output via print \
are escaped accoring to the rules of the corresponding format"
    )]
    output_format: Option<String>,
    program: Option<String>,
    input_files: Vec<String>,
}
macro_rules! fail {
    ($($t:tt)*) => {{
        eprintln_ignore!($($t)*);
        std::process::exit(1)
    }}
}

struct RawPrelude {
    var_decs: Vec<String>,
    field_sep: Option<String>,
    output_sep: Option<&'static str>,
    output_record_sep: Option<&'static str>,
    escaper: Escaper,
}

struct Prelude<'a> {
    var_decs: Vec<(&'a str, &'a ast::Expr<'a, 'a, &'a str>)>,
    field_sep: Option<&'a str>,
    output_sep: Option<&'a str>,
    output_record_sep: Option<&'a str>,
    escaper: Escaper,
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
    IS_X64 && is_x86_feature_detected!("sse2")
}

fn get_vars<'a, 'b>(
    vars: impl Iterator<Item = &'b str>,
    a: &'a Arena,
    buf: &mut Vec<u8>,
) -> Vec<(&'a str, &'a ast::Expr<'a, 'a, &'a str>)> {
    let mut stmts = Vec::new();
    for (i, var) in vars.enumerate() {
        buf.clear();
        let var = a.alloc_str(var);
        let lexer = lexer::Tokenizer::new(var);
        let parser = parsing::syntax::VarDefParser::new();
        match parser.parse(a, buf, lexer) {
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

fn get_prelude<'a>(a: &'a Arena, raw: &RawPrelude) -> Prelude<'a> {
    let mut buf = Vec::new();
    let output_sep = raw
        .output_sep
        .map(|s| lexer::parse_string_literal(s, a, &mut buf));
    let output_record_sep = raw
        .output_record_sep
        .map(|s| lexer::parse_string_literal(s, a, &mut buf));
    let field_sep = raw
        .field_sep
        .as_ref()
        .map(|s| lexer::parse_string_literal(s.as_str(), a, &mut buf));
    Prelude {
        field_sep,
        var_decs: get_vars(raw.var_decs.iter().map(|s| s.as_str()), a, &mut buf),
        escaper: raw.escaper,
        output_sep,
        output_record_sep,
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
            program.output_sep = prelude.output_sep;
            program.output_record_sep = prelude.output_record_sep;
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
    match cfg::ProgramContext::from_prog(a, stmt, prelude.escaper) {
        Ok(ctx) => ctx,
        Err(e) => fail!("failed to create program context: {}", e),
    }
}

fn run_interp_with_context<'a>(
    mut ctx: cfg::ProgramContext<'a, &'a str>,
    stdin: impl LineReader,
    ff: impl runtime::writers::FileFactory,
) {
    let mut interp = match compile::bytecode(&mut ctx, stdin, ff) {
        Ok(ctx) => ctx,
        Err(e) => fail!("bytecode compilation failure: {}", e),
    };
    if let Err(e) = interp.run() {
        fail!("fatal error during execution: {}", e);
    }
}

fn run_llvm_with_context<'a>(
    mut ctx: cfg::ProgramContext<'a, &'a str>,
    stdin: impl IntoRuntime,
    ff: impl runtime::writers::FileFactory,
    cfg: llvm::Config,
) {
    if let Err(e) = compile::run_llvm(&mut ctx, stdin, ff, cfg) {
        fail!("error compiling llvm: {}", e)
    }
}

fn dump_llvm(prog: &str, cfg: llvm::Config, raw: &RawPrelude) -> String {
    let a = Arena::default();
    let mut ctx = get_context(prog, &a, get_prelude(&a, raw));
    match compile::dump_llvm(&mut ctx, cfg) {
        Ok(s) => s,
        Err(e) => fail!("error compiling llvm: {}", e),
    }
}

fn dump_bytecode(prog: &str, raw: &RawPrelude) -> String {
    use std::io::Cursor;
    let a = Arena::default();
    let mut ctx = get_context(prog, &a, get_prelude(&a, raw));
    let fake_inp: Box<dyn io::Read> = Box::new(Cursor::new(vec![]));
    let interp = match compile::bytecode(
        &mut ctx,
        chained(CSVReader::new(
            fake_inp,
            InputFormat::CSV,
            CHUNK_SIZE,
            "unused",
        )),
        runtime::writers::default_factory(),
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
    let mut opts: Opts = Opts::parse();
    if opts.input_format.is_some() && !csv_supported() {
        fail!("CSV/TSV requires an x86 processor with SSE2 support");
    }
    let ifmt = match opts.input_format.as_ref().map(String::as_str) {
        Some("csv") | Some("CSV") => Some(InputFormat::CSV),
        Some("tsv") | Some("TSV") => Some(InputFormat::TSV),
        Some(x) => fail!("invalid input format: {}", x),
        None => None,
    };
    let program_string = {
        if let Some(pfile) = &opts.program_file {
            match std::fs::read_to_string(pfile) {
                Ok(p) => {
                    // We specified a file on the command line, so the "program" will be
                    // interpreted as another input file.
                    if let Some(p) = &opts.program {
                        opts.input_files.push(p.clone());
                    }
                    p
                }
                Err(e) => fail!("failed to read program from {}: {}", pfile, e),
            }
        } else if let Some(p) = &opts.program {
            p.clone()
        } else {
            fail!("must specify program at command line, or in a file via -f");
        }
    };
    let (escaper, output_sep, output_record_sep) =
        match opts.output_format.as_ref().map(String::as_str) {
            Some("csv") => (Escaper::CSV, Some(","), Some("\r\n")),
            Some("tsv") => (Escaper::TSV, Some("\t"), Some("\n")),
            Some(s) => fail!(
                "invalid output format {:?}; expected csv or tsv (or the empty string)",
                s
            ),
            None => (Escaper::Identity, None, None),
        };
    let raw = RawPrelude {
        field_sep: opts.field_sep.clone(),
        var_decs: opts.var.clone(),
        output_sep,
        escaper,
        output_record_sep,
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
            dump_llvm(program_string.as_str(), config, &raw),
        );
    }
    if opts.dump_bytecode {
        let _ = write!(
            std::io::stdout(),
            "{}",
            dump_bytecode(program_string.as_str(), &raw),
        );
    }
    if opts.dump_cfg {
        let a = Arena::default();
        let ctx = get_context(program_string.as_str(), &a, get_prelude(&a, &raw));
        let mut stdout = std::io::stdout();
        let _ = ctx.dbg_print(&mut stdout);
    }
    if skip_output {
        return;
    }

    // This horrid macro is here because all of the different ways of reading input are different
    // types, making functions hard to write. Still, there must be something to be done to clean
    // this up here.
    macro_rules! with_inp {
        ($analysis:expr, $inp:ident, $body:expr) => {
            if opts.input_files.len() == 0 {
                let _reader: Box<dyn io::Read> = Box::new(io::stdin());
                match (ifmt, $analysis) {
                    (Some(ifmt), _) => {
                        let $inp = chained(CSVReader::new(_reader, ifmt, CHUNK_SIZE, "-"));
                        $body
                    }
                    (
                        None,
                        cfg::SepAssign::Potential {
                            field_sep,
                            record_sep,
                        },
                    ) => {
                        let field_sep = field_sep.unwrap_or(" ");
                        let record_sep = record_sep.unwrap_or("\n");
                        if field_sep.len() == 1 && record_sep.len() == 1 {
                            if field_sep == " " && record_sep == "\n" {
                                let $inp = chained(DefaultSplitter::new(_reader, CHUNK_SIZE, "-"));
                                $body
                            } else if bytereader_supported() {
                                let br = ByteReader::new(
                                    _reader,
                                    field_sep.as_bytes()[0],
                                    record_sep.as_bytes()[0],
                                    CHUNK_SIZE,
                                    "-",
                                );
                                let $inp = chained(br);
                                $body
                            } else {
                                let _reader: Box<dyn io::Read> = Box::new(io::stdin());
                                let $inp = chained(RegexSplitter::new(_reader, CHUNK_SIZE, "-"));
                                $body
                            }
                        } else {
                            let $inp = chained(RegexSplitter::new(_reader, CHUNK_SIZE, "-"));
                            $body
                        }
                    }
                    (None, cfg::SepAssign::Unsure) => {
                        let $inp = chained(RegexSplitter::new(_reader, CHUNK_SIZE, "-"));
                        $body
                    }
                }
            } else if let Some(ifmt) = ifmt {
                let iter = opts.input_files.iter().cloned().map(|file| {
                    let reader: Box<dyn io::Read> = Box::new(open_file_read(file.as_str()));
                    CSVReader::new(reader, ifmt, CHUNK_SIZE, file)
                });
                let $inp = ChainedReader::new(iter);
                $body
            } else {
                match $analysis {
                    cfg::SepAssign::Potential {
                        field_sep,
                        record_sep,
                    } => {
                        let field_sep = field_sep.unwrap_or(" ");
                        let record_sep = record_sep.unwrap_or("\n");
                        if field_sep.len() == 1 && record_sep.len() == 1 {
                            let br_valid = bytereader_supported();
                            if field_sep == " " && record_sep == "\n" {
                                let iter = opts.input_files.iter().cloned().map(|file| {
                                    let reader: Box<dyn io::Read> =
                                        Box::new(open_file_read(file.as_str()));
                                    DefaultSplitter::new(reader, CHUNK_SIZE, file)
                                });
                                let $inp = ChainedReader::new(iter);
                                $body
                            } else if br_valid {
                                let iter = opts.input_files.iter().cloned().map(move |file| {
                                    let reader: Box<dyn io::Read> =
                                        Box::new(open_file_read(file.as_str()));
                                    ByteReader::new(
                                        reader,
                                        field_sep.as_bytes()[0],
                                        record_sep.as_bytes()[0],
                                        CHUNK_SIZE,
                                        file,
                                    )
                                });
                                let $inp = ChainedReader::new(iter);
                                $body
                            } else {
                                let iter = opts.input_files.iter().cloned().map(|file| {
                                    let reader: Box<dyn io::Read> =
                                        Box::new(open_file_read(file.as_str()));
                                    RegexSplitter::new(reader, CHUNK_SIZE, file)
                                });
                                let $inp = ChainedReader::new(iter);
                                $body
                            }
                        } else {
                            let iter = opts.input_files.iter().cloned().map(|file| {
                                let reader: Box<dyn io::Read> =
                                    Box::new(open_file_read(file.as_str()));
                                RegexSplitter::new(reader, CHUNK_SIZE, file)
                            });
                            let $inp = ChainedReader::new(iter);
                            $body
                        }
                    }
                    cfg::SepAssign::Unsure => {
                        let iter = opts.input_files.iter().cloned().map(|file| {
                            let reader: Box<dyn io::Read> = Box::new(open_file_read(file.as_str()));
                            RegexSplitter::new(reader, CHUNK_SIZE, file)
                        });
                        let $inp = ChainedReader::new(iter);
                        $body
                    }
                }
            }
        };
    }
    let a = Arena::default();
    let ctx = get_context(program_string.as_str(), &a, get_prelude(&a, &raw));
    let analysis_result = ctx.analyze_sep_assignments();
    macro_rules! with_io {
        (|$inp:ident, $out:ident| $body:expr) => {
            match opts.out_file {
                Some(oup) => {
                    let $out = runtime::writers::factory_from_file(oup.as_str())
                        .unwrap_or_else(|e| fail!("failed to open {}: {}", oup.as_str(), e));
                    with_inp!(analysis_result, $inp, $body);
                }
                None => {
                    let $out = runtime::writers::default_factory();
                    with_inp!(analysis_result, $inp, $body);
                }
            }
        };
    }

    if opts.bytecode {
        opts.opt_level = -1;
    }

    if opts.opt_level < 0 {
        with_io!(|inp, oup| run_interp_with_context(ctx, inp, oup))
    } else {
        with_io!(|inp, oup| run_llvm_with_context(
            ctx,
            inp,
            oup,
            llvm::Config {
                opt_level: opts.opt_level as usize
            },
        ));
    }
}
