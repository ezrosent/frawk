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
#[cfg(feature = "llvm_backend")]
pub mod llvm;
#[allow(unused_parens)] // Warnings appear in generated code
pub mod parsing;
pub mod pushdown;
pub mod runtime;
#[cfg(test)]
mod test_string_constants;
pub mod types;
extern crate cfg_if;
extern crate clap;
extern crate crossbeam;
extern crate crossbeam_channel;
extern crate elsa;
extern crate hashbrown;
#[cfg(feature = "use_jemalloc")]
extern crate jemallocator;
extern crate lalrpop_util;
extern crate lazy_static;
extern crate libc;
#[cfg(feature = "llvm_backend")]
extern crate llvm_sys;
extern crate memchr;
extern crate num_cpus;
extern crate petgraph;
extern crate rand;
extern crate regex;
extern crate ryu;
extern crate smallvec;
extern crate stable_deref_trait;
extern crate unicode_xid;

use clap::{App, Arg};

use arena::Arena;
use cfg::Escaper;
use common::{ExecutionStrategy, Stage};
#[cfg(feature = "llvm_backend")]
use llvm::IntoRuntime;
use runtime::{
    splitter::{
        batch::{ByteReader, CSVReader, InputFormat},
        regex::RegexSplitter,
    },
    ChainedReader, LineReader, CHUNK_SIZE,
};
use std::fs::File;
use std::io::{self, BufReader, Write};
use std::iter::once;

#[cfg(feature = "use_jemalloc")]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

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
    stage: Stage<()>,
}

struct Prelude<'a> {
    var_decs: Vec<(&'a str, &'a ast::Expr<'a, 'a, &'a str>)>,
    field_sep: Option<&'a str>,
    output_sep: Option<&'a str>,
    output_record_sep: Option<&'a str>,
    escaper: Escaper,
    stage: Stage<()>,
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
        let mut _prog = ast::Prog::from_stage(Stage::Main(()));
        match parser.parse(a, buf, &mut _prog, lexer) {
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
        stage: raw.stage.clone(),
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
    let mut prog = ast::Prog::from_stage(prelude.stage.clone());
    let stmt = match parser.parse(a, &mut buf, &mut prog, lexer) {
        Ok(()) => {
            prog.field_sep = prelude.field_sep;
            prog.prelude_vardecs = prelude.var_decs;
            prog.output_sep = prelude.output_sep;
            prog.output_record_sep = prelude.output_record_sep;
            a.alloc_v(prog)
        }
        Err(e) => {
            fail!("{}", e);
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
    num_workers: usize,
) {
    let mut interp = match compile::bytecode(&mut ctx, stdin, ff, num_workers) {
        Ok(ctx) => ctx,
        Err(e) => fail!("bytecode compilation failure: {}", e),
    };
    if let Err(e) = interp.run() {
        fail!("fatal error during execution: {}", e);
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "llvm_backend")] {
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

        const DEFAULT_OPT_LEVEL: i32 = 3;
    } else {
        const DEFAULT_OPT_LEVEL: i32 = -1;
    }
}

fn dump_bytecode(prog: &str, raw: &RawPrelude) -> String {
    use std::io::Cursor;
    let a = Arena::default();
    let mut ctx = get_context(prog, &a, get_prelude(&a, raw));
    let fake_inp: Box<dyn io::Read + Send> = Box::new(Cursor::new(vec![]));
    let interp = match compile::bytecode(
        &mut ctx,
        chained(CSVReader::new(
            once((fake_inp, String::from("unused"))),
            InputFormat::CSV,
            CHUNK_SIZE,
            /*check_utf8=*/ false,
            ExecutionStrategy::Serial,
        )),
        runtime::writers::default_factory(),
        /*num_workers=*/ 1,
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
    #[allow(unused_mut)]
    let mut app = App::new("frawk")
        .version("0.2")
        .author("Eli R.")
        .about("frawk is a pattern scanning and (semi-structured) text processing language")
        .arg("-f, --program-file=[FILE] 'a file containing frawk program'")
        .arg(Arg::new("opt-level")
             .long("opt-level")
             .short('O')
             .about("the optimization level for the program. Positive levels determine the optimization level for LLVM. Level -1 forces bytecode interpretation")
             .possible_values(&["-1", "0", "1", "2", "3"]))
        .arg("--out-file=[FILE] 'the output file used in place of standard input'")
        .arg("--utf8 'validate all input as UTF-8, returning an error if it is invalid'")
        .arg("--dump-cfg 'print untyped SSA form for input program'")
        .arg("--dump-bytecode 'print bytecode for input program'")
        .arg(Arg::new("input-format")
             .long("input-format")
             .short('i')
             .possible_values(&["csv", "tsv"])
             .about("Input is split according to the rules of (csv|tsv). $0 contains the unescaped line. Assigning to columns does nothing."))
        .arg(Arg::new("var")
             .long("var")
             .short('v')
             .multiple(true)
             .takes_value(true)
             .about("Has the form <identifier>=<expr>"))
        .arg("-F, --field-separator=[SEPARATOR] 'Field separator for frawk program.'")
        .arg("-b, --bytecode 'Execute the program with the bytecode interpreter'")
        .arg(Arg::new("output-format")
             .long("output-format")
             .short('o')
             .possible_values(&["csv", "tsv"])
             .about("If set, records output via print are escaped according to the rules of the corresponding format"))
        .arg(Arg::new("program")
             .about("The frawk program to execute")
             .index(1))
        .arg(Arg::new("input-files")
             .about("Input files to be read by frawk program")
             .index(2)
             .multiple(true))
        .arg(Arg::new("parallel-strategy")
             .about("Attempt to execute the script in parallel. Strategy r[ecord] parallelizes within and accross files. Strategy f[ile] parallelizes between input files.")
             .short('p')
             .possible_values(&["r", "record", "f", "file"]))
        .arg(Arg::new("jobs")
                .about("Number or worker threads to launch when executing in parallel, requires '-p' flag to be set")
                .short('j')
                .requires("parallel-strategy")
                .takes_value(true));
    cfg_if::cfg_if! {
        if #[cfg(feature = "llvm_backend")] {
            app = app.arg("--dump-llvm 'print LLVM-IR for the input program'");
        }
    }
    let matches = app.get_matches();
    let ifmt = match matches.value_of("input-format") {
        Some("csv") => Some(InputFormat::CSV),
        Some("tsv") => Some(InputFormat::TSV),
        Some(x) => fail!("invalid input format: {}", x),
        None => None,
    };
    let exec_strategy = match matches.value_of("parallel-strategy") {
        Some("r") | Some("record") => ExecutionStrategy::ShardPerRecord,
        Some("f") | Some("file") => ExecutionStrategy::ShardPerFile,
        None => ExecutionStrategy::Serial,
        Some(x) => fail!(
            "invalid execution strategy (clap arg parsing should handle this): {}",
            x
        ),
    };
    let num_workers = match matches.value_of("jobs") {
        Some(s) => match s.parse::<usize>() {
            Ok(u) => u,
            Err(e) => fail!("value of 'jobs' flag must be numeric: {}", e),
        },
        None => exec_strategy.num_workers(),
    };
    let mut input_files: Vec<String> = matches
        .values_of("input-files")
        .map(|x| x.map(String::from).collect())
        .unwrap_or_else(Vec::new);
    let program_string = {
        if let Some(pfile) = matches.value_of("program-file") {
            match std::fs::read_to_string(pfile) {
                Ok(p) => {
                    // We specified a file on the command line, so the "program" will be
                    // interpreted as another input file.
                    if let Some(p) = matches.value_of("program") {
                        input_files.push(p.into());
                    }
                    p
                }
                Err(e) => fail!("failed to read program from {}: {}", pfile, e),
            }
        } else if let Some(p) = matches.value_of("program") {
            String::from(p)
        } else {
            fail!("must specify program at command line, or in a file via -f");
        }
    };
    let (escaper, output_sep, output_record_sep) = match matches.value_of("output-format") {
        Some("csv") => (Escaper::CSV, Some(","), Some("\r\n")),
        Some("tsv") => (Escaper::TSV, Some("\t"), Some("\n")),
        Some(s) => fail!(
            "invalid output format {:?}; expected csv or tsv (or the empty string)",
            s
        ),
        None => (Escaper::Identity, None, None),
    };
    let raw = RawPrelude {
        field_sep: matches.value_of("field-separator").map(String::from),
        var_decs: matches
            .values_of("var")
            .map(|x| x.map(String::from).collect())
            .unwrap_or_else(Vec::new),
        output_sep,
        escaper,
        output_record_sep,
        stage: exec_strategy.stage(),
    };
    let mut opt_level: i32 = match matches.value_of("opt-level") {
        Some("3") => 3,
        Some("2") => 2,
        Some("1") => 1,
        Some("0") => 0,
        Some("-1") => -1,
        None => DEFAULT_OPT_LEVEL,
        Some(x) =>panic!("this case should be covered by clap argument validation: found unexpected opt-level value {}", x),
    };
    let opt_dump_bytecode = matches.is_present("dump-bytecode");
    let opt_dump_cfg = matches.is_present("dump-cfg");
    cfg_if::cfg_if! {
        if #[cfg(feature="llvm_backend")] {
            let opt_dump_llvm = matches.is_present("dump-llvm");
            if opt_dump_llvm {
                let config = llvm::Config {
                    opt_level: if opt_level < 0 { 3 } else { opt_level as usize },
                    num_workers,
                };
                let _ = write!(
                    std::io::stdout(),
                    "{}",
                    dump_llvm(program_string.as_str(), config, &raw),
                );
            }
        } else {
            let opt_dump_llvm = false;
        }
    }
    let skip_output = opt_dump_llvm || opt_dump_bytecode || opt_dump_cfg;
    if opt_dump_bytecode {
        let _ = write!(
            std::io::stdout(),
            "{}",
            dump_bytecode(program_string.as_str(), &raw),
        );
    }
    if opt_dump_cfg {
        let a = Arena::default();
        let ctx = get_context(program_string.as_str(), &a, get_prelude(&a, &raw));
        let mut stdout = std::io::stdout();
        let _ = ctx.dbg_print(&mut stdout);
    }
    if skip_output {
        return;
    }
    let check_utf8 = matches.is_present("utf8");

    // This horrid macro is here because all of the different ways of reading input are different
    // types, making functions hard to write. Still, there must be something to be done to clean
    // this up here.
    macro_rules! with_inp {
        ($analysis:expr, $inp:ident, $body:expr) => {
            if input_files.len() == 0 {
                let _reader: Box<dyn io::Read + Send> = Box::new(io::stdin());
                match (ifmt, $analysis) {
                    (Some(ifmt), _) => {
                        let $inp = CSVReader::new(
                            once((_reader, String::from("-"))),
                            ifmt,
                            CHUNK_SIZE,
                            check_utf8,
                            exec_strategy,
                        );
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
                                let $inp = ByteReader::new_whitespace(
                                    once((_reader, String::from("-"))),
                                    CHUNK_SIZE,
                                    check_utf8,
                                    exec_strategy,
                                );
                                $body
                            } else {
                                let $inp = ByteReader::new(
                                    once((io::stdin(), String::from("-"))),
                                    field_sep.as_bytes()[0],
                                    record_sep.as_bytes()[0],
                                    CHUNK_SIZE,
                                    check_utf8,
                                    exec_strategy,
                                );
                                $body
                            }
                        } else {
                            let $inp =
                                chained(RegexSplitter::new(_reader, CHUNK_SIZE, "-", check_utf8));
                            $body
                        }
                    }
                    (None, cfg::SepAssign::Unsure) => {
                        let $inp =
                            chained(RegexSplitter::new(_reader, CHUNK_SIZE, "-", check_utf8));
                        $body
                    }
                }
            } else if let Some(ifmt) = ifmt {
                let file_handles: Vec<_> = input_files
                    .iter()
                    .cloned()
                    .map(|file| (open_file_read(file.as_str()), file))
                    .collect();
                let $inp = CSVReader::new(
                    file_handles.into_iter(),
                    ifmt,
                    CHUNK_SIZE,
                    check_utf8,
                    exec_strategy,
                );
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
                            let file_handles: Vec<_> = input_files
                                .iter()
                                .cloned()
                                .map(move |file| (open_file_read(file.as_str()), file))
                                .collect();
                            if field_sep == " " && record_sep == "\n" {
                                let $inp = ByteReader::new_whitespace(
                                    file_handles.into_iter(),
                                    CHUNK_SIZE,
                                    check_utf8,
                                    exec_strategy,
                                );
                                $body
                            } else {
                                let $inp = ByteReader::new(
                                    file_handles.into_iter(),
                                    field_sep.as_bytes()[0],
                                    record_sep.as_bytes()[0],
                                    CHUNK_SIZE,
                                    check_utf8,
                                    exec_strategy,
                                );
                                $body
                            }
                        } else {
                            let iter = input_files.iter().cloned().map(|file| {
                                let reader: Box<dyn io::Read + Send> =
                                    Box::new(open_file_read(file.as_str()));
                                RegexSplitter::new(reader, CHUNK_SIZE, file, check_utf8)
                            });
                            let $inp = ChainedReader::new(iter);
                            $body
                        }
                    }
                    cfg::SepAssign::Unsure => {
                        let iter = input_files.iter().cloned().map(|file| {
                            let reader: Box<dyn io::Read + Send> =
                                Box::new(open_file_read(file.as_str()));
                            RegexSplitter::new(reader, CHUNK_SIZE, file, check_utf8)
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
    let out_file = matches.value_of("out-file");
    macro_rules! with_io {
        (|$inp:ident, $out:ident| $body:expr) => {
            match out_file {
                Some(oup) => {
                    let $out = runtime::writers::factory_from_file(oup)
                        .unwrap_or_else(|e| fail!("failed to open {}: {}", oup, e));
                    with_inp!(analysis_result, $inp, $body);
                }
                None => {
                    let $out = runtime::writers::default_factory();
                    with_inp!(analysis_result, $inp, $body);
                }
            }
        };
    }

    if matches.is_present("bytecode") {
        opt_level = -1;
    }

    if opt_level < 0 {
        with_io!(|inp, oup| run_interp_with_context(ctx, inp, oup, num_workers))
    } else {
        cfg_if::cfg_if! {
            if #[cfg(feature = "llvm_backend")] {
                with_io!(|inp, oup| run_llvm_with_context(
                        ctx,
                        inp,
                        oup,
                        llvm::Config {
                            opt_level: opt_level as usize,
                            num_workers,
                        },
                ));
            } else {
                fail!("opt level is {} but compiled without LLVM support", opt_level);
            }
        }
    }
}
