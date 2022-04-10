#![recursion_limit = "512"]
#![cfg_attr(feature = "unstable", feature(core_intrinsics))]
#![cfg_attr(feature = "unstable", feature(test))]
#![cfg_attr(feature = "unstable", feature(write_all_vectored))]
#[macro_use]
pub mod common;

pub mod arena;
pub mod ast;
pub mod builtins;
pub mod bytecode;
pub mod cfg;
#[macro_use]
pub mod codegen;
pub mod compile;
pub mod cross_stage;
pub mod dataflow;
mod display;
pub mod dom;
#[cfg(test)]
pub mod harness;
mod input_taint;
pub mod interp;
pub mod lexer;
#[allow(unused_parens)] // Warnings appear in generated code
#[allow(clippy::all)]
pub mod parsing;
pub mod pushdown;
pub mod runtime;
mod string_constants;
#[cfg(test)]
mod test_string_constants;
pub mod types;

use clap::{Arg, Command};

use arena::Arena;
use cfg::Escaper;
use codegen::intrinsics::IntoRuntime;
use common::{CancelSignal, ExecutionStrategy, Stage};
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
use std::mem;

#[cfg(feature = "use_jemalloc")]
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

macro_rules! fail {
    ($($t:tt)*) => {{
        eprintln_ignore!($($t)*);
        std::process::exit(1)
    }}
}

#[derive(Clone)]
struct PreludeScalars {
    arbitrary_shell: bool,
    fold_regexes: bool,
    parse_header: bool,
    escaper: Escaper,
    stage: Stage<()>,
}

struct RawPrelude {
    argv: Vec<String>,
    var_decs: Vec<String>,
    field_sep: Option<String>,
    output_sep: Option<&'static str>,
    output_record_sep: Option<&'static str>,
    scalars: PreludeScalars,
}

struct Prelude<'a> {
    var_decs: Vec<(&'a str, &'a ast::Expr<'a, 'a, &'a str>)>,
    field_sep: Option<&'a [u8]>,
    output_sep: Option<&'a [u8]>,
    output_record_sep: Option<&'a [u8]>,
    argv: Vec<&'a str>,
    scalars: PreludeScalars,
}

// TODO: make file reading lazy
fn open_file_read(f: &str) -> impl io::BufRead {
    enum LazyReader<F, R> {
        Uninit(F),
        Init(R),
    }

    impl<R, F: FnMut() -> io::Result<R>> LazyReader<F, R> {
        fn delegate<T>(&mut self, next: impl FnOnce(&mut R) -> io::Result<T>) -> io::Result<T> {
            match self {
                LazyReader::Uninit(f) => {
                    *self = LazyReader::Init(f()?);
                    self.delegate(next)
                }
                LazyReader::Init(r) => next(r),
            }
        }
    }

    // TODO: delegate other methods on read.
    impl<R: io::Read, F: FnMut() -> io::Result<R>> io::Read for LazyReader<F, R> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.delegate(|r| r.read(buf))
        }
    }

    let filename = String::from(f);
    BufReader::new(LazyReader::Uninit(move || File::open(filename.as_str())))
}

fn chained<LR: LineReader>(lr: LR) -> ChainedReader<LR> {
    ChainedReader::new(std::iter::once(lr))
}

fn get_vars<'a, 'b>(
    vars: impl Iterator<Item = &'b str>,
    a: &'a Arena,
    buf: &mut Vec<u8>,
) -> Vec<(&'a str, &'a ast::Expr<'a, 'a, &'a str>)> {
    let mut res = Vec::new();
    let mut split_buf = Vec::new();
    for var in vars {
        buf.clear();
        split_buf.clear();
        split_buf.extend(var.splitn(2, '='));
        if split_buf.len() != 2 {
            fail!(
                "received -v flag without an '=' sign: {} (split_buf={:?})",
                var,
                split_buf
            );
        }
        let ident = a.alloc_str(split_buf[0].trim());
        if !lexer::is_ident(ident) {
            fail!(
                "invalid identifier for left-hand side of -v flag: {}",
                ident
            );
        }
        let str_lit = lexer::parse_string_literal(split_buf[1], a, buf);
        res.push((ident, a.alloc(ast::Expr::StrLit(str_lit))))
    }
    res
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
        scalars: raw.scalars.clone(),
        output_sep,
        output_record_sep,
        argv: raw.argv.iter().map(|s| a.alloc_str(s.as_str())).collect(),
    }
}

fn get_context<'a>(
    prog: &str,
    a: &'a Arena,
    mut prelude: Prelude<'a>,
) -> cfg::ProgramContext<'a, &'a str> {
    let prog = a.alloc_str(prog);
    let lexer = lexer::Tokenizer::new(prog);
    let mut buf = Vec::new();
    let parser = parsing::syntax::ProgParser::new();
    let mut prog = ast::Prog::from_stage(a, prelude.scalars.stage.clone());
    prog.argv = mem::take(&mut prelude.argv);
    let stmt = match parser.parse(a, &mut buf, &mut prog, lexer) {
        Ok(()) => {
            prog.field_sep = prelude.field_sep;
            prog.prelude_vardecs = prelude.var_decs;
            prog.output_sep = prelude.output_sep;
            prog.output_record_sep = prelude.output_record_sep;
            prog.parse_header = prelude.scalars.parse_header;
            a.alloc(prog)
        }
        Err(e) => {
            fail!("{}", e);
        }
    };
    match cfg::ProgramContext::from_prog(a, stmt, prelude.scalars.escaper) {
        Ok(mut ctx) => {
            ctx.allow_arbitrary_commands = prelude.scalars.arbitrary_shell;
            ctx.fold_regex_constants = prelude.scalars.fold_regexes;
            ctx
        }
        Err(e) => fail!("failed to create program context: {}", e),
    }
}

fn run_interp_with_context<'a>(
    mut ctx: cfg::ProgramContext<'a, &'a str>,
    stdin: impl LineReader,
    ff: impl runtime::writers::FileFactory,
    num_workers: usize,
) {
    let rc = {
        let mut interp = match compile::bytecode(&mut ctx, stdin, ff, num_workers) {
            Ok(ctx) => ctx,
            Err(e) => fail!("bytecode compilation failure: {}", e),
        };
        match interp.run() {
            Err(e) => fail!("fatal error during execution: {}", e),
            Ok(0) => return,
            Ok(n) => n,
        }
    };
    std::process::exit(rc);
}

fn run_cranelift_with_context<'a>(
    mut ctx: cfg::ProgramContext<'a, &'a str>,
    stdin: impl IntoRuntime,
    ff: impl runtime::writers::FileFactory,
    cfg: codegen::Config,
    signal: CancelSignal,
) {
    if let Err(e) = compile::run_cranelift(&mut ctx, stdin, ff, cfg, signal) {
        fail!("error compiling cranelift: {}", e)
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "llvm_backend")] {
        fn run_llvm_with_context<'a>(
            mut ctx: cfg::ProgramContext<'a, &'a str>,
            stdin: impl IntoRuntime,
            ff: impl runtime::writers::FileFactory,
            cfg: codegen::Config,
            signal: CancelSignal,
        ) {
            if let Err(e) = compile::run_llvm(&mut ctx, stdin, ff, cfg, signal) {
                fail!("error compiling llvm: {}", e)
            }
        }

        fn dump_llvm(prog: &str, cfg: codegen::Config, raw: &RawPrelude) -> String {
            let a = Arena::default();
            let mut ctx = get_context(prog, &a, get_prelude(&a, raw));
            match compile::dump_llvm(&mut ctx, cfg) {
                Ok(s) => s,
                Err(e) => fail!("error compiling llvm: {}", e),
            }
        }

    }
}

const DEFAULT_OPT_LEVEL: i32 = 3;

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
            Default::default(),
        )),
        runtime::writers::default_factory(),
        /*num_workers=*/ 1,
    ) {
        Ok(ctx) => ctx,
        Err(e) => fail!("bytecode compilation failure: {}", e),
    };
    let mut v = Vec::<u8>::new();
    for (i, func) in interp.instrs().iter().enumerate() {
        writeln!(&mut v, "function {} {{", i).unwrap();
        for (j, inst) in func.iter().enumerate() {
            writeln!(&mut v, "\t[{:2}] {:?}", j, inst).unwrap();
        }
        writeln!(&mut v, "}}\n").unwrap();
    }
    String::from_utf8(v).unwrap()
}

fn main() {
    #[allow(unused_mut)]
    let mut app = Command::new("frawk")
        .version("0.4.5")
        .author("Eli R.")
        .about("frawk is a pattern scanning and (semi-structured) text processing language")
        .arg(Arg::new("program-file")
             .long("program-file")
             .short('f')
             .takes_value(true)
             .multiple_occurrences(true)
             .help("Read the program source from the file program-file, instead of from the command line. Multiple '-f' options may be used"))
        .arg(Arg::new("opt-level")
             .long("opt-level")
             .short('O')
             .takes_value(true)
             .allow_hyphen_values(true)
             .help("The optimization level for the program. Positive levels determine the optimization level for LLVM. Level `-1` forces bytecode interpretation")
             .possible_values(&["-1", "0", "1", "2", "3"]))
        .arg(Arg::new("out-file")
             .long("out-file")
             .takes_value(true)
             .value_name("FILE")
             .help("Write to specified output file instead of standard output"))
        .arg(Arg::new("utf8")
             .long("utf8")
             .takes_value(false)
             .help("Validate all input as UTF-8, returning an error if it is invalid"))
        .arg(Arg::new("dump-cfg")
             .long("dump-cfg")
             .takes_value(false)
             .help("Print untyped SSA form for input program"))
        .arg(Arg::new("dump-bytecode")
             .long("dump-bytecode")
             .takes_value(false)
             .help("Print bytecode for input program"))
        .arg(Arg::new("parse-header")
             .long("parse-header")
             .short('H')
             .takes_value(false)
             .help("Consume the first line of input and populate the `FI` variable with column names mapping to column indexes"))
        .arg(Arg::new("input-format")
             .long("input-format")
             .short('i')
             .value_name("csv|tsv")
             .conflicts_with("field-separator")
             .help("Input is split according to the rules of (csv|tsv). $0 contains the unescaped line. Assigning to columns does nothing")
             .possible_values(&["csv", "tsv"]))
        .arg(Arg::new("var")
             .short('v')
             .takes_value(true)
             .multiple_occurrences(true)
             .value_name("var=val")
             .help("Assign the value <val> to the variable <var>, before execution of the frawk program begins. Multiple '-v' options may be used"))
        .arg(Arg::new("field-separator")
             .long("field-separator")
             .short('F')
             .takes_value(true)
             .value_name("FS")
             .conflicts_with("input-format")
             .help("Field separator `FS` for frawk program"))
        .arg(Arg::new("backend")
             .long("backend")
             .short('B')
             .help("The backend used to run the frawk program, ranging from fastest to compile and slowest to execute, and slowest to compile and fastest to execute. Cranelift is the default")
             .possible_values(&["interp", "cranelift", "llvm"]))
        .arg(Arg::new("output-format")
             .long("output-format")
             .short('o')
             .value_name("csv|tsv")
             .help("If set, records output via print are escaped according to the rules of the corresponding format")
             .possible_values(&["csv", "tsv"]))
        .arg(Arg::new("program")
             .index(1)
             .help("The frawk program to execute"))
        .arg(Arg::new("input-files")
             .index(2)
             .multiple_values(true)
             .help("Input files to be read by frawk program"))
        .arg(Arg::new("parallel-strategy")
             .short('p')
             .help("Attempt to execute the script in parallel. Strategy r[ecord] parallelizes within the current input file. Strategy f[ile] parallelizes between input files")
             .possible_values(&["r", "record", "f", "file"]))
        .arg(Arg::new("chunk-size")
             .long("chunk-size")
             .takes_value(true)
             .help("Buffer size when reading input. This is present primarily for debugging purposes; it's possible that tuning this will help performance, but it should not be necessary"))
        .arg(Arg::new("arbitrary-shell")
             .short('A')
             .long("arbitrary-shell")
             .takes_value(false)
             .help("By default, strings that are passed to the shell via pipes or the 'system' function are restricted from potentially containing user input. This flag bypasses that check, for the cases where such a use is known to be safe"))
        .arg(Arg::new("jobs")
             .short('j')
             .requires("parallel-strategy")
             .takes_value(true)
             .help("Number or worker threads to launch when executing in parallel, requires '-p' flag to be set. When using record-level parallelism, this value is an upper bound on the number of worker threads that will be spawned; the number of active worker threads is chosen dynamically"));
    cfg_if::cfg_if! {
        if #[cfg(feature = "llvm_backend")] {
            app = app.arg(Arg::new("dump-llvm")
             .long("dump-llvm")
             .takes_value(false)
             .help("Print LLVM-IR for the input program"));
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

    // NB: do we want this to be a command-line param?
    let chunk_size = if let Some(cs) = matches.value_of("chunk-size") {
        match cs.parse::<usize>() {
            Ok(u) => u,
            Err(e) => fail!("value of 'chunk-size' flag must be numeric: {}", e),
        }
    } else {
        CHUNK_SIZE
    };
    let num_workers = match matches.value_of("jobs") {
        Some(s) => match s.parse::<usize>() {
            Ok(u) => u,
            Err(e) => fail!("value of 'jobs' flag must be numeric: {}", e),
        },
        None => exec_strategy.num_workers(),
    };
    let argv: Vec<String> = std::env::args()
        .next()
        .into_iter()
        .chain(
            matches
                .values_of("input-files")
                .into_iter()
                .flat_map(|x| x.map(String::from)),
        )
        .collect();
    let mut input_files: Vec<String> = matches
        .values_of("input-files")
        .map(|x| x.map(String::from).collect())
        .unwrap_or_else(Vec::new);
    let program_string = {
        if let Some(pfiles) = matches.values_of("program-file") {
            // We specified a file on the command line, so the "program" will be
            // interpreted as another input file.
            if let Some(p) = matches.value_of("program") {
                input_files.insert(0, p.into());
            }
            let mut prog = String::new();
            for pfile in pfiles {
                match std::fs::read_to_string(pfile) {
                    Ok(p) => {
                        prog.push_str(p.as_str());
                        prog.push('\n');
                    }
                    Err(e) => fail!("failed to read program from {}: {}", pfile, e),
                }
            }
            prog
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
    let arbitrary_shell = matches.is_present("arbitrary-shell");
    let parse_header = matches.is_present("parse-header");

    let opt_level: i32 = match matches.value_of("opt-level") {
        Some("3") => 3,
        Some("2") => 2,
        Some("1") => 1,
        Some("0") => 0,
        Some("-1") => -1,
        None => DEFAULT_OPT_LEVEL,
        Some(x) => panic!("this case should be covered by clap argument validation: found unexpected opt-level value {}", x),
    };
    let raw = RawPrelude {
        field_sep: matches.value_of("field-separator").map(String::from),
        var_decs: matches
            .values_of("var")
            .map(|x| x.map(String::from).collect())
            .unwrap_or_else(Vec::new),
        output_sep,
        scalars: PreludeScalars {
            escaper,
            arbitrary_shell,
            fold_regexes: opt_level >= 3,
            stage: exec_strategy.stage(),
            parse_header,
        },
        output_record_sep,
        argv,
    };
    let opt_dump_bytecode = matches.is_present("dump-bytecode");
    let opt_dump_cfg = matches.is_present("dump-cfg");
    cfg_if::cfg_if! {
        if #[cfg(feature="llvm_backend")] {
            let opt_dump_llvm = matches.is_present("dump-llvm");
            if opt_dump_llvm {
                let config = codegen::Config {
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
    let signal = CancelSignal::default();

    // This horrid macro is here because all of the different ways of reading input are different
    // types, making functions hard to write. Still, there must be something to be done to clean
    // this up here.
    macro_rules! with_inp {
        ($analysis:expr, $inp:ident, $body:expr) => {{
            if input_files.len() == 0 {
                let _reader: Box<dyn io::Read + Send> = Box::new(io::stdin());
                match (ifmt, $analysis) {
                    (Some(ifmt), _) => {
                        let $inp = CSVReader::new(
                            once((_reader, String::from("-"))),
                            ifmt,
                            chunk_size,
                            check_utf8,
                            exec_strategy,
                            signal.clone(),
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
                        let field_sep = field_sep.unwrap_or(b" ");
                        let record_sep = record_sep.unwrap_or(b"\n");
                        if field_sep.len() == 1 && record_sep.len() == 1 {
                            if field_sep == b" " && record_sep == b"\n" {
                                let $inp = ByteReader::new_whitespace(
                                    once((_reader, String::from("-"))),
                                    chunk_size,
                                    check_utf8,
                                    exec_strategy,
                                    signal.clone(),
                                );
                                $body
                            } else {
                                let $inp = ByteReader::new(
                                    once((io::stdin(), String::from("-"))),
                                    field_sep[0],
                                    record_sep[0],
                                    chunk_size,
                                    check_utf8,
                                    exec_strategy,
                                    signal.clone(),
                                );
                                $body
                            }
                        } else {
                            let $inp =
                                chained(RegexSplitter::new(_reader, chunk_size, "-", check_utf8));
                            $body
                        }
                    }
                    (None, cfg::SepAssign::Unsure) => {
                        let $inp =
                            chained(RegexSplitter::new(_reader, chunk_size, "-", check_utf8));
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
                    chunk_size,
                    check_utf8,
                    exec_strategy,
                    signal.clone(),
                );
                $body
            } else {
                match $analysis {
                    cfg::SepAssign::Potential {
                        field_sep,
                        record_sep,
                    } => {
                        let field_sep = field_sep.unwrap_or(b" ");
                        let record_sep = record_sep.unwrap_or(b"\n");
                        if field_sep.len() == 1 && record_sep.len() == 1 {
                            let file_handles: Vec<_> = input_files
                                .iter()
                                .cloned()
                                .map(move |file| (open_file_read(file.as_str()), file))
                                .collect();
                            if field_sep == b" " && record_sep == b"\n" {
                                let $inp = ByteReader::new_whitespace(
                                    file_handles.into_iter(),
                                    chunk_size,
                                    check_utf8,
                                    exec_strategy,
                                    signal.clone(),
                                );
                                $body
                            } else {
                                let $inp = ByteReader::new(
                                    file_handles.into_iter(),
                                    field_sep[0],
                                    record_sep[0],
                                    chunk_size,
                                    check_utf8,
                                    exec_strategy,
                                    signal.clone(),
                                );
                                $body
                            }
                        } else {
                            let iter = input_files.iter().cloned().map(|file| {
                                let reader: Box<dyn io::Read + Send> =
                                    Box::new(open_file_read(file.as_str()));
                                RegexSplitter::new(reader, chunk_size, file, check_utf8)
                            });
                            let $inp = ChainedReader::new(iter);
                            $body
                        }
                    }
                    cfg::SepAssign::Unsure => {
                        let iter = input_files.iter().cloned().map(|file| {
                            let reader: Box<dyn io::Read + Send> =
                                Box::new(open_file_read(file.as_str()));
                            RegexSplitter::new(reader, chunk_size, file, check_utf8)
                        });
                        let $inp = ChainedReader::new(iter);
                        $body
                    }
                }
            }
        }};
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
    match matches.value_of("backend") {
        Some("llvm") => {
            cfg_if::cfg_if! {
                if #[cfg(feature = "llvm_backend")] {
                    with_io!(|inp, oup| run_llvm_with_context(
                            ctx,
                            inp,
                            oup,
                            codegen::Config {
                                opt_level: opt_level as usize,
                                num_workers,
                            },
                            signal,
                    ));
                } else {
                    fail!("backend specified as LLVM, but compiled without LLVM support");
                }
            }
        }
        Some("interp") => {
            with_io!(|inp, oup| run_interp_with_context(ctx, inp, oup, num_workers))
        }
        None | Some("cranelift") => {
            with_io!(|inp, oup| run_cranelift_with_context(
                ctx,
                inp,
                oup,
                codegen::Config {
                    opt_level: opt_level as usize,
                    num_workers,
                },
                signal,
            ));
        }
        Some(b) => {
            fail!("invalid backend: {:?}", b);
        }
    }
}
