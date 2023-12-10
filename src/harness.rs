//! Some utility functions for running AWK programs from Rust code. It is primarily used to test
//! "everything but main" end to end; there's a test suite at the end.
#[cfg(feature = "unstable")]
use crate::bytecode::Interp;
#[cfg(feature = "llvm_backend")]
use crate::codegen::llvm;
use crate::{
    arena::Arena,
    ast,
    cfg::{self, Escaper},
    codegen::{self, intrinsics::IntoRuntime},
    common::{ExecutionStrategy, Result},
    compile, lexer,
    parsing::syntax,
    pushdown::FieldSet,
    runtime::{
        self,
        splitter::{
            batch::{ByteReader, CSVReader, InputFormat},
            regex::RegexSplitter,
        },
        writers::testing::FakeFs,
        ChainedReader,
    },
    types::{self, get_types},
};

use cfg_if::cfg_if;
use hashbrown::HashMap;
use std::io::{self, Write};
use std::iter::FromIterator;

const FILE_BREAK: &str = "<<<FILE BREAK>>>";

fn split_stdin(
    stdin: String,
) -> impl Iterator<Item = (Box<dyn io::Read + Send>, String)> + 'static {
    Vec::from_iter(
        stdin
            .split(FILE_BREAK)
            .map(String::from)
            .enumerate()
            .map(|(i, x)| {
                let reader: Box<dyn io::Read + Send> = Box::new(std::io::Cursor::new(x));
                (reader, format!("fake_stdin_{}", i))
            }),
    )
    .into_iter()
}
macro_rules! with_reader {
    ($report:expr, $inp:expr, |$id:ident| $body:expr) => {
        match $report {
            cfg::SepAssign::Unsure => {
                let $id = simulate_stdin_regex($inp);
                $body
            }
            cfg::SepAssign::Potential {
                field_sep,
                record_sep,
            } => {
                // TODO: unify this code with the code in main.
                let field_sep = field_sep.unwrap_or(b" ");
                let record_sep = record_sep.unwrap_or(b"\n");
                if field_sep.len() == 1 && record_sep.len() == 1 {
                    if field_sep == b" " && record_sep == b"\n" {
                        let $id = simulate_stdin_whitespace($inp);
                        $body
                    } else {
                        let $id = simulate_stdin_singlechar(field_sep[0], record_sep[0], $inp);
                        $body
                    }
                } else {
                    let $id = simulate_stdin_regex($inp);
                    $body
                }
            }
        }
    };
}

fn simulate_stdin_singlechar(
    field_sep: u8,
    record_sep: u8,
    inp: impl Into<String>,
) -> impl IntoRuntime + runtime::LineReader {
    ByteReader::new(
        split_stdin(inp.into()),
        field_sep,
        record_sep,
        runtime::CHUNK_SIZE,
        /*check_utf8=*/ true,
        ExecutionStrategy::Serial,
        Default::default(),
    )
}

fn simulate_stdin_whitespace(inp: impl Into<String>) -> impl IntoRuntime + runtime::LineReader {
    ByteReader::new_whitespace(
        split_stdin(inp.into()),
        runtime::CHUNK_SIZE,
        /*check_utf8=*/ true,
        ExecutionStrategy::Serial,
        Default::default(),
    )
}

fn simulate_stdin<LR: runtime::LineReader>(
    inp: impl Into<String>,
    mut f: impl FnMut(Box<dyn io::Read + Send>, String) -> LR,
) -> ChainedReader<LR>
where
    ChainedReader<LR>: IntoRuntime,
{
    ChainedReader::new(split_stdin(inp.into()).map(|(r, name)| f(r, name)))
}

fn simulate_stdin_csv(
    ifmt: InputFormat,
    inp: impl Into<String>,
    strat: ExecutionStrategy,
) -> impl IntoRuntime + runtime::LineReader {
    CSVReader::new(
        split_stdin(inp.into()),
        ifmt,
        runtime::CHUNK_SIZE,
        /*check_utf8=*/ true,
        strat,
        Default::default(),
    )
}

fn simulate_stdin_regex(inp: impl Into<String>) -> impl IntoRuntime + runtime::LineReader {
    simulate_stdin(inp, |reader, name| {
        RegexSplitter::new(
            reader,
            runtime::CHUNK_SIZE,
            name,
            /*check_utf8=*/ false,
        )
    })
}

const _PRINT_DEBUG_INFO: bool = false;

type Prog<'a> = &'a ast::Prog<'a, 'a, &'a str>;

type ProgResult<'a> = Result<(
    String,                        /* output */
    String,                        /* debug info */
    HashMap<&'a str, compile::Ty>, /* type info */
)>;

const CODEGEN_CONFIG: codegen::Config = codegen::Config {
    opt_level: 0,
    num_workers: 1,
};

pub(crate) fn run_program<'a>(
    a: &'a Arena,
    prog: &str,
    stdin: impl Into<String>,
    esc: Escaper,
    ifmt: Option<InputFormat>,
    strat: ExecutionStrategy,
) -> ProgResult<'a> {
    let stmt = parse_program(prog, a, esc, strat)?;
    run_prog(a, stmt, stdin, esc, ifmt, strat)
}

cfg_if! {
    if #[cfg(all(feature="llvm_backend", feature="unstable"))] {
        pub(crate) fn compile_llvm(prog: &str, esc: Escaper) -> Result<()> {
            let a = Arena::default();
            let stmt = parse_program(prog, &a, esc, ExecutionStrategy::Serial)?;
            let mut ctx = cfg::ProgramContext::from_prog(&a, stmt, esc)?;
            compile::compile_llvm(&mut ctx, CODEGEN_CONFIG)
        }
    }
}

cfg_if! {
    if #[cfg(feature="llvm_backend")] {
        pub(crate) fn dump_llvm(prog: &str, esc: Escaper) -> Result<String> {
            let a = Arena::default();
            let stmt = parse_program(prog, &a, esc, ExecutionStrategy::Serial)?;
            let mut ctx = cfg::ProgramContext::from_prog(&a, stmt, esc)?;
            compile::dump_llvm(&mut ctx, CODEGEN_CONFIG)
        }

        // The run_llvm path implements a subset of the logic in main that specializes the input
        // readers for whitespace, or single-byte-separator splitter implementations. The bytecode
        // path does not, because we want to ensure that the logic works when using the more
        // general regex-based splitter.
        pub(crate) fn run_llvm(
            prog: &str,
            stdin: impl Into<String>,
            esc: Escaper,
            ifmt: Option<InputFormat>,
            strat: ExecutionStrategy,
        ) -> Result<String> {
            let a = Arena::default();
            let stmt = parse_program(prog, &a, esc, strat)?;
            let mut ctx = cfg::ProgramContext::from_prog(&a, stmt, esc)?;
            ctx.fold_regex_constants = true;
            let sep_analysis = ctx.analyze_sep_assignments();
            if _PRINT_DEBUG_INFO {
                let mut buf = Vec::<u8>::new();
                ctx.dbg_print(&mut buf).unwrap();
                eprintln!("{}", String::from_utf8(buf).unwrap());
            }
            let fake_fs = FakeFs::default();
            if let Some(ifmt) = ifmt {
                compile::run_llvm(
                    &mut ctx,
                    simulate_stdin_csv(ifmt, stdin, strat),
                    fake_fs.clone(),
                    llvm::Config {
                        opt_level: CODEGEN_CONFIG.opt_level,
                        num_workers: strat.num_workers(),
                    },
                    Default::default(),
                )?;
            } else {
                with_reader!(sep_analysis, stdin, |reader| {
                    compile::run_llvm(&mut ctx, reader, fake_fs.clone(), CODEGEN_CONFIG, Default::default())?;
                });
            }
            let v = fake_fs.stdout.read_data();
            match String::from_utf8(v) {
                Ok(s) => Ok(s),
                Err(e) => err!("program produced invalid unicode: {}", e),
            }
        }
    }
}

pub(crate) fn run_cranelift(
    prog: &str,
    stdin: impl Into<String>,
    esc: Escaper,
    ifmt: Option<InputFormat>,
    strat: ExecutionStrategy,
) -> Result<String> {
    let a = Arena::default();
    let stmt = parse_program(prog, &a, esc, strat)?;
    let mut ctx = cfg::ProgramContext::from_prog(&a, stmt, esc)?;
    ctx.fold_regex_constants = true;
    let sep_analysis = ctx.analyze_sep_assignments();
    if _PRINT_DEBUG_INFO {
        let mut buf = Vec::<u8>::new();
        ctx.dbg_print(&mut buf).unwrap();
        eprintln!("{}", String::from_utf8(buf).unwrap());
    }
    let fake_fs = FakeFs::default();
    if let Some(ifmt) = ifmt {
        compile::run_cranelift(
            &mut ctx,
            simulate_stdin_csv(ifmt, stdin, strat),
            fake_fs.clone(),
            codegen::Config {
                opt_level: CODEGEN_CONFIG.opt_level,
                num_workers: strat.num_workers(),
            },
            Default::default(),
        )?;
    } else {
        with_reader!(sep_analysis, stdin, |reader| {
            compile::run_cranelift(
                &mut ctx,
                reader,
                fake_fs.clone(),
                CODEGEN_CONFIG,
                Default::default(),
            )?;
        });
    }
    let v = fake_fs.stdout.read_data();
    match String::from_utf8(v) {
        Ok(s) => Ok(s),
        Err(e) => err!("program produced invalid unicode: {}", e),
    }
}

#[cfg(feature = "unstable")]
pub(crate) fn bench_program(
    prog: &str,
    stdin: impl Into<String>,
    esc: Escaper,
    strat: ExecutionStrategy,
) -> Result<String> {
    let a = Arena::default();
    let stmt = parse_program(prog, &a, esc, strat)?;
    let (mut interp, stdout) = compile_program(&a, stmt, stdin, esc, strat)?;
    run_prog_nodebug(&mut interp, stdout)
}

pub(crate) fn program_compiles(prog: &str, allow_arbitrary: bool) -> Result<()> {
    let a = Arena::default();
    let esc = Escaper::Identity;
    let stmt = parse_program(prog, &a, esc, ExecutionStrategy::Serial)?;
    let mut ctx = cfg::ProgramContext::from_prog(&a, stmt, esc)?;
    ctx.allow_arbitrary_commands = allow_arbitrary;
    compile::context_compiles(&mut ctx)
}

pub(crate) fn used_fields(prog: &str) -> Result<FieldSet> {
    let a = Arena::default();
    let esc = Escaper::Identity;
    let stmt = parse_program(prog, &a, esc, ExecutionStrategy::Serial)?;
    let mut ctx = cfg::ProgramContext::from_prog(&a, stmt, esc)?;
    compile::used_fields(&mut ctx)
}

pub(crate) fn parse_program<'a>(
    prog: &str,
    a: &'a Arena,
    esc: Escaper,
    strat: ExecutionStrategy,
) -> Result<Prog<'a>> {
    let prog = a.alloc_str(prog);
    let lexer = lexer::Tokenizer::new(prog);
    let mut buf = Vec::new();
    let mut program = ast::Prog::from_stage(a, strat.stage());
    let parser = syntax::ProgParser::new();
    match parser.parse(a, &mut buf, &mut program, lexer) {
        Ok(()) => {
            match esc {
                Escaper::CSV => program.output_sep = Some(b","),
                Escaper::TSV => program.output_sep = Some(b"\t"),
                Escaper::Identity => {}
            };
            Ok(a.alloc(program))
        }
        Err(e) => {
            let mut ix = 0;
            let mut msg: String = "failed to parse program:\n======\n".into();
            for line in prog.lines() {
                msg.push_str(format!("[{:3}] {}\n", ix, line).as_str());
                ix += line.len() + 1;
            }
            err!("{}=====\nError: {:?}", msg, e)
        }
    }
}

#[cfg(feature = "unstable")]
fn compile_program<'a>(
    a: &'a Arena,
    prog: Prog<'a>,
    stdin: impl Into<String>,
    esc: Escaper,
    strat: ExecutionStrategy,
) -> Result<(Interp<'a, impl runtime::LineReader>, FakeFs)> {
    let mut ctx = cfg::ProgramContext::from_prog(a, prog, esc)?;
    let fake_fs = FakeFs::default();
    Ok((
        compile::bytecode(
            &mut ctx,
            simulate_stdin_regex(stdin),
            fake_fs.clone(),
            strat.num_workers(),
        )?,
        fake_fs,
    ))
}

#[cfg(feature = "unstable")]
fn run_prog_nodebug<LR: runtime::LineReader>(
    interp: &mut Interp<LR>,
    fake_fs: FakeFs,
) -> Result<String /*output*/> {
    interp.run()?;
    let v = fake_fs.stdout.read_data();
    match String::from_utf8(v) {
        Ok(s) => Ok(s),
        Err(e) => err!("program produced invalid unicode: {}", e),
    }
}

pub(crate) fn run_prog<'a>(
    arena: &'a Arena,
    prog: Prog<'a>,
    stdin: impl Into<String>,
    esc: Escaper,
    ifmt: Option<InputFormat>,
    strat: ExecutionStrategy,
) -> ProgResult<'a> {
    let mut ctx = cfg::ProgramContext::from_prog(arena, prog, esc)?;
    // NB the invert_ident machinery only works for global identifiers. We could get it to work in
    // a limited capacity for locals, but it would require a lot more bookkeeping.
    let ident_map = ctx._invert_ident();
    let fake_fs = FakeFs::default();
    let (instrs, type_map) = {
        let mut instrs_buf = Vec::<u8>::new();
        writeln!(&mut instrs_buf, "\nCFG:").unwrap();
        ctx.dbg_print(&mut instrs_buf).unwrap();
        writeln!(&mut instrs_buf).unwrap();
        let types::TypeInfo { var_tys, func_tys } = get_types(&ctx)?;
        // ident_map : Ident -> &str (but only has globals)
        // ts: Ident -> Type
        //
        // We want the types of all the entries in ts that show up in ident_map.
        let type_map: HashMap<&'a str, compile::Ty> = var_tys
            .iter()
            .flat_map(|((ident, _, _), ty)| ident_map.get(&ident._base()).map(|s| (*s, *ty)))
            .collect();
        macro_rules! with_interp {
            ($interp:ident, $body: expr) => {
                if let Some(ifmt) = ifmt {
                    let mut $interp = compile::bytecode(
                        &mut ctx,
                        simulate_stdin_csv(ifmt, stdin, strat),
                        fake_fs.clone(),
                        strat.num_workers(),
                    )?;
                    $body
                } else {
                    let mut $interp = compile::bytecode(
                        &mut ctx,
                        simulate_stdin_regex(stdin),
                        fake_fs.clone(),
                        strat.num_workers(),
                    )?;
                    $body
                }
            };
        }
        with_interp!(interp, {
            for (i, func) in interp.instrs().iter().enumerate() {
                writeln!(&mut instrs_buf, "function {} {{", i).unwrap();
                for (j, inst) in func.iter().enumerate() {
                    writeln!(&mut instrs_buf, "\t[{:2}] {:?}", j, inst).unwrap();
                }
                writeln!(&mut instrs_buf, "}}").unwrap();
            }
            let instrs = String::from_utf8(instrs_buf).unwrap();
            if _PRINT_DEBUG_INFO {
                eprintln!(
                    "func_tys={:?}\nvar_tys={:?}\n=========\n",
                    func_tys, var_tys
                );
                eprintln!("{}", instrs);
            }
            interp.run()?;
            (instrs, type_map)
        })
    };

    let v = fake_fs.stdout.read_data();
    match String::from_utf8(v) {
        Ok(s) => Ok((s, instrs, type_map)),
        Err(e) => err!("program produced invalid UTF-8: {}", e),
    }
}

mod tests {
    use super::*;

    macro_rules! test_program_parallel {
        ($desc:ident, $strat:tt, $e:expr, $in:expr, $out:expr) => {
            mod $desc {
                use super::*;

                #[test]
                fn bytecode() {
                    let a = Arena::default();
                    let out = run_program(
                        &a,
                        $e,
                        $in,
                        Escaper::Identity,
                        Some(InputFormat::CSV),
                        ExecutionStrategy::$strat,
                    );
                    match out {
                        Ok((out, instrs, ts)) => {
                            let expected = $out;
                            assert_eq!(out, expected, "{}\nTypes:\n{:?}", instrs, ts);
                        }
                        Err(e) => panic!("failed to run program: {}", e),
                    }
                }
                #[cfg(feature = "llvm_backend")]
                #[test]
                fn llvm() {
                    match run_llvm(
                        $e,
                        $in,
                        Escaper::Identity,
                        Some(InputFormat::CSV),
                        ExecutionStrategy::$strat,
                    ) {
                        Ok(out) => assert_eq!(
                            out,
                            $out,
                            "llvm=\n{}",
                            dump_llvm($e, Escaper::Identity).expect("failed to dump llvm")
                        ),
                        Err(e) => panic!("{}", e),
                    }
                }
            }
        };
    }

    // TODO our unprincipled way of parsing arguments for this macro is making it pretty unwieldy.
    macro_rules! test_program {
        ($desc:ident, $e:expr, $out:expr) => {
            test_program!($desc, $e, $out, @input "", @types [], @out_fmt Escaper::Identity, @csv None);
        };
        ($desc:ident, $e:expr, $out:expr, @out_fmt $esc:expr) => {
            test_program!($desc, $e, $out, @input "", @types [], @out_fmt $esc, @csv None);
        };
        ($desc:ident, $e:expr, $out:expr, @csv $csv:expr) => {
            test_program!($desc, $e, $out, @input "", @types [], @out_fmt Escaper::Identity, @csv $csv);
        };
        ($desc:ident, $e:expr, $out:expr, @input $inp:expr) => {
            test_program!($desc, $e, $out, @input $inp, @types [], @out_fmt Escaper::Identity, @csv None);
        };
        ($desc:ident, $e:expr, $out:expr, @input $inp:expr, @csv $csv:expr) => {
            test_program!($desc, $e, $out, @input $inp, @types [], @out_fmt Escaper::Identity, @csv $csv);
        };
        ($desc:ident, $e:expr, $out:expr, @input $inp:expr,
         @types [ $($i:ident :: $ty:expr),* ]) => {
            test_program!($desc, $e, $out, @input $inp, @types [$($i :: $ty),*], @out_fmt Escaper::Identity, @csv None);
        };
        ($desc:ident, $e:expr, $out:expr, @input $inp:expr,
         @types [ $($i:ident :: $ty:expr),* ], @out_fmt $esc:expr, @csv $csv:expr) => {
            mod $desc {
                use super::*;
                #[test]
                fn bytecode() {
                    let a = Arena::default();
                    let out = run_program(&a, $e, $inp, $esc, $csv, ExecutionStrategy::Serial);
                    match out {
                        Ok((out, instrs, ts)) => {
                            let expected = $out;
                            assert_eq!(out, expected, "{}\nTypes:\n{:?}", instrs, ts);
                            {
                                #[allow(unused)]
                                use crate::compile::Ty::*;
                                $(
                                    assert_eq!(
                                      ts.get(stringify!($i)).cloned(),
                                      Some($ty),
                                      "Expected identifier {} to have type {:?}. Types: {:?}\n{}\n",
                                      stringify!($i), $ty, ts, instrs,
                                    );
                                 )*
                            }
                        }
                        Err(e) => panic!("failed to run program: {}", e),
                    }
                }

                #[test]
                fn cranelift() {
                    match run_cranelift($e, $inp, $esc, $csv, ExecutionStrategy::Serial) {
                        Ok(out) => assert_eq!(out, $out),
                        Err(e) => panic!("{}", e),
                    }
                }

                #[cfg(feature = "llvm_backend")]
                #[test]
                fn llvm() {
                    match run_llvm($e, $inp, $esc, $csv, ExecutionStrategy::Serial) {
                        Ok(out) => assert_eq!(
                            out, $out,
                            "llvm=\n{}", dump_llvm($e, $esc).expect("failed to dump llvm")),
                        Err(e) => panic!("{}", e),
                    }
                }
            }
        };
    }

    macro_rules! test_program_csv {
        ($desc:ident, $e:expr, $out:expr, @input $inp:expr) => {
            test_program!(
                $desc, $e, $out, @input $inp,
                @types [], @out_fmt Escaper::Identity, @csv Some(InputFormat::CSV)
            );
        };
    }

    macro_rules! test_program_tsv {
        ($desc:ident, $e:expr, $out:expr, @input $inp:expr) => {
            test_program!(
                $desc, $e, $out, @input $inp,
                @types [], @out_fmt Escaper::Identity, @csv Some(InputFormat::TSV)
            );
        };
    }

    // TODO: extend testing macros to read non-stdout files and test more of those cases.

    #[test]
    fn basic_used_fields() {
        let p1 = r#"{ print $0; x=1; if (z) { x=3 } else { x=4 }; print $x, $5; }"#;
        let mut s1 = FieldSet::singleton(0);
        s1.set(3);
        s1.set(4);
        s1.set(5);
        assert_eq!(s1, used_fields(p1).unwrap());

        let p2 = r#"{ for (i=1; i<=NF; i++) { print $i; }; }"#;
        assert_eq!(FieldSet::all(), used_fields(p2).unwrap());
    }

    #[test]
    fn used_fields_with_joins() {
        let p1 = r#"{ print $0; x=1; if (z) { x=3 } else { x=4 }; print join_fields(x, 8); }"#;
        let mut s1 = FieldSet::singleton(0);
        s1.set(3);
        s1.set(4);
        s1.set(5);
        s1.set(6);
        s1.set(7);
        s1.set(8);
        assert_eq!(s1, used_fields(p1).unwrap());
    }

    #[test]
    fn used_fields_maps() {
        let p1 = r#"{ m[1]=1; m[2]=3; y=m[8]; for (i in m) print $i; }"#;
        let mut s1 = FieldSet::singleton(1);
        s1.set(2);
        s1.set(8);
        assert_eq!(s1, used_fields(p1).unwrap());

        let p2 = r#"{ m[1]=1; m[2]=3; y=m[8]; for (i in m) print $m[i]; }"#;
        let mut s2 = FieldSet::singleton(1);
        s2.set(3);
        s2.set(0);
        assert_eq!(s2, used_fields(p2).unwrap());

        let p3 = r#"{ m[1]=1; m[2]=3; print $m[$2]; }"#;
        s2.set(2);
        assert_eq!(s2, used_fields(p3).unwrap());
    }

    #[test]
    fn used_fields_functions() {
        // We handle functions pretty imprecisely at the moment, but basic stuff does work.

        let p1 = r#"function one() { return 1; } { print $one(); }"#;
        let s1 = FieldSet::singleton(1);
        assert_eq!(s1, used_fields(p1).unwrap());
    }

    #[test]
    fn used_fields_global_variable_store_poisons() {
        // frawk used to get this one wrong and build a used-field set of {2}.
        let p1 = r#"function unused() { print x; } { x=2; x=NF; print $x; }"#;
        let s1 = FieldSet::all();
        assert_eq!(s1, used_fields(p1).unwrap());
    }

    test_program_parallel!(
        parallel_aggs,
        ShardPerFile,
        r#"BEGIN { y="hello"; }
        { x += $1; y= y " there"; m[1]++; p=PID;}
        PREPARE {
            if (y == "hello") {
                y = "hello there";
            }
            if (PID == 1) {
                print "worker done";
            }
        }
        END {
            print x, y, m[1], length(m), (p>0);
        }"#,
        r#"1,2<<<FILE BREAK>>>3,4<<<FILE BREAK>>>5,6<<<FILE BREAK>>>7,8
9,10"#,
        "worker done\n25 hello there 5 1 1\n"
    );

    test_program_parallel!(
        header_first,
        ShardPerFile,
        r#"BEGIN { getline; print $1,$2; }
        { x += $1;}
        END { print x; }"#,
        r#"1,2<<<FILE BREAK>>>3,4<<<FILE BREAK>>>5,6<<<FILE BREAK>>>7,8
9,10"#,
        "1 2\n24\n"
    );

    test_program!(
        map_default_args,
        r#"
        function unused() { return 72; }
       function ap(n,  m) {
       m[n] = n + 1;
       for (k in m) {
        print k, m[k]
       }
       }
       BEGIN{ ap(0, m); ap(1); ap(2); ap(3); }"#,
        "0 1\n1 2\n2 3\n3 4\n"
    );

    test_program!(
        map_join_points_1,
        r#"BEGIN { if (1) { m1[1]=3; y=m1; } else { m2[2]=5; y=m2}; print y[1], y[2]; }"#,
        "3 0\n"
    );

    test_program!(
        map_join_points_2,
        r#"{ if ($1) { y=ARGV; print y[1];} }"#,
        ""
    );

    test_program!(
        rand_basics,
        r#" BEGIN {
        srand(1234);
        seed1 = srand(1234);
        x1 = rand(); y1 = rand(); z1 = rand();
        seed2 = srand(1234);
        x2 = rand(); y2 = rand(); z2 = rand();
        print (seed1 == seed2), (x1 == x2), (y1 == y2), (z1 == z2);
        }"#,
        "1 1 1 1\n"
    );

    test_program!(
        basic_csv_render,
        r#"BEGIN { print "hi", "there"; print "comma,\"in field","and a\ttab"; }"#,
r#"hi,there
"comma,""in field","and a\ttab"
"#,
        @out_fmt Escaper::CSV
    );

    test_program!(
        basic_tsv_render,
        r#"BEGIN { print "hi", "there"; print "comma,\"in field","and a\ttab"; }"#,
        "hi\tthere\ncomma,\"in field\tand a\\ttab\n",
        @out_fmt Escaper::TSV
    );

    test_program!(
        basic_multi_file,
        // test some OFS/ORS behavior for good measure
        r#"BEGIN { OFS="  "; ORS="~\n"} { print "["FILENAME,NR,FNR"]", $0; }"#,
          r#"[fake_stdin_0  1  1]  this is~
[fake_stdin_0  2  2]  the first file~
[fake_stdin_1  3  1]  And this~
[fake_stdin_1  4  2]  is the second file~
[fake_stdin_1  5  3]  it has one more line~
"#,
          @input r#"this is
the first file<<<FILE BREAK>>>And this
is the second file
it has one more line"#
    );

    test_program!(
        break_in_begin,
        r#" BEGIN { while(1) { i++; if (i > 6) break; } print i }"#,
        "7\n"
    );

    test_program!(
        basic_next,
        r#"{
        if ((NR%2) == 0) { next; };
        print "["FILENAME,NR,FNR"]", $0;}"#,
          r#"[fake_stdin_0 1 1] this is
[fake_stdin_1 3 1] And this
[fake_stdin_1 5 3] it has one more line
"#,
          @input r#"this is
the first file<<<FILE BREAK>>>And this
is the second file
it has one more line"#
    );
    test_program!(
        basic_next_file,
        r#"
        NR == 1 { nextfile; }
        { print "["FILENAME,NR,FNR"]", $0;}"#,
          r#"[fake_stdin_1 2 1] And this
[fake_stdin_1 3 2] is the second file
[fake_stdin_1 4 3] it has one more line
"#,
          @input r#"this is
the first file<<<FILE BREAK>>>And this
is the second file
it has one more line"#
    );

    test_program_csv!(
        csv_no_escaping,
        r#"function max(x, y) { return x<y?y:x; }
        { m=max($2+0, m); }
        END { print m; }"#,
          "3\n",
          @input "help,1\nsomeone,2\nout,3\n"
    );

    test_program_csv!(
        csv_no_escaping_partial,
        r#"function max(x, y) { return x<(y+0)?y:x; }
        { m=max($2, m);}
        END { print m; }"#,
          "3.5\n",
          @input "help,1\nsomeone,2\nout,3.5"
    );
    test_program_csv!(
        csv_quote_escape,
        r#"{ print $2; }"#,
          "1,2\t,3\"4\n",
          @input r#"help,"1,2\t,3""4",5"#
    );
    test_program_tsv!(
        tsv_escaping,
        r#"{ print $1,$2,$3; }"#,
        "1 2 3\n1\t23 4 5\n\t6\n",
        @input "1\t2\t3\n1\\t23\t4\t5\\n\\t6\n"
    );

    test_program!(
        tsv_join,
        r#"{ print join_tsv(2, 5);}"#,
        "2,b\t3,c\t4,d\t5,e\n",
        @input "1,a 2,b 3,c 4,d 5,e"
    );

    test_program!(
        csv_join,
        r#"{ print join_csv(2, 5);}"#,
        concat!(r#""2,b","3,c","4,d","5,e""#, "\n"),
        @input "1,a 2,b 3,c 4,d 5,e"
    );

    test_program!(
        raw_getline,
        r#"{ print "even", $0; getline; print "odd", $0; }"#,
        "even 0\nodd 1\neven 2\nodd 3\n",
        @input "0\n1\n2\n3"
    );

    test_program!(single_stmt, r#"BEGIN {print "hello"}"#, "hello\n");
    test_program!(
        to_lower_upper,
        r#"BEGIN { print tolower("Hi1 there"), toupper("hI there"), tolower(tolower("hi there")); }"#,
        "hi1 there HI THERE hi there\n"
    );
    test_program!(
        factorial,
        r#"BEGIN {
    fact=1
    for (i=1; i<7; i++) {
      fact *= i
    }
    print fact
}"#,
        "720\n"
    );
    test_program!(
        factorial_read_line,
        r#"{
target=$1
fact=1
for (i=1; i<=target; ++i) fact *= i
print fact
}"#,
        "24\n120\n",
        @input "4\n5\n"
    );

    test_program!(
        summorial_while,
        r#"BEGIN {
do {
    i++;
    j += i;
} while( i <= -1)
print i, j;
while (w <= 6) {
z += w++;
}
print w,z;
}"#,
        "1 1\n7 21\n"
    );

    test_program!(
        print_no_space,
        r#"{print}"#,
        "test\n",
        @input "test\n"
    );

    test_program!(
        uncalled_function,
        r#"function unused() { return 5; }
        BEGIN { print "x"; }"#,
        "x\n"
    );

    test_program!(
        join_fields,
        r#"BEGIN{OFS="!";} { print join_fields(1, 3); print join_fields(2, 200, "~"); }"#,
        "1!2!3\n2~3~4~5~6\n",
        @input "1 2 3 4 5 6"
    );

    test_program!(
        map_ops_simple,
        r#"BEGIN {
        for (i=0; i<10; ++i) m[i]=2*i;
        for (i=0; i<10; ++i) {
            res = res OFS m[i]
        }
        print res
}"#,
        " 0 2 4 6 8 10 12 14 16 18\n"
    );

    test_program!(
        recursive_array_func,
        r#"
BEGIN {
    f()
}
            
function f(depth, a, k) {
    ++n
    a[n] = 100-n
    if (depth < 3) {
        f(depth+1)
    }
    for (k in a) {
        printf "depth %d: k=%d v=%d\n", depth, k, a[k]
    }
}"#,
        r#"depth 3: k=4 v=96
depth 2: k=3 v=97
depth 1: k=2 v=98
depth 0: k=1 v=99
"#
    );

    test_program!(
        map_ops,
        r#"BEGIN {
        for (i=0; i<10; ++i) m[i]=2*i;
        for (i in m)
            m[i]++
        for (i=0; i<10; ++i) {
            if (res) {
                res = res OFS m[i]
            } else {
                res = m[i]
            }
        }
        print res
}"#,
        "1 3 5 7 9 11 13 15 17 19\n"
    );

    test_program!(
        basic_regex,
        r#"BEGIN {
        if (!0) print "not!";
        if ("hi" ~ /h./) print "yes1"
        if ("hi" ~ "h.") print "yes2"
        if ("no" ~ /h./) { } else print "no1"
        if ("no" !~ /h./) print "yes3"
        if ("no" !~ /n./) print "no2"
        if ("0123" ~ /\d+/) print "yes4"

}"#,
        "not!\nyes1\nyes2\nno1\nyes3\nyes4\n"
    );

    test_program!(
        str_index,
        r#"BEGIN {
        print index("foobar", "foo"), index("ofoobar", "foo"), index("ozdffooobar", "foo"), index("ozdffooobar", "xxx")
        }"#,
        "1 2 5 0\n"
    );

    test_program!(
        pattern_1,
        r#"/y.*/"#,
        "yellow\nyells\nyard\n",
        @input "yellow\ndog\nyells\nin\nthe\nyard"
    );

    test_program!(
        pattern_2,
        r#"
        $1 ~ /\d+/ { print $2; }
        $1 ~ /blorg.*/ { print $3; }
        "#,
        "numbers\nare\nvery\nfun!\n",
        @input r#"1 numbers
        2 are
        blorgme not very
        3 fun!"#
    );

    test_program!(
        float_funcs,
        r#"BEGIN {
        cos(0);sin(0);atan(0);log(0);log2(0);log10(0);sqrt(0);atan2(0,0);
        print sqrt(4);
        print log10("100");
        print log2("32");
        print exp(1);
        }"#,
        "2\n2\n5\n2.718281828459045\n"
    );

    test_program!(
        explicit_split_fs,
        r#" BEGIN {
    split("where is all of this going", m1);
    for (i=1; i<=6; i++) print i, m1[i]
    }"#,
        "1 where\n2 is\n3 all\n4 of\n5 this\n6 going\n",
        @input "",
        @types [ m1 :: MapIntStr, i :: Int]
    );

    test_program!(
        explicit_split,
        r#" BEGIN {
    split("where-is-this-all-going", m1, "-");
    for (i=1; i<=6; i++) print i, m1[i]
    }"#,
        "1 where\n2 is\n3 this\n4 all\n5 going\n6 \n",
        @input "",
        @types [ m1 :: MapIntStr, i :: Int]
    );

    test_program!(
        flowy_operators,
        r#" BEGIN {
        x = 1 && 0
        y = x || 3;
        w = x || "0";
        z = 3 ? 5 : 0 ? 8 : 7
        print w, x, y, z
        }"#,
        "1 0 1 5\n"
    );

    test_program!(
        map_contains,
        r#" BEGIN {
            m[0] = 1;
            m[1] = 2;
            if (0 in m) { print "yes 0!"; }
            if (1 in m) { print "yes 1!"; }
            if ("hi" in m) { print "no!"; }
            if (1)
            delete m[0]
            if (0 in m) { print "yes 2!"; }
        }"#,
        "yes 0!\nyes 1!\n",
        @input "",
        @types [m :: MapStrInt]
    );

    test_program!(
        map_clear,
        r#"BEGIN {
          m[1] = 2;
          for (k in m) print k, m[k]
          delete m
          for (k in m) print k, m[k]
        }"#,
        "1 2\n"
    );

    // Just want to make sure this compiles
    test_program!(map_clear_degenerate, r#"BEGIN { delete m; }"#, "");

    test_program!(
        function_keyword_overlap,
        r#"
        function down(i) { return i-1; } BEGIN { do { print down(3); } while(y--) ; }
        "#,
        "2\n"
    );

    test_program!(
        lengths,
        r#" BEGIN {
        h="12345"
        m[0]=1;m[1]=2;m[2]=3;
        print length(h), length(m)
        }"#,
        "5 3\n"
    );

    test_program!(
        division_parse,
        r#"BEGIN { a[0] = 4; t = 2; print "test/test\t" (a[0] / t)}"#,
        "test/test\t2\n"
    );

    test_program!(
        identity_function,
        r#"function id(x) { return x; }
        BEGIN { print 1, id(1) }"#,
        "1 1\n"
    );

    test_program!(
        degenerate_function,
        r#"function d(x) { a x; }
        BEGIN { print 1, d(1) }"#,
        "1 \n"
    );

    test_program!(
        basic_polymorphism,
        r#"function x(a, b) { return length(a) + b; }
        BEGIN {
            r0 = x(a, b)
            m[0]=0;
            r1 = x(m, 0)
            r2 = x(m, 1.5)
            r3 = x("hi", 2)
            print r0,r1,r2,r3
        }"#,
        "0 1 2.5 4\n",
        @input "",
        @types [ r0 :: Int, r1 :: Int, r2 :: Float, r3 :: Int, m :: MapIntInt  ]
    );

    test_program!(
        recursion,
        r#"function fib(n) {
            if (n == 0 || n == 1) return n;
            return fib(n-1) + fib(n-2);
        }
        END {
        for (i=0; i<8; i++) {
            print fib(i);
        }
        }"#,
        "0\n1\n1\n2\n3\n5\n8\n13\n"
    );

    test_program!(
        print_in_function,
        r#"function print_all(a, b, c) {
            print a, b, c;
        }
        BEGIN {
            print_all(1, 2);
        }"#,
        "1 2 \n"
    );

    test_program!(
        polymorphic_recursion,
        r#"function X(a, b) {
            if (length(a) > 0) {
                return X("", 5.0+b);
            }
            return length(a) + b*b;
        }
        END {
            m[0]=0;
            print X(m, 2);
        }
        "#,
        "49\n"
    );

    test_program!(
        global_from_function,
        r#"function setx(a) { x=a; }
        END {
        setx(1); setx("2"); setx(3.5);
        print x;
        }
        "#,
        "3.5\n",
        @input "",
        @types [x :: Str]
    );

    test_program!(
        local_globals,
        r#"END { x = "hi"; x = 1; y = x; }"#,
        "",
        @input "",
        @types [y :: Int]
    );

    test_program!(
        printf_1,
        r#"BEGIN { x=1; y=2.5; z="hello";
        printf "%d %10s %d\n\n", x, z, y;}"#,
        "1      hello 2\n\n"
    );

    test_program!(
        printf_2,
        r#"BEGIN { x=1; y=2.5; z="hello";
        printf("%02d %10s %02o\n\n", x+231, z, y<x);}"#,
        "232      hello 00\n\n"
    );

    test_program!(
        printf_3,
        r#"BEGIN { x=1; y=2.5; z="hello";
        printf("%02d %10s %02o\n\n", x+231, z, y<x);
        printf("%02d %10s %02o %s %d\n\n", x+231, z, y<x, 2.56, "320");}"#,
        "232      hello 00\n\n232      hello 00 2.56 320\n\n"
    );

    test_program!(
        sprintf_1,
        r#"BEGIN { x=1; y=2.5; z="hello";
        print sprintf("%02d %10s %02o\n\n", x+231, z, y<x);
        print sprintf("%02d %10s %02o %s %d\n\n", x+231, z, y<x, 2.56, "320");}"#,
        "232      hello 00\n\n\n232      hello 00 2.56 320\n\n\n"
    );

    test_program!(
        comma_patterns,
        r#"
        /START/,/END/ { print; }
        "#,
        r#"START
Hello there
how are you
END
"#,
        @input r#"This won't get printed
START
Hello there
how are you
END
dont print this!
Or this"#
    );

    test_program!(
        comma_patterns_next,
        r#"
        /START/,/END/ { next; }
        { print; }
        "#,
        r#"This will get printed
and this!
this as well
"#,
        @input r#"This will get printed
START
this shant be printed
nor this
END
and this!
this as well"#
    );

    test_program!(
        basic_match_loc,
        r#"BEGIN {
        x=match("something that should match", /t.?h/)
        print x, RSTART, RLENGTH
        y=match("something that will not match", /xxx/)
        print y, RSTART, RLENGTH
        }"#,
        "5 5 2\n0 0 -1\n"
    );

    test_program!(degenerate_map, r#"BEGIN { print m[1]; }"#, "\n");

    test_program!(
        substitutions,
        r#"BEGIN {
        x="hi there"
        y="snow ball"
        xr=sub(/h./,"b",x)
        yr=gsub(/l/,"na",y)
        print x, xr, y, yr;
        }"#,
        "b there 1 snow banana 2\n"
    );

    test_program!(
        map_substitutions,
        r#"BEGIN {
        m[1] = "snow ball";
        yr=gsub(/l/,"na",m[1])
        print m[1], yr;
        }"#,
        "snow banana 2\n"
    );

    test_program!(
        map_lookup_adds_key,
        r#"BEGIN { m[1]; h["hi"]; print (1 in m), (2 in m), ("hi" in h), ("hello" in h); }"#,
        "1 0 1 0\n"
    );

    test_program!(
        add_zero_prints_float,
        r#"{ print $2 + 1; }"#,
        "3.5\n",
        @input "x 2.5\n"
    );

    test_program!(
        bitwise_operators,
        r#"BEGIN {
        print xor(1,3);
        print or(1, 3);
        print compl(-2);
        print lshift(1, 8);
        print rshift(256, 8);
        print rshift(-1, 62);
        print rshiftl(-1, 62);
        }"#,
        "2\n3\n1\n256\n1\n-1\n3\n"
    );

    test_program!(
        column_substitutions,
        r#"{
        yr=gsub(/l/,"na")
        print $0, yr
        sub(/banana/, "globe", $2);
        print;
        }"#,
        "snow banana 2\nsnow globe\n",
        @input "snow ball"
    );

    test_program!(
        substrings,
        r#"BEGIN {
        s="this is a string";
        print(
            sprintf("[%s]", substr(s, 1, 1)),
            substr(s, 6, 4),
            substr(s, -20, 4),
            substr(s, 11, 100),
            substr(s, 11),
        );
    }"#,
        "[t] is a this string string\n"
    );
    test_program!(
        out_of_range_substr,
        r#"BEGIN { print substr("hi", 5, 10); }"#,
        "\n"
    );

    test_program!(
        arithmetic,
        r#"BEGIN { print 1+1; print 1+2*7; print 1+2*7^3; print 1+2*7^3*4; }"#,
        "2\n15\n687\n2745\n"
    );

    test_program!(
        global_variable_only_printed,
        r#"function unused() { return xyz; } BEGIN { print "hi", xyz; }"#,
        "hi \n"
    );

    test_program!(
        int_conversions,
        r#"BEGIN {
        x="123.456"
        y=int(x) + 2
        z=int(x+2.6)
        print y, z, 0XFea6, -0x63abc, hex("0xFFww"), hex("DEADBEEF");
    }"#,
        "125 126 65190 -408252 255 3735928559\n"
    );

    test_program!(
        basic_subsep,
        "BEGIN { m[1,2] = 3; for (k in m) { split(k, arr, SUBSEP); print arr[1], arr[2], m[k]; } }",
        "1 2 3\n"
    );

    test_program!(
        function_locals,
        r#"function p(n,  i,res) {
            for (i=0;i<n;i++) res+=i;
            return res;
        }
        { SUM += p($1+0); }
        END { print SUM; }"#,
        "6\n",
        @input "4"
    );

    test_program!(
        filter_newline,
        "$1 == 2\n",
        "2\n2\n",
        @input "4\n3\n2\n3\n2\n3\n"
    );

    test_program!(
        scientific_notation,
        "$1 <= 1e-8",
        r#"6.18163e-27
1.80782e-40
1.92843e-09
7.37465e-39
"#,
        @input "6.18163e-27\n1.80782e-40\n2.38296e-05\n1.92843e-09\n7.37465e-39\n"
    );

    test_program!(
        gensub_basic,
        r#"{$0 = gensub("(Hello), ([a-zA-Z]+)", "\\2", "g", $0)}; {print}"#,
        "Joe\nNick\nBye, Joe\nRick\nBye, Rich\n",
        @input "Hello, Joe\nHello, Nick\nBye, Joe\nHello, Rick\nBye, Rich\n"
    );

    test_program!(
        gensub_first,
        r#"BEGIN { v = "1234"; v = gensub("([0-9])([0-9])", "\\2", "1", v); print v}"#,
        "234\n"
    );

    test_program!(
        gensub_second,
        r#"BEGIN { v = "1234"; v = gensub("([0-9])([0-9])", "\\2", "2", v); print v}"#,
        "124\n"
    );

    test_program!(
        gensub_third, // this matches nothing
        r#"BEGIN { v = "1234"; v = gensub("([0-9])([0-9])", "\\2", "3", v); print v}"#,
        "1234\n"
    );

    test_program!(
        gensub_on_input, // this matches nothing
        r#"{print gensub("a", "b", "g")}"#,
        "bbobb\n",
        @input "aboba\n"
    );

    test_program!(map_global_var, r#"
BEGIN {
	unused_string_map["a"] = "abc"
}

function do_something(v) {
	return v
}

{ print do_something($0) }
    "#,
        "hello\n",
        @input "hello\n"
    );

    // TODO test more operators, consider more edge cases around functions
}

#[cfg(all(feature = "unstable", test))]
mod bench {
    extern crate test;
    use super::*;
    use test::{black_box, Bencher};

    // TODO if we ever want to benchmark stdin, the program_only benchmarks here will not work,
    // because "reset" does not work on stdin today.
    macro_rules! bench_program {
        ($desc:ident, $e:expr) => {
            bench_program!($desc, $e, @input "");
        };
        ($desc:ident, $e:expr, @input $inp:expr) => {
            mod $desc {
                use super::*;
                mod bytecode {
                    use super::*;
                    #[bench]
                    fn end_to_end(b: &mut Bencher) {
                        b.iter(|| {
                            black_box(bench_program($e, $inp, Escaper::Identity, ExecutionStrategy::Serial).unwrap());
                        });
                    }
                    #[bench]
                    fn program_only(b: &mut Bencher) {
                        let strat = ExecutionStrategy::Serial;
                        let a = Arena::default();
                        let prog = parse_program($e, &a, Escaper::Identity, strat).unwrap();
                        let (mut interp, fake_fs) =
                            compile_program(&a, prog, $inp, Escaper::Identity, strat).unwrap();
                        b.iter(|| {
                            black_box(
                                run_prog_nodebug(
                                    &mut interp,
                                    fake_fs.clone(),
                                ).unwrap()
                            );
                            interp.reset();
                            fake_fs.stdout.clear();
                        });
                    }
                }

                #[cfg(feature = "llvm_backend")]
                mod llvm {
                    use super::*;
                    #[bench]
                    fn end_to_end(b: &mut Bencher) {
                        b.iter(|| {
                            black_box(run_llvm($e, $inp, Escaper::Identity, None, ExecutionStrategy::Serial).unwrap());
                        });
                    }
                    #[bench]
                    fn compile_only(b: &mut Bencher) {
                        b.iter(|| {
                            black_box(compile_llvm($e, Escaper::Identity).unwrap());
                        });
                    }
                }
            }
        };
    }

    // Looks to be memory corruption somewhere in this benchmark. Inspect the llvm output
    bench_program!(
        str_movs_split,
        r#"END {
            S="It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.  However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.  “My dear Mr. Bennet,” said his lady to him one day, “have you heard that Netherfield Park is let at last?” Mr. Bennet replied that he had not.  “But it is,” returned she; “for Mrs. Long has just been here, and she told me all about it.” Mr. Bennet made no answer.  “Do you not want to know who has taken it?” cried his wife impatiently.  “You want to tell me, and I have no objection to hearing it.” This was invitation enough.  “Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune from the north of England; that he came down on Monday in a chaise and four to see the place, and was so much delighted with it, that he agreed with Mr. Morris immediately; that he is to take possession before Michaelmas, and some of his servants are to be in the house by the end of next week.”"
            n=split(S, words)
            for (i=0; i<n; i++) {
                word = words[i]
                if (word ~ /[a-zA-Z]*/) {
                    concat = concat "~" words[i];
                }
            }
            print concat;
        }"#
    );

    bench_program!(
        sum_integer_10k,
        r#"END { for (i=0; i<10000; i++) {SUM += i;}; print SUM }"#
    );
    bench_program!(
        sum_integer_hist_10k,
        r#"END { for (i=0; i<10000; i++) {SUMS[i]++; SUM += i;}; print SUM }"#
    );
    bench_program!(
        sum_integer_str_hist_10k,
        r#"END { for (i=0; i<10000; i++) {SUMS[i ""]++; SUM += i;}; print SUM }"#
    );
    bench_program!(
        recursive_fib_15,
        r#"
function fib(n) {
if (n == 0 || n == 1) {
return n;
}
return fib(n-1) + fib(n-2);
}
END { print fib(15); }"#
    );
}
