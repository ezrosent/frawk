//! This module includes some utility functions for running AWK programs from Rust code.
//!
//! TODO: make this test-only
use crate::{arena, ast, cfg, common::Result, compile, lexer, syntax};

type Stmt<'a> = &'a ast::Stmt<'a, 'a, &'a str>;

pub(crate) fn run_program(prog: &str, stdin: impl Into<String>) -> Result<(String, String)> {
    let a = arena::Arena::default();
    let stmt = parse_program(prog, &a)?;
    run_stmt(stmt, stdin)
}

pub(crate) fn parse_program<'a, 'inp, 'outer>(
    prog: &'inp str,
    a: &'a arena::Arena<'outer>,
) -> Result<Stmt<'a>> {
    let prog = a.alloc_str(prog);
    let lexer = lexer::Tokenizer::new(prog);
    let mut buf = Vec::new();
    let parser = syntax::ProgParser::new();
    match parser.parse(a, &mut buf, lexer) {
        Ok(program) => Ok(a.alloc_v(program.desugar(a))),
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

pub(crate) fn run_stmt<'a>(stmt: Stmt<'a>, stdin: impl Into<String>) -> Result<(String, String)> {
    use std::cell::RefCell;
    use std::io;
    use std::rc::Rc;
    #[derive(Clone, Default)]
    struct FakeStdout(Rc<RefCell<Vec<u8>>>);
    impl io::Write for FakeStdout {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.borrow_mut().write(buf)
        }
        fn flush(&mut self) -> io::Result<()> {
            self.0.borrow_mut().flush()
        }
    }
    let ctx = cfg::Context::from_stmt(stmt)?;
    // TODO remove commented-out eprintln
    // eprintln!("{}", petgraph::dot::Dot::new(ctx.cfg()));
    let stdin = stdin.into();
    let stdout = FakeStdout::default();
    let instrs = {
        let mut instrs = String::default();
        let mut interp = compile::bytecode(&ctx, std::io::Cursor::new(stdin), stdout.clone())?;
        for (i, inst) in interp.instrs().iter().enumerate() {
            instrs.push_str(format!("[{:2}] {:?}\n", i, inst).as_str());
        }
        // eprintln!("bcode=\n{}", instrs);
        interp.run()?;
        instrs
    };
    let v = match Rc::try_unwrap(stdout.0) {
        Ok(v) => v.into_inner(),
        Err(rc) => rc.borrow().clone(),
    };
    match String::from_utf8(v) {
        Ok(s) => Ok((s, instrs)),
        Err(e) => err!("program produced invalid unicode: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_program {
        ($desc:ident, $e:expr, $out:expr) => {
            test_program!($desc, $e, $out, "");
        };
        ($desc:ident, $e:expr, $out:expr, $inp:expr) => {
            #[test]
            fn $desc() {
                let out = run_program($e, $inp);
                match out {
                    Ok((out, instrs)) => {
                        let expected = $out;
                        assert_eq!(out, expected, "Bytecode:\n{}", instrs);
                    }
                    Err(e) => panic!("failed to run program: {}", e),
                }
            }
        };
    }

    test_program!(single_stmt, r#"BEGIN {print "hello"}"#, "hello\n");
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
        "4\n5\n"
    );

    // TODO more tests
    // * do/while, while loops
    // * Maps
    //   - print out all members
    //   - incrementing fields in a map
    //   - Do above for a mixture of key and value types,
    //     for string-keyed maps, use mixture of integers, floats
    //     in literals.
    // * All operators, even if the example is simple. Pay close attention to concatenation.
    // * Splitting.
    // * Matching.
}
