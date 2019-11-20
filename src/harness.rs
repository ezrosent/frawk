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
    let stdin = stdin.into();
    let stdout = FakeStdout::default();
    let instrs = {
        let mut instrs = format!("cfg:\n{}\ninstrs:\n", petgraph::dot::Dot::new(ctx.cfg()));
        let mut interp = compile::bytecode(&ctx, std::io::Cursor::new(stdin), stdout.clone())?;
        for (i, inst) in interp.instrs().iter().enumerate() {
            instrs.push_str(format!("[{:2}] {:?}\n", i, inst).as_str());
        }
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
        mixed_map,
        r#"BEGIN {
m[1]=2
m["1"]++
m["hi"]=5
for (k in m) {
    print k,k+0,  m[k]
}}"#,
        "1 1.0 3\nhi 0.0 5\n"
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
        pattern_1,
        r#"/y.*/"#,
        "yellow\nyells\nyard\n",
        "yellow\ndog\nyells\nin\nthe\nyard"
    );

    test_program!(
        pattern_2,
        r#"
        $1 ~ /\d+/ { print $2; }
        $1 ~ /blorg.*/ { print $3; }
        "#,
        "numbers\nare\nvery\nfun!\n",
        r#"1 numbers
        2 are
        blorgme not very
        3 fun!"#
    );

    // TODO: I suspect there is an issue with how we handle default types here that is causing a
    // failure. We need split to create a dependency on the key and value types of its argument.
    test_program!(
        explicit_split,
        r#" BEGIN {
    # XXX different failures with and without this line.
    m[0]="XXX"
    split("where is all of this going", m1, /[ \t]+/);
    for (i=1; i<=6; i++) print i, m1[i]
    }"#,
        "1 where\n2 is\n3 all\n4 of\n5 this\n6 going\n"
    );
    // TODO test more operators
}
