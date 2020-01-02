//! This module includes some utility functions for running AWK programs from Rust code.
//!
//! TODO: make this test-only
use crate::{
    arena::Arena,
    ast,
    cfg::{self, Ident},
    common::Result,
    compile, lexer, syntax,
    types::{self, get_types},
};
use hashbrown::HashMap;
use std::io::Write;

const PRINT_DEBUG_INFO: bool = false;

type Prog<'a> = &'a ast::Prog<'a, 'a, &'a str>;

type ProgResult<'a> = Result<(
    String,                        /* output */
    String,                        /* debug info */
    HashMap<&'a str, compile::Ty>, /* type info */
)>;

pub(crate) fn run_program<'a>(
    a: &'a Arena,
    prog: &str,
    stdin: impl Into<String>,
) -> ProgResult<'a> {
    let stmt = parse_program(prog, a)?;
    run_prog(a, stmt, stdin)
}

pub(crate) fn parse_program<'a, 'inp, 'outer>(
    prog: &'inp str,
    a: &'a Arena<'outer>,
) -> Result<Prog<'a>> {
    let prog = a.alloc_str(prog);
    let lexer = lexer::Tokenizer::new(prog);
    let mut buf = Vec::new();
    let parser = syntax::ProgParser::new();
    match parser.parse(a, &mut buf, lexer) {
        Ok(program) => Ok(a.alloc_v(program)),
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

pub(crate) fn run_prog<'a>(
    arena: &'a Arena,
    prog: Prog<'a>,
    stdin: impl Into<String>,
) -> ProgResult<'a> {
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
    let ctx = cfg::ProgramContext::from_prog(arena, prog)?;
    // NB the invert_ident machinery only works for global identifiers. We could get it to work in
    // a limited capacity for locals, but it would require a lt more bookkeeping.
    let ident_map = ctx._invert_ident();
    let stdin = stdin.into();
    let stdout = FakeStdout::default();
    let (instrs, type_map) = {
        let mut instrs_buf = Vec::<u8>::new();
        for func in ctx.funcs.iter() {
            let name = func.name.unwrap_or("<main>");
            write!(&mut instrs_buf, "cfg:\nfunction {}/{}(", name, func.ident).unwrap();
            for (i, arg) in func.args.iter().enumerate() {
                // Help with pretty-printing
                use crate::display::Wrap;
                if i == func.args.len() - 1 {
                    write!(&mut instrs_buf, "{}/{}", arg.name, Wrap(arg.id)).unwrap()
                } else {
                    write!(&mut instrs_buf, "{}/{}, ", arg.name, Wrap(arg.id)).unwrap()
                }
            }
            write!(
                &mut instrs_buf,
                ") {{\n {}\n}}\nbytecode:\n",
                petgraph::dot::Dot::new(&func.cfg)
            )
            .unwrap();
        }
        let types::TypeInfo { var_tys, func_tys } = get_types(&ctx)?;
        // ident_map : Ident -> &str (but only has globals)
        // ts: Ident -> Type
        //
        // We want the types of all the entries in ts that show up in ident_map.
        let type_map: HashMap<&'a str, compile::Ty> = var_tys
            .iter()
            .flat_map(|((Ident { low, global, .. }, _, _), ty)| {
                ident_map
                    .get(&Ident {
                        low: *low,
                        sub: 0,
                        global: *global,
                    })
                    .map(|s| (*s, ty.clone()))
            })
            .collect();
        let mut interp = compile::bytecode(&ctx, std::io::Cursor::new(stdin), stdout.clone())?;
        for (i, func) in interp.instrs().iter().enumerate() {
            write!(&mut instrs_buf, "function {} {{\n", i).unwrap();
            for (j, inst) in func.iter().enumerate() {
                write!(&mut instrs_buf, "\t[{:2}] {:?}\n", j, inst).unwrap();
            }
            write!(&mut instrs_buf, "}}\n").unwrap();
        }
        let instrs = String::from_utf8(instrs_buf).unwrap();
        if PRINT_DEBUG_INFO {
            eprintln!(
                "func_tys={:?}\nvar_tys={:?}\n=========\n",
                func_tys, var_tys
            );
            eprintln!("{}", instrs);
        }
        interp.run()?;
        (instrs, type_map)
    };

    let v = match Rc::try_unwrap(stdout.0) {
        Ok(v) => v.into_inner(),
        Err(rc) => rc.borrow().clone(),
    };
    match String::from_utf8(v) {
        Ok(s) => Ok((s, instrs, type_map)),
        Err(e) => err!("program produced invalid unicode: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_program {
        ($desc:ident, $e:expr, $out:expr) => {
            test_program!($desc, $e, $out, @input "", @types []);
        };
        ($desc:ident, $e:expr, $out:expr, @input $inp:expr) => {
            test_program!($desc, $e, $out, @input $inp, @types []);
        };
        ($desc:ident, $e:expr, $out:expr, @input $inp:expr,
         @types [ $($i:ident :: $ty:expr),* ]) => {
            #[test]
            fn $desc() {
                let a = Arena::default();
                let out = run_program(&a, $e, $inp);
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
        "1 1.0 3\nhi 0.0 5\n",
        @input "",
        @types [ m :: MapStrInt, k :: Str ]
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
        lengths,
        r#" BEGIN {
        h="12345"
        m[0]=1;m[1]=2;m[2]=3;
        print length(h), length(m)
        }"#,
        "5 3\n"
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
        "49.0\n"
    );

    // This test fails, for rather problematic reasons.
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

    // TODO test functions
    // - assigning to global variables from within different function invocations
    // TODO test more operators
}
