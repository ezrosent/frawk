#![feature(test)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
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
pub mod lexer;
pub mod runtime;
pub mod types;
extern crate elsa;
extern crate hashbrown;
extern crate jemallocator;
extern crate lalrpop_util;
extern crate lazy_static;
extern crate libc;
extern crate petgraph;
extern crate rand;
extern crate regex;
extern crate ryu;
extern crate simd_json;
extern crate smallvec;
extern crate stable_deref_trait;
extern crate unicode_xid;

use lalrpop_util::lalrpop_mod;
use petgraph::dot;

lalrpop_mod!(syntax);

fn parse_prog<'a, 'inp, 'outer>(
    prog: &'inp str,
    a: &'a arena::Arena<'outer>,
) -> &'a ast::Stmt<'a, 'a, &'a str> {
    let prog = a.alloc_str(prog);
    let lexer = lexer::Tokenizer::new(prog);
    let mut buf = Vec::new();
    let parser = syntax::ProgParser::new();
    match parser.parse(a, &mut buf, lexer) {
        Ok(program) => {
            let program: ast::Prog<'a, 'a, &'a str> = program;
            a.alloc_v(program.desugar(a))
        }
        Err(e) => {
            let mut ix = 0;
            let mut msg: String = "failed to parse program:\n======\n".into();
            for line in prog.lines() {
                msg.push_str(format!("[{:3}] {}\n", ix, line).as_str());
                ix += line.len() + 1;
            }
            panic!("{}=====\nError: {:?}", msg, e)
        }
    }
}

// TODO: put jemalloc behind a feature flag
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn main() {
    // TODO: develop a test harness that maps program text, stdin => stdout.
    // TODO: add tests, debug
    let a = arena::Arena::default();
    let ast3 = parse_prog(r#" { FS=","; print x, y, z > "/tmp/x"; }"#, &a);
    eprintln!("{:?}", ast3);
    let ast1 = parse_prog(
        r#"BEGIN {
    x=1
    y=2; z=3;
    m[x]+=5;


    for (j in m) {
        if (y) {
            z++
        }
    }
    }
    { print x, y, z >> "/tmp/x"}"#,
        &a,
    );
    let ast2 = cfg::Context::from_stmt(ast1).expect("ast1 must be valid");
    use common::NodeIx;
    for e in ast2.cfg().edges(NodeIx::new(0)) {
        eprintln!("EDGE {}", e.weight());
    }
    eprintln!("n_idents={}", ast2.num_idents());
    for (k, v) in types::get_types(ast2.cfg(), ast2.num_idents())
        .expect("types!")
        .iter()
    {
        eprintln!("{:?} : {:?}", k, v);
    }
    println!("{}", dot::Dot::new(&ast2.cfg()));
    let mut bcode = compile::bytecode(&ast2, std::io::stdin()).expect("error in compilation!");
    eprintln!("INSTRS:");
    for (i, inst) in bcode.instrs().iter().enumerate() {
        eprintln!("\t[{:2}] {:?}", i, inst);
    }

    bcode.run().expect("error interpreting");
}
