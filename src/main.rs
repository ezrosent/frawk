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
        Err(e) => panic!(
            "failed to parse program:\n======\n{}\n=====\nError: {:?}",
            prog, e
        ),
    }
}

// TODO: put jemalloc behind a feature flag
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn main() {
    let a = arena::Arena::default();
    let ast0 = parse_prog(
        r#"BEGIN {
    x=1
    y=2; z=3;
    m[x]+=5;
    for (j in m) {
        if (y) {
            z++
        }
    }}"#,
        &a,
    );
    eprintln!("{:?}", ast0);
    let ast1: &ast::Stmt<&'static str> = {
        use ast::{Binop::*, Expr::*, Stmt::*};
        a.alloc(|| {
            Block(vec![
                a.alloc(|| Expr(a.alloc(|| Assign(a.alloc(|| Var("i")), a.alloc(|| ILit(1)))))),
                a.alloc(|| {
                    Expr(a.alloc(|| {
                        Assign(
                            a.alloc(|| Var("j")),
                            a.alloc(|| Binop(Plus, a.alloc(|| Var("i")), a.alloc(|| Var("j")))),
                        )
                    }))
                }),
                a.alloc(|| {
                    If(
                        a.alloc(|| Var("i")),
                        a.alloc(|| {
                            Expr(a.alloc(|| {
                                AssignOp(a.alloc(|| Var("i")), Mult, a.alloc(|| FLit(2.0)))
                            }))
                        }),
                        None,
                    )
                }),
                a.alloc(|| {
                    Expr(a.alloc(|| {
                        AssignOp(
                            a.alloc(|| {
                                a.alloc(|| Index(a.alloc(|| Var("z")), a.alloc(|| FLit(0.0))))
                            }),
                            Plus,
                            a.alloc(|| StrLit("23")),
                        )
                    }))
                }),
                a.alloc(|| {
                    ForEach(
                        "x",
                        a.alloc(|| Var("z")),
                        a.alloc(|| {
                            Print(
                                vec![
                                    a.alloc(|| Var("x")),
                                    a.alloc(|| StrLit(" SEP ")),
                                    a.alloc(|| Var("i")),
                                ],
                                None,
                            )
                        }),
                    )
                }),
                // Creates an error
                // a.alloc(|| {
                //     Print(
                //         vec![
                //             a.alloc(|| Binop(Ok(Plus), a.alloc(|| Var("z")), a.alloc(|| Var("z"))))
                //         ],
                //         None,
                //     )
                // }),
            ])
        })
    };
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
