#![feature(test)]
#[macro_use]
pub mod common;
pub mod arena;
pub mod ast;
pub mod builtins;
pub mod bytecode;
pub mod cfg;
mod display;
pub mod dom;
pub mod strton;
pub mod types;
extern crate elsa;
extern crate hashbrown;
extern crate jemallocator;
extern crate lazy_static;
extern crate libc;
extern crate petgraph;
extern crate regex;
extern crate ryu;
extern crate simd_json;
extern crate smallvec;
extern crate stable_deref_trait;

use petgraph::dot;

// TODO: put jemalloc behind a feature flag
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn main() {
    let a = arena::Arena::default();
    let ast1: &ast::Stmt<&'static str> = {
        use ast::Binop::*;
        use ast::Expr::*;
        use ast::Stmt::*;
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
                        a.alloc(|| Print(vec![a.alloc(|| Var("x")), a.alloc(|| Var("i"))], None)),
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
    eprintln!("n_idents={}", ast2.num_idents());
    eprintln!("{:?}", types::get_types(ast2.cfg(), ast2.num_idents()));
    println!("{}", dot::Dot::new(&ast2.cfg()));
}
