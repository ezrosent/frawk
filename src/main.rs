#![feature(test)]
#[macro_use]
pub mod common;
pub mod arena;
pub mod ast;
pub mod cfg;
pub mod dom;
pub mod types;
extern crate elsa;
extern crate hashbrown;
extern crate petgraph;
extern crate petgraph_graphml;
extern crate smallvec;
extern crate stable_deref_trait;

fn main() {
    let a = arena::Arena::default();
    let ast1: &ast::Stmt<&'static str> = {
        use ast::Expr::*;
        use ast::NumBinop::*;
        use ast::Stmt::*;
        a.alloc(|| {
            Block(vec![
                a.alloc(|| Expr(a.alloc(|| Assign(a.alloc(|| Var("i")), a.alloc(|| ILit(1)))))),
                a.alloc(|| {
                    Expr(a.alloc(|| {
                        Assign(
                            a.alloc(|| Var("j")),
                            a.alloc(|| Binop(Ok(Plus), a.alloc(|| Var("i")), a.alloc(|| Var("j")))),
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
                    ForEach(
                        "x",
                        a.alloc(|| Var("z")),
                        a.alloc(|| Print(vec![a.alloc(|| Var("x"))], None)),
                    )
                }),
            ])
        })
    };
    macro_rules! tup {
        ($x:expr) => {{
            ("".into(), $x)
        }};
    }
    println!("ast1={:?}", ast1);
    let ast2 = cfg::Context::from_stmt(ast1).expect("ast1 must be valid");
    let gml = petgraph_graphml::GraphMl::new(ast2.cfg())
        .pretty_print(true)
        .export_node_weights(Box::new(|node| vec![tup!(format!("{:?}", node).into())]))
        .export_edge_weights(Box::new(|edge| vec![tup!(format!("{:?}", edge).into())]));
    println!("{}", gml.to_string());
    println!("entry={:?}", ast2.entry());
    println!(
        "{:?}",
        dom::DomInfo::new(&ast2.cfg(), ast2.entry()).dom_frontier()
    );
}
