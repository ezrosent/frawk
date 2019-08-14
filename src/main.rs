#![feature(test)]
pub mod arena;
pub mod ast;
pub mod cfg;
pub mod ssa;
extern crate hashbrown;
extern crate petgraph;
extern crate petgraph_graphml;
extern crate smallvec;

fn main() {
    let a = arena::Arena::with_size(1024);
    let ast1: &ast::Stmt<&'static str> = {
        use ast::Expr::*;
        use ast::NumBinop::*;
        use ast::Stmt::*;
        a.alloc(|| {
            Block(vec![
                a.alloc(|| Expr(a.alloc(|| Assign(a.alloc(|| Var("i")), a.alloc(|| NumLit(1.0)))))),
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
                                AssignOp(a.alloc(|| Var("i")), Mult, a.alloc(|| NumLit(2.0)))
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
    let mut ast2 = cfg::Context::default();
    ast2.standalone_block(ast1);
    let gml = petgraph_graphml::GraphMl::new(ast2.cfg())
        .pretty_print(true)
        .export_node_weights(Box::new(|node| vec![tup!(format!("{:?}", node).into())]))
        .export_edge_weights(Box::new(|edge| vec![tup!(format!("{:?}", edge).into())]));
    println!("{}", gml.to_string());
    println!("entry={:?}", ast2.entry());
    println!("{:?}", ssa::dom_frontier(&ast2));
}
