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
pub mod harness;
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
extern crate smallvec;
extern crate stable_deref_trait;
extern crate unicode_xid;

use lalrpop_util::lalrpop_mod;

lalrpop_mod!(syntax);

// TODO: put jemalloc behind a feature flag
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn main() {
    // TODO add a real main function
    let a = arena::Arena::default();
    println!(
        "{}",
        harness::run_program(&a, r#"END { print "hello" }"#, "")
            .unwrap()
            .0
    );
}
