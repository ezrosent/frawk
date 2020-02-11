#![feature(test)]
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
pub mod llvm;
pub mod runtime;
#[cfg(test)]
mod test_string_constants;
pub mod types;
extern crate elsa;
extern crate hashbrown;
extern crate jemallocator;
extern crate lalrpop_util;
extern crate lazy_static;
extern crate libc;
extern crate llvm_sys;
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
// #[global_allocator]
// static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

const _PROGRAM: &'static str = r#"
function fib(n) {
if (n == 0 || n == 1) {
return n;
}
return fib(n-1) + fib(n-2);
}
END { print fib(35); }"#;

const _PROGRAM_2: &'static str = r#"
END { for (i=0; i<100000000; i++) {SUM += i;}; print SUM }"#;
const _PROGRAM_3: &'static str = r#"
END { for (i=0; i<100000000; i++) {SUMS[i]++; SUM += i;}; print SUM }"#;
const _PROGRAM_4: &'static str = r#"
END { for (i=0; i<1000000; i++) {SUMS[i ""]++; SUM += i;}; print SUM }"#;
const _PROGRAM_5: &'static str = r#"
END { for (i=0; i<100; i++) {SUMS[i ""]++; SUM += i;}; print SUM }"#;
const _PROGRAM_6: &'static str = r#"
END { for (i=0; i<100000; i++) {CD = CD i;}; print CD }"#;
const _PROGRAM_7: &'static str = r#"
END { m[0] = 1; m[1] = 2; for (i in m) { print i, m[i]; } }"#;

fn main() {
    // TODO add a real main function
    if false {
        println!("{}", harness::bench_program(_PROGRAM_6, "").unwrap());
    }
    harness::dump_llvm(_PROGRAM_7).expect("error generating llvm:");
    println!(
        "output=[{}]",
        harness::run_llvm(_PROGRAM_7, "").expect("error generating llvm:")
    );
    // To debug bytecode, look at setting PRINT_DEBUG_INFO to true and using code.
    // let a = arena::Arena::default();
    // harness::run_program(
    //     &a,
    //     r#"BEGIN { for (i=0;i<3;i++) { x=i*i; z=i x; print z; } }"#,
    //     "",
    // )
    // .expect("error running program:");
    eprintln!("exiting cleanly");
}
