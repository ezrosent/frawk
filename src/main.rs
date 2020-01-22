#![feature(pattern)]
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
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

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
const _PROGRAM_5: &'static str = r#"END {
            S="It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.  However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.  “My dear Mr. Bennet,” said his lady to him one day, “have you heard that Netherfield Park is let at last?” Mr. Bennet replied that he had not.  “But it is,” returned she; “for Mrs. Long has just been here, and she told me all about it.” Mr. Bennet made no answer.  “Do you not want to know who has taken it?” cried his wife impatiently.  “You want to tell me, and I have no objection to hearing it.” This was invitation enough.  “Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune from the north of England; that he came down on Monday in a chaise and four to see the place, and was so much delighted with it, that he agreed with Mr. Morris immediately; that he is to take possession before Michaelmas, and some of his servants are to be in the house by the end of next week.”"
            n=split(S, words)
            for (i=0; i<n; i++) { 
                word = words[i]
                if (word ~ /[a-zA-Z]*/) {
                    concat = concat "~" words[i];
                }
            }
            print concat;
        }"#;

fn main() {
    unsafe { llvm::test_codegen() };
    let p = llvm::__test_print as *mut u8;
    let _ = unsafe { std::ptr::read_volatile(p) };
    // TODO add a real main function
    if false {
        println!("{}", harness::bench_program(_PROGRAM_5, "").unwrap());
    }
    eprintln!("exiting cleanly");
}
