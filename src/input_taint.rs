//! This module performs a static _taint analysis_ of a (typed) frawk program.
//!
//! The goal is to detect any strings that are passed to the system shell whose content is
//! "tainted" by user input. At a high level, we want to allow executing string constants and
//! concatentations of multiple string constants, but we don't want to allow anything that has
//! "touched" user input. We do this to avoid scripts being unexpectedly hijacked based on user
//! input abusing (e.g.) shell escaping rules in an unexpected way. "tainted", a term we borrow
//! from Perl's taint checking, essentially means "whose contents are derived from". For example:
//!
//! >  x = "dog" $1;
//!
//! Taints the variable `x`, because it contains text from user input ($1).  For the time being,
//! numbers derived from user input also count, so `x` is tainted in this case as well:
//!
//! > x = "dup" length($1);
//!
//! On the other hand:
//!
//! > if ($1 ~ "dog) { x = "echo" } else { x = "tee" }
//!
//! Does not taint `x`, because while `x`'s value depends on user input, the set of possible values
//! `x` might take on dynamically remains the same regardless of what values are passed as input.
//! Note that unlike Perl's taint checking, frawk's is (a) on by default and (b) is checked
//! statically. The static nature of the analysis drastically limits the overhead of the checking,
//! but it does mean we have to be conservative at points. Some of these limitations are inherent,
//! some are a product of the simplicity of the analysis:
//!
//! * We do not handle functions in all generality. We reject programs like:
//!
//! > function x(a, b) { print $b; return "echo " a; }
//! > BEGIN { while (x("hi", 0) | getline) print; }
//!
//! Even though the returned command only contains "static" components found in the program.
//!
//! * Map inserts that contain user inputs taint all of that map's keys and values.
//! * The analysis is flow-insensitive; the following programs may be rejected (though you may get
//!   lucky, depending on SSA), even though a dynamic analysis would probably run them program
//!   without issue:
//!
//! > BEGIN { x= "tee /tmp/out"; print "TEST" | x; x=$1; }
//! > BEGIN { x= 1?"tee /tmp/out":$1; print "TEST" | x;}
//!
//! Users who wish to execute a script they believe is safe, but is rejected by the analysis
//! (either because the analysis is too conservative, or because they trust user input) can opt out
//! of taint analysis using the -A flag.
use crate::bytecode::Instr;
use crate::common::{FileSpec, Graph, NodeIx, NumTy, WorkList};
use crate::compile::HighLevel;
use crate::dataflow::{self, Key};

use hashbrown::HashMap;
use petgraph::Direction;

#[derive(Default)]
pub struct TaintedStringAnalysis {
    flows: Graph</*tainted=*/ bool, ()>,
    regs: HashMap<Key, NodeIx>,
    queries: Vec<Key>,
    wl: WorkList<NodeIx>,
}

impl TaintedStringAnalysis {
    pub(crate) fn visit_hl(&mut self, cur_fn_id: NumTy, inst: &HighLevel) {
        // A little on how this handles functions. We do not implement a call-sensitive analysis.
        // Instead, we ask "is the return value of a function always tainted by user input?" when
        // analyzing the function body and "are any of the arguments tainted by user input" at the
        // call site. This is a bit simplistic, i.e. it rules out scripts like:
        //
        // function cmd(x, y) { print $x; return y; }
        // { print "X" | cmd(2, "tee empty-line") }
        //
        // Which should be safe.
        dataflow::boilerplate::visit_hl(inst, cur_fn_id, |dst, src| self.add_dep(dst, src.unwrap()))
    }
    pub(crate) fn visit_ll<'a>(&mut self, inst: &Instr<'a>) {
        // NB: this analysis currently tracks taint even in string-to-integer operations. I cannot
        // currently think of any security issues around interpolating an arbitrary integer (or
        // float, though perhaps that is more plausible) into a shell command. It's a easy and
        // mechanical fix to break the chain of infection on integer boundaries like this, but we
        // should read up on the potential attack surface first.
        use Instr::*;
        match inst {
            ReadErr(dst, cmd, is_file) => {
                self.add_src(dst, true);
                if !*is_file {
                    self.queries.push(cmd.into());
                }
            }
            NextLine(dst, cmd, is_file) => {
                self.add_src(dst, true);
                if !*is_file {
                    self.queries.push(cmd.into())
                }
            }
            GetColumn(dst, _) => self.add_src(dst, true),
            ReadErrStdin(dst) => self.add_src(dst, true),
            NextLineStdin(dst) => self.add_src(dst, true),
            StoreConstStr(dst, _) => self.add_src(dst, /*tainted=*/ false),
            StoreConstInt(dst, _) => self.add_src(dst, /*tainted=*/ false),
            StoreConstFloat(dst, _) => self.add_src(dst, /*tainted=*/ false),
            PrintAll {
                output: Some((cmd, FileSpec::Cmd)),
                ..
            }
            | Printf {
                output: Some((cmd, FileSpec::Cmd)),
                ..
            } => self.queries.push(cmd.into()),
            RunCmd(dst, cmd) => {
                self.queries.push(cmd.into());
                self.add_src(dst, true);
            }
            _ => dataflow::boilerplate::visit_ll(inst, |dst, src| {
                if let Some(src) = src {
                    self.add_dep(dst, src)
                }
            }),
        }
    }
    fn get_node(&mut self, k: Key) -> NodeIx {
        let flows = &mut self.flows;
        let wl = &mut self.wl;
        self.regs
            .entry(k)
            .or_insert_with(|| {
                let ix = flows.add_node(false);
                wl.insert(ix);
                ix
            })
            .clone()
    }
    fn add_dep(&mut self, dst_reg: impl Into<Key>, src_reg: impl Into<Key>) {
        let src_node = self.get_node(src_reg.into());
        let dst_node = self.get_node(dst_reg.into());
        self.flows.add_edge(src_node, dst_node, ());
    }
    fn add_src(&mut self, reg: impl Into<Key>, tainted: bool) {
        let ix = self.get_node(reg.into());
        let w = self.flows.node_weight_mut(ix).unwrap();
        if *w != tainted {
            *w = tainted;
            self.wl.insert(ix);
        }
    }

    pub(crate) fn ok(&mut self) -> bool {
        // TODO: add context to the "false" case here.
        if self.queries.len() == 0 {
            return true;
        }
        self.solve();
        for q in self.queries.iter() {
            if *self.flows.node_weight(self.regs[q]).unwrap() {
                return false;
            }
        }
        true
    }

    fn solve(&mut self) {
        while let Some(n) = self.wl.pop() {
            let start = *self.flows.node_weight(n).unwrap();
            if start {
                continue;
            }
            let mut new = start;
            for n in self.flows.neighbors_directed(n, Direction::Incoming) {
                new |= *self.flows.node_weight(n).unwrap();
            }
            if !new {
                continue;
            }
            *self.flows.node_weight_mut(n).unwrap() = new;
            for n in self.flows.neighbors_directed(n, Direction::Outgoing) {
                self.wl.insert(n)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::common::Result;

    fn compiles(p: &str, allow_arbitrary: bool) -> Result<()> {
        use crate::arena::Arena;
        use crate::ast;
        use crate::cfg;
        use crate::common::{ExecutionStrategy, Stage};
        use crate::compile;
        use crate::lexer::Tokenizer;
        use crate::parsing::syntax::ProgParser;
        use crate::runtime::{
            self,
            splitter::batch::{CSVReader, InputFormat},
        };

        use std::io;
        let a = Arena::default();
        let prog = a.alloc_str(p);
        let lexer = Tokenizer::new(prog);
        let mut buf = Vec::new();
        let parser = ProgParser::new();
        let mut prog = ast::Prog::from_stage(Stage::Main(()));
        if let Err(e) = parser.parse(&a, &mut buf, &mut prog, lexer) {
            return err!("parse failure: {}", e);
        }
        let mut ctx = cfg::ProgramContext::from_prog(&a, a.alloc_v(prog), cfg::Escaper::default())?;
        ctx.allow_arbitrary_commands = allow_arbitrary;
        let fake_inp: Box<dyn io::Read + Send> = Box::new(io::Cursor::new(vec![]));
        compile::bytecode(
            &mut ctx,
            CSVReader::new(
                std::iter::once((fake_inp, String::from("unused"))),
                InputFormat::CSV,
                /*chunk_size=*/ 1024,
                /*check_utf8=*/ false,
                ExecutionStrategy::Serial,
            ),
            runtime::writers::default_factory(),
            /*num_workers=*/ 1,
        )?;
        Ok(())
    }

    fn assert_analysis_reject(p: &str) {
        compiles(p, /*allow_arbitrary=*/ true)
            .expect(format!("program should compile without taint checks prog=[{}]", p).as_str());
        assert!(
            compiles(p, /*allow_arbitrary=*/ false).is_err(),
            "failed to rule out: {}",
            p
        );
    }

    fn assert_analysis_accept(p: &str) {
        compiles(p, /*allow_arbitrary=*/ true)
            .expect(format!("program should compile without taint checks prog=[{}]", p).as_str());
        compiles(p, /*allow_arbitrary=*/ false).expect("taint analysis should pass");
    }

    #[test]
    fn rules_out() {
        let progs: &[&str] = &[
            "BEGIN { print $1 | $2; }",
            "BEGIN { while ($1 | getline) print; }",
            "BEGIN { while (length($1) | getline) print; }",
            r#"BEGIN { while (getline x) print y | ("echo " x); }"#,
            "BEGIN { if ($2) { j = $1 3; x=j;} print y | x }",
            r#"function x(a, b) { return $2 a b; }
            BEGIN { while (x(2, 3) | getline) print; }"#,
            r#"function x(a, b) { return a b; }
            BEGIN {  print "hello" | x($2, "dog"); }"#,
            r#"function x(a, b) { return a b; }
            BEGIN {  system(x($2, "dog")); }"#,
            r#"BEGIN { for (i=1; i<10; i++) m[i]=$i; system(m[3]); }"#,
            r#"BEGIN { for (i=1; i<10; i++) m[$i]=i; for (i in m) system(i); }"#,
        ];

        for p in progs.iter() {
            assert_analysis_reject(*p);
        }
    }

    #[test]
    fn rules_in() {
        let progs: &[&str] = &[
            r#"BEGIN { print "hello" | "command"; }"#,
            r#"BEGIN { while ("command" | getline) print; }"#,
            r#"BEGIN { if ($1) x=5; else y="hi"; print "should work" | x; }"#,
            r#"function x(a, b) { print $2; return a b;}
            BEGIN { while(x("echo ", "hi") | getline) print; }"#,
            r#"function x(a, b) { return a b; }
            BEGIN {  system(x($2, "dog") ? "echo hello" : "echo goodbye"); }"#,
        ];
        for p in progs.iter() {
            assert_analysis_accept(*p);
        }
    }
}
