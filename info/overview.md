# frawk overview

_This document assumes some basic familiarity with Awk. I've found that [Awk in
20 minutes](https://ferd.ca/awk-in-20-minutes.html) is a solid introduction,
while the [grymoire entry](https://www.grymoire.com/Unix/Awk.html) provides
more detail. In keeping with common practice, I have been inconsistent in this
repo with how I capitalize "AWK."_

My copy of the [AWK book](https://en.wikipedia.org/wiki/The_AWK_Programming_Language)
begins with a simple message:

> Computer users spend a lot of time doing simple, mechanical data manipulation
> --- changing the format of data, checking its validity, finding items with
> some property, adding up numbers, printing reports and the like ... Awk is a
> programming language that makes it possible to handle such tasks with very
> short programs.

I find this diagnosis to be true today: I spend a good deal of time doing menial
text gardening. For all its foibles as a language, I've found Awk to be a very
valuable tool when such a "short program" is desirable. I wrote frawk to be able
to write Awk programs under more circumstances. This does not mean that I intend
for frawk to be a version of Awk with higher-level features; I appreciate that
Awk rarely escapes the lab of one-liners and have no desire to write large
programs in an Awk-like language.

frawk addresses two primary shortcomings I have found in Awk.

1. Lack of support for structured CSV input data.
2. Sometimes-lackluster performance.

We can take each of these in turn, and then move on to how Awk is implemented.
Before getting too far into the weeds, I want to clarify that my main goal in
starting this project was to learn something new: I wanted to write a small
compiler, I wanted to learn about LLVM, and I wanted to do some basic static
analysis. On that score frawk has been an unalloyed success.

**Disclaimer** frawk is still incomplete. I have found that it is sufficient for
my day-to-day scripting needs, but some features from Awk are not implemented
and the implementation is likely far less stable than the ones you have come to
know and love. With those caveats aside, here is why I think frawk is
interesting.

## Slightly Structured Data

frawk with the `-i csv` option will properly parse and escape CSV data. This
section explains why this is valuable.

Awk processes data line by line, splitting by a "record separator" which is
(essentially) a regular expression. That means it's easy enough to write the
script

```
awk -F',' 'NR>1 { SUM+=$2 } END { print SUM }'
```

To sum the second column in a file where commas always delimit fields. The
following input will yield `6`.

```
Item,Quantity
Carrot,2
Banana,4
```

However, the script will produce the wrong result if the input is a CSV file
with embedded commas in some fields, such as

```
Item,Quantity
Carrot,2
"The Deluge: The Great War, America and the Remaking of the Global Order, 1916-1931", 3
Banana,4
```

In this case, the second field of the third line will be the text "America and
the Remaking of the Global Order", which will be silently coerced to the number
0, thereby contributing nothing to the total and undercounting the sum by 3.

In other words, the standard Awk syntax of referencing columns by `$1`,`$n`,
etc.  may not work if the input is an escaped CSV file. In practice, I've found
that I can't trust Awk to work on a large CSV file where I cannot manually
verify that no fields contain embedded `,`s. frawk with the `-i csv` option will
properly parse and escape CSV data. Awk is a sufficiently expressive language
that one could parse the CSV manually, but doing so is both difficult and
inefficient.

## Efficiency, and Purpose-Built Tools

frawk is often a good deal faster than utilities like
[gawk](https://www.gnu.org/software/gawk/) and
[mawk](https://invisible-island.net/mawk/) when parsing large data files or
performing particularly computation-intensive tasks. The main reasons for
frawk's higher performance are:

1. frawk infers types for its variables, so it decides which variables are
   numbers and which are strings before it runs the program. This can speed up
   arithmetic in tight loops, and it also eliminates branching when it comes to
   performing coercions between strings and numbers. frawk achieves this while
   maintaining just about all of Awk's semantics: the only type errors frawk
   gives you are type errors in Awk, as well.
1. The fact that frawk produces a typed representation allows it to generate
   fairly simple LLVM IR and then JIT that IR to machine code at runtime. This
   avoids the overhead of an interpreter at the cost of a few milliseconds of
   time at startup. frawk provides a bytecode interpreter (enabled via the `-b`
   flag) for smaller scripts and for help in testing.
1. frawk uses some fairly recent techniques for [efficiently validating
   UTF-8](https://github.com/lemire/fastvalidate-utf-8), [parsing
   CSV](https://github.com/geofflangdale/simdcsv), and [parsing floating point
   numbers](https://github.com/lemire/fast_double_parser). On top of that, it
   leverages high-quality implementations of [regular
   expressions](https://github.com/rust-lang/regex) and [standard
   collections](https://github.com/rust-lang/hashbrown), along with several
   other useful crates from the Rust community.

The fact that existing Awk implementations tend to lack great support for CSV,
combined with lackluster performance on larger datasets when compared with a
language like Rust, has meant that developers have resorted to building custom
tools for processing CSV and TSV files. Tools like
[xsv](https://github.com/BurntSushi/xsv) and
[tsv-utils](https://github.com/eBay/tsv-utils) are great; if you haven't checked
them out and you find yourself processing large amounts of tabular data, it's
worth your while to take them for a spin. But while these tools are fast, they
aren't a full substitute for Awk. They do not provide a full programming
language, and it can be cumbersome to perform even moderately complex operations
with them.

I've found that even for short programs, frawk performs comparably to xsv on
CSV data, and within a factor of 2 or 3 on TSV data when compared with
tsv-utils.  frawk can perform tsv-utils-like queries on CSV data in
substantially less time than the bundled `csv2tsv` tool can convert the data to
TSV. I think that is a pretty good trade-off if you want to perform a
higher-level operation that these other tools do not support. See the
[benchmarks](https://github.com/ezrosent/frawk/blob/master/info/performance.md)
doc for hard numbers on this.

## frawk's structure

frawk is structured like a conventional compiler and interpreter. Given source
code, frawk parses it, converts it into a few intermediate representations,
generates lower level code, and executes it.

1. The [lexer](https://github.com/ezrosent/frawk/blob/master/src/lexer.rs)
   tokenizes the frawk source code.
1. The parser produces an abstract syntax tree
   ([AST](https://github.com/ezrosent/frawk/blob/master/src/ast.rs)) from the
   stream of tokens.
1. The AST is converted to an untyped control-flow-graph
   ([CFG](https://github.com/ezrosent/frawk/blob/master/src/cfg.rs)). We perform
   [SSA](https://en.wikipedia.org/wiki/Static_single_assignment_form)
   [conversion](https://github.com/ezrosent/frawk/blob/master/src/dom.rs) on
   this CFG.
1. With the CFG in SSA form, an [inference
   algorithm](https://github.com/ezrosent/frawk/blob/master/src/types.rs)
   assigns types to all variables in the program.
1. Given the untyped CFG and the results of the inference algorithm, we can
   produce a typed CFG with explicit bytecode instructions. (happens
   [here](https://github.com/ezrosent/frawk/blob/master/src/compile.rs))
1. From there, the code is lowered into one of (a) [bytecode
   instructions](https://github.com/ezrosent/frawk/blob/master/src/bytecode.rs)
   that can be
   [interpreted](https://github.com/ezrosent/frawk/blob/master/src/interp.rs)
   directly or (b)
   [LLVM-IR](https://github.com/ezrosent/frawk/blob/master/src/llvm/mod.rs) that
   is JIT-compiled and then run.

Most of this is fairly standard. The first few steps can be found (for example)
in the [Tiger Book](https://www.cs.princeton.edu/~appel/modern/ml/). I used
that as a primary reference, along with some reading on alternatives to the
Lengauer-Tarjan algorithm for SSA construction that were published after the
Tiger Book.

You can view a textual representation of the untyped CFG by passing the
`--dump-cfg` flag to frawk. Bytecode and LLVM can be viewed with the
`--dump-bytecode` and `--dump-llvm` options. The latter will be optimized;
passing `-O0` will roughly show the LLVM constructed by frawk.

To avoid long compile times and complicated builds, the LLVM code makes function
calls into the same runtime that is used to interpret bytecode instructions.
Smuggling more of the runtime code into the LLVM at build time would likely
result in a faster program, because it would give LLVM more opportunities to
inline and optimize runtime calls. The current approach helps keep build times
low, and the build setup simple.

### Static Analysis

I read through the delightful [_Static Program
Analysis_](https://cs.au.dk/~amoeller/spa/) book while building frawk. Among
other things, it showed me that many properties about a program can be
approximated as the solution of (potentially recursive!) equations defined on a
suitable partial order, so long as the functions defining those equations are
[monotone](https://en.wikipedia.org/wiki/Monotonic_function#Monotonicity_in_order_theory).
Furthermore, one can solve these equations by running them through simple
[propagator-style](https://www.youtube.com/watch?v=s2dknG7KryQ) networks until
their values stop changing. Primary examples of this in frawk are:

* [Inferring types](https://github.com/ezrosent/frawk/blob/master/src/types.rs)
  for frawk variables. For more on type inference, see
  [this doc](https://github.com/ezrosent/frawk/blob/master/info/types.md).
* [Inferring which columns do not have to be
  parsed.](https://github.com/ezrosent/frawk/blob/master/src/pushdown.rs)
* Determining [which global
  variables](https://github.com/ezrosent/frawk/blob/0cf6bd7554ba14193f32337ea54bd1a8f1401f1f/src/compile.rs#L694)
  are referenced by a function, and the functions that it calls.

These were all implemented with the help of the very useful
[petgraph](https://github.com/petgraph/petgraph) library.

## Differences from AWK

frawk's structure and language are borrowed almost wholesale from AWK; using
frawk feels very similar to using mawk, nawk, or gawk. frawk also supports many
of the more difficult Awk features to implement, like printf, and user-defined
functions. While many common idioms from AWK are supported in frawk, some
features are missing while still others provide subtly incompatible semantics.
Please file a feature request if a particular piece of behavior that you rely on
is missing; nothing in frawk's implementation precludes features from the
standard AWK language, though some might be troublesome to implement.

This list of differences is not exhaustive. In particular, I would not be at all
surprised to discover there were bugs in frawk's parser.

### What is missing

* frawk does not currently support piping output to separate commands, or the
  `system` function. From what I understand, functions like this (where an
  arbitrary string is passed wholesale to a shell) are considered anti-patterns,
  and have been deprecated [in some
  languages](https://www.python.org/dev/peps/pep-0324/#id14). I'd be open to
  alternative interfaces: e.g. only opening pipes to a set of strings known at
  compile time, or passing arguments and commands separately, if anyone was
  interested.
* By default, frawk uses the [ryu](https://github.com/dtolnay/ryu) crate to
  print floating point numbers, rather than the `CONVFMT` variable. Explicitly
  changing the precision of floating point output requires an appropriate
  invocation of `printf` or `sprintf`.
* frawk does not support the `m[k, v]` syntactic sugar for `m[k SUBSEP v]`. This
  is because I wanted to hold open the possibility of having true tuples as keys
  for maps in frawk.
* `next`,  or `nextfile` are supported in frawk, but they can only be invoked
  from the main loop. I haven't come across any Awk scripts that use either of
  these commands from within a function, and it's a major simplification to just
  disallow this case. Again, let me know if this is an important use-case for
  you.
* Some basic Awk commands are missing (e.g. `exit`), because I have not gotten
  to them yet. Many of the extensions in gawk (e.g. bitwise operators,
  coprocessors, multidimensional arrays) are also not implemented.
* While it has never been tried, I sincerely doubt that frawk will run at all
  well --- or at all --- on a 32-bit platform. I suspect it would run much
  slower on a 64-bit non-x86 architecture.

### What is new

* frawk supports the `-i csv` and `-i tsv` command-line options, which split all
  inputs (regardless of the value of `FS` and `RS`) according to the CSV and TSV
  formats, assigning `$0` to the raw line and `$N` to the Nth field in the
  current row, fully escaped. There is also equivalent functionality for output
  CSV-escaped lines (enabled via `-o csv` and `-o tsv`).
* frawk has a builtin `join_fields` function that produces a string of a
  particular range of input columns.
* frawk provides an `int` function for converting a scalar value to an integer,
  and a `hex` function for converting a hexidecimal string to an integer. It
  also supports hexadecimal numeric literals.

### What is different

None of these differences are fundamental to frawk's approach; they _can_ be
dispensed with, if at some cost. Let me know if you find more discrepancies, or
if you find that the following are a serious hindrance:

* *Regex Syntax* frawk currently uses rust's
  [regex](https://docs.rs/regex/1.3.7/regex/) syntax. This is similar, but not
  identical, to Awk's regex syntax. I've considered implementing my own regex
  engine, or compiling Awk regexes to rust regexes; it just isn't something I've
  gotten around to doing.
* *String comparisons* Comparing one string to another string always uses
  lexicographic ordering.  When comparing two strings, Awk first tests if both
  strings are numbers and then compares them numerically if they are. I find
  these semantics fairly counter-intuitive: for one thing, it means that two
  strings that are "equal" can hash to different values in an array. It also
  means that it's pretty hard to explicitly opt into lexicographic comparison.
  On the other hand, opting into numeric comparison is fairly easy: just add one
  of the operands to 0. To preserve some idioms, frawk coerces all operands to
  numbers if one of their operands is a number; this preserves the common
  use-case of (e.g.) filtering a numeric column by a numeric constant. I've
  found these semantics to be more predictable, and also more straightforward to
  implement.
* *Null values and join points* Null values in frawk may occasionally be coerced
  to integers. For example `if (0) { x = 5 }; printf "[%s]", x;` will print `[]`
  in Awk and will print `[0]` in frawk. This is the main pattern in which
  frawk's approach to types can "leak" into actual programs.
* *UTF-8* frawk validates all input data as UTF-8. This makes it incomparable to
  some Awk implementations in terms of what input it accepts. For example in
  mawk, NUL characters are disallowed, but otherwise arbitrary byte sequences
  can be provided as input. I did this early-on because it fit my use-cases and
  allowed for UTF-8 in regex patterns, but I now think this functionality should
  have been optional. However, refactoring the code to support arbitrary
  byte-streams would be a lot of work, as the implementation uses Rust strings
  throughout.
