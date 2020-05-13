# frawk overview

_This document assumes some basic familiarity with awk. I've found that [Awk in
20 minutes](https://ferd.ca/awk-in-20-minutes.html) is a solid introduction,
while the [grymoire entry](https://www.grymoire.com/Unix/Awk.html)
provides more detail._

My copy of the [AWK book](https://en.wikipedia.org/wiki/The_AWK_Programming_Language)
begins with a simple message:

> Computer users spend a lot of time doing simple, mechanical data manipulation
> --- changing the format of data, checking its validity, finding items with
> some property, adding up numbers, printing reports and the like ... Awk is a
> programming language that makes it possible to handle such tasks with very
> short programs.

I find this diagnosis to be true today: I spend a good deal of time doing menial
text gardening. For all its foibles as a language, I've found awk to be a very
valuable tool when such a "short program" is readily apparent. I wrote frawk to
be able to write awk programs under more circumstances. This does not mean that
I intend for frawk to be a version of awk with higher-level features; I
appreciate that awk rarely escapes the lab of one-liners and have no desire to
write large programs in an awk-like language.
<!-- Awk was partially intended as a language in which new systems could be
prototyped before being translated into Pascal, C, or
[C++](https://www.cs.princeton.edu/~bwk/btl.mirror/awkc++.pdf). -->

frawk addresses two primary shortcomings I have found in awk.

1. Lack of support for structured CSV or TSV input data.
2. Lackluster performance.

We can take each of these in turn, and then move on to how awk is implemented.
Before getting too far into the weeds, I want to clarify that my main goal in
starting this project was to learn something new: I wanted to write a small
compiler, I wanted to learn about LLVM, and I wanted to do some basic static
analysis. On that score frawk has been an unalloyed success.

## Slightly Structured Data

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

In other words, the standard awk syntax of referencing columns by `$1,$n` etc.
may not work if the input is an escaped CSV file. In practice, I've found that I
can't trust awk to work on a large CSV file where I cannot manually verify that
no fields contain embedded `,`s. frawk with the `-i csv` option will properly
parse and escape CSV data. Awk is a sufficiently expressive language that one
could parse the CSV manually, but doing so is both difficult and inefficient.

## Efficiency, and Purpose-Built Tools

frawk is often a good deal faster than utilities like
[gawk](https://www.gnu.org/software/gawk/) and
[mawk](https://invisible-island.net/mawk/) when
parsing large data files. The main reasons for higher performance are:

1. frawk infers types for its variables, making numbers numbers and strings
   strings. This can speed up arithmetic in tight loops, which shows up on
   occasion. It does this while maintaining just about all of awk's semantics:
   the only type errors frawk gives you are type errors in awk, as well.
1. the fact that frawk produces a typed representation allows it to generate
   fairly simple LLVM IR and then JIT that IR to machine code at runtime. This
   avoids the overhead of an interpreter at the cost of a few milliseconds of
   time at startup. frawk provides a bytecode interpreter for smaller scripts
   and for help in testing.
1. frawk uses some fairly recent techniques for [efficiently validating
   UTF-8](https://github.com/lemire/fastvalidate-utf-8), [parsing
   CSV](https://github.com/geofflangdale/simdcsv), and [parsing floating point
   numbers](https://github.com/lemire/fast_double_parser). On top of that, it
   leverages high-quality implementations of [regular
   expressions](https://github.com/rust-lang/regex) and [standard
   collections](https://github.com/rust-lang/hashbrown), along with several
   other useful crates from the Rust community.

The fact that existing awk implementations tend to lack great support for CSV,
combined with lackluster performance on larger datasets when compared with a
language like Rust, has meant that developers have resorted to building custom
tools for processing CSV and TSV files. Tools like
[xsv](https://github.com/BurntSushi/xsv) and
[tsv-utils](https://github.com/eBay/tsv-utils) are great; if you haven't checked
them out and you find yourself processing large amounts of tabular data, it's
worth your while to take them for a spin. But while these tools are fast, they
aren't a full substitute for awk. They do not provide a full programming
language, and it can be cumbersome to perform even moderately complex operations
with them.

I've found that even for short programs, frawk performs comparably to xsv on CSV
data, and within a factor of 2 or 3 on TSV data when compared with tsv-utils.
frawk can perform tsv-utils-like queries on CSV data in substantially less time
than the bundled `csv2tsv` tool can convert the data to TSV. I think that is a
pretty good trade-off if you think you may have to do a higher-level operation
that these tools do not support.

TODO: link to benchmarking doc

Lastly, I built frawk as an excuse to learn things: about compilers, about LLVM
about static analysis, about optimization. On this score at the very least, it
is a success.

## frawk's structure

* lexing, parsing, AST.
* AST lowered into an untyped CFG which is converted to SSA form
* The untyped SSA has types inferred for values as well as functions (link to
  type inference doc).
* With the types inferred, we generate insert coercions and typed high-level
  bytecode which is either translated into a register machine and interpreted or
  is translated to LLVM-IR and JIT compiled.
* Many heavyweight pieces of frawk functionality (regular expressions, strings)
  are delegated to the runtime module. Both the interpreter and LLVM backends
  make calls into the runtime module.

## Differences from AWK

frawk's structure and language are borrowed almost wholesale from AWK; using
frawk feels very similar to using awk, nawk or gawk. While many common idioms
from AWK are supported in frawk, some features are missing while still others
provide subtly incompatible semantics. Please file a feature request if a
particular piece of behavior that you rely on is missing; nothing in frawk's
implementation precludes features from the standard AWK language, though some
might be troublesome to implement.

### What is missing

* Pipes/System
* CONVFMT, SUBSEP
* `next`,  or `nextfile` inside a function.

### What is new

* proper csv/tsv handling
* `join_fields`
* `int`, `hex` functions


### What is different

* regular expression syntax
* string comparisons
* assignments to integers/strings across join points
