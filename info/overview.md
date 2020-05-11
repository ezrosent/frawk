# frawk overview

_This document assumes some basic familiarity with awk. I've found that [Awk in
20 minutes](https://ferd.ca/awk-in-20-minutes.html) is a solid introduction,
while the [grymoire entry](https://www.grymoire.com/Unix/Awk.html)
provides more detail._

The [AWK book](https://en.wikipedia.org/wiki/The_AWK_Programming_Language)
begins with a simple message:

> Computer users spend a lot of time doing simple, mechanical data manipulation
> --- changing the format of data, checking its validity, finding items with
> some property, adding up numbers, printing reports and the like ... Awk is a
> programming language that makes it possible to handle such tasks with very
> short programs.

For all its foibles as a language, I enjoy using awk when such a "short program"
is readily apparent. I wrote frawk to be able to write awk programs under more
circumstances. This does not mean that I intend for frawk to be a version of awk
with higher-level features; I appreciate that awk rarely escapes the lab of
one-liners and have no desire to write large programs in an awk-like language.
<!-- Awk was partially intended as a language in which new systems could be
prototyped before being translated into Pascal, C, or
[C++](https://www.cs.princeton.edu/~bwk/btl.mirror/awkc++.pdf). -->

frawk addresses two primary shortcomings I have found in awk.

1. Lack of support for structured CSV or TSV input data.
2. Lackluster performance.

TODO: rest of doc under construction

* motivating example: counting lines and unique lines in a file. (comparison
  with runiq, but pointing out that a fully designed solution is better).
* motivating example: computing statistics on a 1GB CSV (compare mawk, xsv).

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
