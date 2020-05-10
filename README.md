# frawk

frawk is a small programming language for writing short programs processing
textual data. To a first approximation, it is an implementation of the
[AWK](https://en.wikipedia.org/wiki/AWK) language; many common AWK programs
produce equivalent output when passed to frawk. You might be interested in frawk
if you want your scripts to handle escaped CSV/TSV like standard AWK fields, or
if you want your scripts to execute faster.

For more information about how frawk works, check out these documents:
* TODO: frawk I: overview
* TODO: frawk II: type inference
* TODO: frawk III: bytecode and JIT compilation
* TODO: frawk IV: other optimizations
* TODO: frawk features and differences from AWK

frawk is dual-licensed under MIT or Apache 2.0.

## Installation

You will need an installation of LLVM 9.0 on your machine. See [this
site](https://apt.llvm.org/) helpful on Linux, and `brew install llvm@9` or
similar to work on Mac OS.

## Bugs and Feature Requests

frawk has bugs. If you notice a bug in frawk, filing a bug with an explanation
of how to reproduce the error would be very helpful. There are no guarantees on
response time or latency for a fix. No one works on frawk full-time. The same
policy holds for feature requests.
