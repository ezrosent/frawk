# frawk

frawk is a small programming language for writing short programs processing
textual data. To a first approximation, it is an implementation of the
[AWK](https://en.wikipedia.org/wiki/AWK) language; many common Awk programs
produce equivalent output when passed to frawk. You might be interested in frawk
if you want your scripts to handle escaped CSV/TSV like standard Awk fields, or
if you want your scripts to execute faster.

The info subdirectory has more in-depth information on frawk:

* [Overview](https://github.com/ezrosent/frawk/blob/master/info/overview.md):
  what frawk is all about, how it differs from Awk.
* [Types](https://github.com/ezrosent/frawk/blob/master/info/types.md): A
  quick gloss on frawk's approach to types and type inference.
* [Benchmarks](https://github.com/ezrosent/frawk/blob/master/info/performance.md):
  A sense of the relative performance of frawk and other tools when processing
  large CSV or TSV files.

frawk is dual-licensed under MIT or Apache 2.0.

## Installation

In addition to [installing Rust](https://rustup.rs/), you will need an
installation of LLVM 9.0 on your machine. See [this site](https://apt.llvm.org/)
helpful on Linux, and `brew install llvm@9` or similar to work on Mac OS. Other
versions of LLVM may work as well.

## Bugs and Feature Requests

frawk has bugs. If you notice a bug in frawk, filing a bug with an explanation
of how to reproduce the error would be very helpful. There are no guarantees on
response time or latency for a fix. No one works on frawk full-time. The same
policy holds for feature requests.
