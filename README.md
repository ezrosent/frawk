# frawk

*Note (2024, ezrosent@) While the [policy](https://github.com/ezrosent/frawk?tab=readme-ov-file#bugs-and-feature-requests)
on bugs and feature requests remains unchanged I've had much less time over the last 1-2 years to devote to bug fixes and
feature requests for frawk. Other awks are more actively maintained, and CSV support is now a much
more common feature in awk compared to when this project started; I'll update this notice if frawk's status changes.*

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
* [Parallelism](https://github.com/ezrosent/frawk/blob/master/info/parallelism.md):
  An overview of frawk's parallelism support.
* [Benchmarks](https://github.com/ezrosent/frawk/blob/master/info/performance.md):
  A sense of the relative performance of frawk and other tools when processing
  large CSV or TSV files.
* [Builtin Functions Reference](https://github.com/ezrosent/frawk/blob/master/info/reference.md):
  A list of builtin functions implemented by frawk, including some that are new
  when compared with Awk.

frawk is dual-licensed under MIT or Apache 2.0.

## Installation

*Note: frawk uses some nightly-only Rust features by default.
Build [without the `unstable`](https://github.com/ezrosent/frawk#building-using-stable)
feature to build on stable.*  

You will need to [install Rust](https://rustup.rs/). If you have not updated rust in a while, 
run `rustup update nightly` (or `rustup update` if building using stable). If you would like
to use the LLVM backend, you will need an installation of LLVM 12 on your machine: 

* See [this site](https://apt.llvm.org/) for installation instructions on some debian-based Linux distros.
  See also the comments on [this issue](https://github.com/ezrosent/frawk/issues/63) for docker files that
  can be used to build a binary on Ubuntu.
* On Arch `pacman -Sy llvm llvm-libs` and a C compiler (e.g. `clang`) are sufficient as of September 2020.
* `brew install llvm@12` or similar seem to work on Mac OS.

Depending on where your package manager puts these libraries, you may need to
point `LLVM_SYS_120_PREFIX` at the llvm library installation (e.g.
`/usr/lib/llvm-12` on Linux or `/usr/local/opt/llvm@12` on Mac OS when installing llvm@12 via Homebrew).

### Building Without LLVM

While the LLVM backend is recommended, it is possible to build frawk only with
support for the Cranelift-based JIT and its bytecode interpreter. To do this,
build without the `llvm_backend` feature. The Cranelift backend provides
comparable performance to LLVM for smaller scripts, but LLVM's optimizations
can sometimes deliver a substantial performance boost over Cranelift (see the
[benchmarks](https://github.com/ezrosent/frawk/blob/master/info/performance.md)
document for some examples of this).

### Building Using Stable

frawk currently requires a nightly compiler by default. To compile frawk using stable,
compile without the `unstable` feature. Using `rustup default nightly`, or some other
method to run a nightly compiler release is otherwise required to build frawk.

### Building a Binary

With those prerequisites, cloning this repository and a `cargo build --release`
or `cargo [+nightly] install --path <frawk repo path>` will produce a binary that you can
add to your `PATH` if you so choose:

```
$ cd <frawk repo path>
# With LLVM
$ cargo +nightly install --path .
# Without LLVM, but with other recommended defaults
$ cargo +nightly install --path . --no-default-features --features use_jemalloc,allow_avx2,unstable
```

frawk is now on [crates.io](https://crates.io/crates/frawk), so running 
`cargo +nightly install frawk` with the desired features should also work.

While there are no _deliberate_ unix-isms in frawk, I have not tested it on Windows.
frawk does appear to build on Windows with default features disabled; see comments on [this issue](https://github.com/ezrosent/frawk/issues/87)
for more information.

## Bugs and Feature Requests

frawk has bugs, and many rough edges. If you notice a bug in frawk, filing an issue
with an explanation of how to reproduce the error would be very helpful. There are
no guarantees on response time or latency for a fix. No one works on frawk full-time.
The same policy holds for feature requests.
