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

In addition to [installing Rust](https://rustup.rs/), you will need an
installation of LLVM 10.0 on your machine: 

* See [this site](https://apt.llvm.org/) for installation instructions on some debian-based Linux distros.
* On Arch `pacman -Sy llvm llvm-libs` and a C compiler (e.g. `clang`) are sufficient as of September 2020.
* `brew install llvm@10` or similar seem to work on Mac OS.

Depending on where your package manager puts these libraries, you may need to
point `LLVM_SYS_100_PREFIX` at the llvm library installation (e.g.
`/usr/lib/llvm-10`). While the LLVM backend is recommended, it is possible to
build frawk only with support for its bytecode interpreter: to do so, build
without the `llvm_backend` feature.

frawk currently requires a `nightly` compiler. Using `rustup default nightly`,
or some other method to run a nightly compiler release is currently required to 
build frawk.

With those prerequisites, cloning this repository and a `cargo build --release`
or `cargo [+nightly] install --path <frawk repo path>` will produce a binary that you can
add to your `PATH` if you so choose:

```
$ cd <frawk repo path>
# With LLVM
$ cargo +nightly install --path .
# Without LLVM, but with other recommended defaults
$ cargo +nightly install --path . --no-default-features --features use_jemalloc,allow_avx2 
```

frawk is now on [crates.io](https://crates.io/crates/frawk), so running 
`cargo install frawk` with the desired features should also work.

While there are no _deliberate_ unix-isms in frawk, I have not tested it on Windows.

## Bugs and Feature Requests

frawk has bugs, and many rough edges. If you notice a bug in frawk, filing an issue
with an explanation of how to reproduce the error would be very helpful. There are
no guarantees on response time or latency for a fix. No one works on frawk full-time.
The same policy holds for feature requests.
