# frawk Builtin Functions and Commands

This document lists all of the builtin functions and commands supported by
frawk. For those interested in a source of truth on these components, check out
the "builtins" module in
[`src/builtins.rs`](https://github.com/ezrosent/frawk/blob/master/src/builtins.rs).

Unlike Awk, builtin functions must have parentheses directly following the
function name. Awk supports C-style syntax like `length (s)`, but only with
builtin functions: user-defined functions must still be called like `foo(x)`. In
frawk, builtin and user-defined functions are called with the same syntax: with
no spaces allowed.

## Operators

_Binary operators:_
* Arithmetic: `+`, `-`, `/`, `*`, `*`, `^` (which is exponentiation), and `%`
* Comparison (which also work on strings): `<`, `>`, `<=`, `>=`, `==`, `!=`.

_Unary Operators:_

* `$x`: Get column `x`.
* `+`, `-`: Unary "positive" and negation.
* `!`: logical negation.

## Math

* Floating-point operations: `sin`, `cos`, `atan`, `atan2`, `log`, `log2`,
  `log10`, `sqrt`, `exp` are delegated to the Rust standard library, or LLVM
  intrinsics where available.
* `rand()`: Returns a uniform random floating-point number between 0 and 1.
* `srand(x)`: Seeds the random number generator used by `rand`, returns the old
  seed.
* Bitwise operations. All of these operations coerce their operands to integers
  before being evaluated.
  * `compl(x)`: Bitwise complement.
  * `and(x, y)`: Bitwise and.
  * `or(x, y)`: Bitwise or.
  * `xor(x, y)`: Bitwise xor.
  * `lshift(x, y)`: Shift `x` left by `y` bits.
  * `rshift(x, y)`: Arithmetic right shift of `x` by `y` bits.
  * `rshiftl(x, y)`: Logical right shift of `x` by `y` bits.

## String Operations

* `s ~ re`: 1 if string `s` matches regular expression in `re`.
* `s !~ re`: Equivalent to negating the result of `s ~ re`.
* `match(s, re)`: 1 if string `s` matches the regular expression in `re`. If `s`
  matches, the `RSTART` variable is set with the start of the leftmost match of
  `re`, and `RLENGTH` is set with the length of this match.
* `substr(s, i[, j])`: The 1-indexed substring of string `s` starting from index `i`
  and continuing for the next `j` characters or until the end of `s` if `i+j`
  exceeds the length of `s` or if `s` is not provided.
* `sub(re, t, s)`: Substitutes `t` for the first matching occurrence of regular
  expression `re` in the string `s`.
* `gsub(re, t, s)`: Like `sub`, but with all occurrences substituted, not just
  the first.
* `index(haystack, needle)`: The first index within `haystack` in which the
  string `needle` occurs, 0 if `needle` does not appear.
* `split(s, m[, fs])`: Splits the string `s` according to `fs`, placing the
  results in the array `m`. If `fs` is not specified then the `FS` variable is
  used to split `s`.
* `sprintf(fmt, s, ...)`: Returns a string formatted according to `fmt` and
  provided arguments. The goal is to provide the semantics of the libc `sprintf`
  function.
* `print(s, ...) [>[>] out]`: Print the arguments `s` separated by `OFS`. If `>>
  out` is provided then the output is appended to the file `out`, if `> out` is
  provided then any data in `out` is overwritten. Parentheses are optional in
  `print`, but parsing of non-parenthesized arguments proceeds differently to
  avoid potential ambiguities.
* `printf(fmt, s, ...) [>[>] out]`: Like `sprintf` but the result of the
  operation is written to standard output, or to `out` according to the append
  or overwrite semantics specified by `>` or `>>`. Like `print`, `printf` can be
  called without parentheses around its arguments, though arguments are parsed
  differently in this mode to avoid ambiguities.
* `hex(s)`: Returns the hexadecimal integer (e.g. `0x123abc`) encoded in `s`, or
  `0` otherwise.
* `join_fields(i, j[, sep])`: Returns columns `i` through `j` (1-indexed,
  inclusive) concatenated together, joined by `sep`, or by `OFS` if `sep` is not
  provided.
* `escape_csv(s)`: Returns `s` escaped as a CSV column, adding quotes if
  necessary, replacing quotes with double-quotes, and escaping other whitespace.
* `escape_tsv(s)`: Returns `s` escaped as a TSV column. There is less to do with
  CSV, but tab and newline characters are replaced with `\t` and `\n`.
* `join_csv(i, j)`: Like `join_fields` but with columns joined by `,` and
  escaped using `escape_csv`.
* `join_tsv(i, j)`: Like `join_fields` but with columns joined by tabs and
  escaped using `escape_tsv`.
* `int(s)`: Convert `s` to an integer. Floating-point numbers are also converted
  (rounded down), potentially without a round-trip through a string
  representation.
* `tolower(s)`: Returns a copy of `s` where all uppercase ASCII characters are
  replaced with their lowercase counterparts; other characters are unchanged.
* `toupper(s)`: Returns a copy of `s` where all lowercase ASCII characters are
  replaced with their uppercase counterparts; other characters are unchanged.
* `exit [code]`: Exits the current process with the given code. `exit` attempts
  to flush any open file buffers. For parallel scripts, other worker threads
  have inputs cut off. Once those threads exit their main loop the process
  exits with the given exit code. This means that scripts with long loop
  iterations may not exit immediately. `exit` can be called with and without
  parentheses.

# Other Functions

* `close(s)` flushes all pending output to file `s` and then closes it.
* `length(x)` returns the length of `x`, where `x` can be either a string or an
  array.
* `system(s)` runs the command contained in the string `s` in a subshell,
  returning the error code, or the integer `1` if an error code was
  unavailable. The string `s` is subject to taint analysis by default.

