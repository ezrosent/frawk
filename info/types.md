# The Role of Types in frawk

One of the more unique aspects of frawk is its approach to types. frawk analyzes
the input program and generates static and simple types for all its variables.
In doing so, it makes different trade-offs than the implementations of Awk of
which I am aware.

## Types in Awk

There are two types of values in Awk: scalars and associative arrays (the frawk
implementation calls these "maps").  These two kinds of variable cannot overlap,
so the command:

```
$ awk 'BEGIN { x=1; x[2]=3; }'
```

Yields an error:

> awk: cmd. line:1: fatal: attempt to use scalar `x' as an array.

The Awk book describes scalars as being represented by a structure containing
both a string and a numeric representation of the variable. Arithmetic
operations operate on the numeric portions of their operands, while string
operations like concatenation make use of the string representation. New scalar
values initialized with one representation can fill in the other eagerly or
lazily via Awk's coercion rules. That means commands like:

```
$ awk 'BEGIN { x = "0"; y = x + 2; x = y + 1; print x, y;}'
```

Are perfectly valid. This one prints `3 2`. This places Awk somewhere between
languages like Python, where one variable can be a string at one moment and a
dictionary at the next, and languages like Rust, where one must make a new type
to repesent a variable's ability to contain either a number or a string.

Awk arrays have string keys and have scalars as values, though some Awk
implementations have specialized arrays to handle the case where all the keys
happen to have exact integer representations.

## Types in frawk

One goal of frawk is to provide performance competitive with the equivalent
script written in Rust. To that end, frawk takes a different approach to types
than Awk.  While it retains the syntax, and most frawk programs that I write
produce the same output that they do in Awk, the runtime representation of
scalars and maps are different. frawk compiles a program to a representation in
which scalars are either always 64-bit signed integers, always double-precision
floating point values, or always strings. Associative arrays can be specialized
for the case where their keys are only integers, or their values are only
integers, floats, or strings. That gives frawk 3 scalar, and 6 array types.

> Internally, frawk also has a "Null" type to represent uninitialized variables
> and "iterator" types to handle foreach loops.

frawk does all this while retaining Awk's dynamic feel: you can assign a
variable to both a string and a number, you can add strings to other strings and
get a number, and no type declarations are necessary. This may seem like a hard
thing to do efficiently.

I say "efficiently" because one easy way to implement this idea would be to make
all variables strings, and then coerce them to their appropriate values when,
say, we needed to perform arithmetic on them. This is a recipe for very slow
execution, because there is no caching of string-to-number coercions.  To avoid
this slowdown, frawk deduces when variables can be safely represented as
integers or floating point values, rather than as strings, where "safe" here
means no program could observe the difference. There are a host of advantages to
representing variables this way.

Not only does representing a variable as an integer rather than an integer,
string pair reduce space consumption. It also allows us to eliminate redundant
string conversions (in the eager case) and branching during arithmetic (in the
lazy case). The same goes for arrays with all-integer keys. Furthermore,
representing variables as their raw types and making coercions explicit provides
LLVM with more opportunities to optimize the frawk program.

The standard tactic for deducing types in a "dynamic" language is to observer
program behavior within a [tracing
JIT](https://en.wikipedia.org/wiki/Tracing_just-in-time_compilation) compiler.
This approach lets you generate custom code based on the type a variable has _in
practice_. While tracing JITs have been quite successful for languages like
JavaScript, I think they may be overkill for a language like Awk, which is a lot
simpler, and is optimized for programs that only run for a few seconds.

frawk achieves Awk-style semantics and runtime efficiency while implementing a
static compilation model. It does this by combining a heuristic to "split"
single variables into multiple variables with individually more precise types
(SSA form) with a analysis algorithm to assign types to variables, array keys,
and array values.

## SSA Form

[Static Single
Assignment](https://en.wikipedia.org/wiki/Static_single_assignment_form) (SSA)
form is a popular compiler intermediate representation. Without getting into too
many details, SSA conversion transforms programs that look like:

```
x = "string"
x = 3
```
Into programs like:
```
x0 = "string"
x1 = 3
```

SSA typically handles control flow by decomposing a program into "basic blocks"
of branch-free instructions along with (potentially conditional) jumps to other
basic blocks. These basic blocks and branches form a directed graph called a
[Control Flow Graph](https://en.wikipedia.org/wiki/Control-flow_graph). The
value of a variable might depend on the path used to reach the basic block
where the assignment takes place, for that SSA has the notion of a phi
function. Phi functions (aka "phi nodes") can query the last basic block to
have been visited, and pick a value based on that information So we might
translate the following program:

```
x = 0
if (x) {
    x = 3
} else {
    x = 7
}
print x
```

Into the following SSA:
```
0:
    x0 = 0
    # If x0, jump to label 1, otherise jump to label 2
    brif x0 1: 2:
1:
    x1 = 3
    jmp 3:
2:
    x2 = 7
    jmp 3:
3:
    x3 = phi[1: x1, 2: x2]
    print(x3)
```

SSA makes a lot of things easier, but the initial motivation for transforming
frawk programs to SSA is that it breaks up assignments of multiple scalar types,
so the program `x=1; x="hello"` turns into an assignment to two separate
variables. While this conversion allows us to model some programs more
precisely, it doesn't work in all cases. For one thing, we cannot perform this
same conversion on global variables (except for the ones that are only accessed
from the main loop), or to the types of map keys and values.  It also doesn't
help if variables with different types land in the same phi node (e.g. `x = y ?
"z" : 3`). In these cases, frawk has rules for approximating the type: In the
previous example, where `x` is potentially assigned to either a string and a
number, `x`'s static type will be promoted to `String`.

To see the untyped SSA output for a frawk program, pass the `--dump-cfg` flag.

## Type Inference

Once a program is in SSA form, frawk still has to pick types for all the
variables and insert coercions where a variable is used in a context not
matching its type. The former task is harder than the latter.

For example, `+` in Awk is only ever numeric. The expressions `"3"+2.0` and `3+2.0`
both evaluate to `5.0`. But once frawk knows that "3" is a string, it isn't too
hard to figure out that it should be coerced to a floating point value before
adding it to `2.0`.

While it is simple enough to infer types for variables that are only assigned to
constants, we need to do more work to infer types for variables that are
assigned to other variables: e.g. `x=3; m[x] = "y"; z=m[5]`. The solution I'm
most familiar with for this problem is to apply a type inference procedure.

In my experience, [type
infererence](https://papl.cs.brown.edu/2017/Type_Inference.html) usually refers
to unification-based algorithms for
[Damas-Hindley-Milner](https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system)-esque
systems. I recommend the links in this paragraph for a more careful exposition,
but for a hint about how these algorithms work, consider this program:

```
x = 1
y = x
z = y
x = z
```

The goal of a type inference algorithm is to provide an assignment from
variables to types. In this case, we'd probably want all variables mapped to
"number".

### Equality Constraints

A classic inference algorithm computes an assignment by generating a set of
constraints, and then solving those constraints. We can generate a constraint
for each line in the above program. The first line will constrain `x` to have
some numeric type (depending on your language).  The next three lines will
introduce _equality constraints_: e.g.  line 3 will constrain any assignment to
assign the same type to variable `z` as is assigned to `y`.  You can extend
HM-type algorithms to compute the approximations from the previous section (e.g.
`x=3;x="z"` could be resolved to "x has the type `String`"), but all the
variants I've encountered make heavy use of equality constraints of some kind.

The issue with equality constraints for Awk is that type information flows in
both directions. When I write `x=y`, anything I learn about the types for `y`
will flow into types for `x` _and vice-versa_. We want to avoid that for frawk:
consider the following cases (supposing that SSA cannot help us "split" these
variables).

```
y = 3
x = "string"
x = y
```

We have to pessimize `x` to have type "string" here, but `y` is only ever
assigned to integer values. If we unified the types of `x` and `y` we would
have to pessimize `y` as well. The same principle applies to loading and
storing into a map. This particular example can seem contrived, but as the
program grows in size, it can become increasingly difficult to maintain numeric
variables if having a string on either side of an assignment "infects" both
sides.

### Information Flow

frawk's type inference algorithm gives assignment to scalar variables
_unidirectional_ information flow. In the example from before:

```
y = 3
x = "string"
x = y
```

Line 1 introduces a "flows" constraint from `Integer` into `y`, from `String`
into `x` and from `y` into `x`. These are modelled as directed edges in a graph:
the type of a particular variable is taken as the "most general" of all its incoming
edges: `Integer` and `Float` is approximated as `Float`, and `String` is
considered more general than anything else. Note that unification is a special
case of this model, where "flows" constraints are added in both directions. This
is how map assignment is implemented.

This is the core of how frawk's type inference works: it wires up a directed
graph and runs the "flows" rule until the values in the graph's nodes stop
changing. This is slower than unification-based algorithms (which get to use
union-find, which provides an efficient mechanism for iteratively shrinking
this graph), but essentially all of the Awk programs I write are mercifully
short, so this hasn't been much a of problem.

> **Note:** This distinction between bidirectional and unidirectional
> constraints, where the former is more efficient to implement while the latter
> is more precise, shows up elsewhere in static analysis. For example, see the
> difference between Andersen's and Steensgaard's points-to analyses, detailed
> in [this book](https://cs.au.dk/~amoeller/spa/).

> **Note:** As we do more static analysis in frawk, it would be worth looking
> into a replacement for this system based on [Abstracting Definitional
> Interpreters](https://arxiv.org/abs/1707.04755). I've played around with
> implementing this in Rust, but I found that it was a lot harder to get the
> pluggable, nested effects right than it appears to be in Racket or Haskell.
> One of these days I'll try again.

### Other Subtleties

To get this idea to handle the entire awk language we need to add more
constraints and more rules. The [full
implementation](https://github.com/ezrosent/frawk/blob/master/src/types.rs) is
currently the best source on how all the given pieces fit together. This section
gives a feel for what else is going on to get this working.

**More Rules** Maps have rules that act like "flows", but only on the key type
and map type of their value. User-defined functions have rules that encode the
ordering of their arguments (giving the graph hyperedges).

**Flexible Return Types** frawk return types can depend on argument types. For 
example, the type of `a + b` depends on the type of `a` and the type of `b`: 
adding an integer to an integer produces an integer, but adding a string to an
integer produces a floating point value (as strings may contain non-integer 
numbers), and adding a float to an integer produces a float (by convention).
Luckily, these rules can be implemented in such a way that their values do not
"oscillate" in unexpected ways, allowing the analysis to converge. This 
domain-specific logic is present in the [builtins](https://github.com/ezrosent/frawk/blob/master/src/builtins.rs)
module.

Furthermore, all of these subtleties are in play when handling user-defined
functions. We use some standard techniques to (a) essentially copy the function
body for each unique invocation and (b) cache recursive invocations in a way
that doesn't result in (too much) over-approximation when it comes to types.

## Incompatibilities

frawk's approach isn't perfect. A program making pervasive use of global
variables accessed from multiple functions might do more coercions than the same
program in gawk or mawk. frawk also allows values that are null in some
execution paths to be represented as integers, so the program:

```BEGIN { if (0) { x=6; }; print x; }```

Prints an empty line in Awk, and prints 0 in frawk. This was a pragmatic choice
I made based on the Awk programs that I write and read about. It hasn't been a
problem so far. If it _does_ become a problem, the solution seems to be to
replace frawk's "string" type with the more traditional "number and string
tuple" approach, and use the existing type machinery as a fast path.
