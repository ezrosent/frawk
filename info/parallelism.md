# Parallelism in frawk

frawk provides a simple model allowing for many simple scripts to be run in
parallel. One relatively aspect of frawk's implementation is that it can achieve
nontrivial speedups even when the input is only a single input file or stream.
The first part of this doc explains the architecture that frawk uses to read
formats like CSV in a parallel-friendly manner. Only a small amount of syntax
was added to frawk to support parallel programming, but running a script in
parallel can change the meaning of a script.The second portion provides an
overview of the semantics of a frawk script when it is run in parallel.

> Note: frawk only supports parallel execution for CSV, TSV, and scripts that
> only use a unique, single-byte field separator and single-byte record
> separator. In time, this limitation will be relaxed, but those formats are
> unlikely to support the same level of performance with record-level
> parallelism.

## Reading Input In Parallel

Why expand the language in this way in the first place? After all, existing
tools like [GNU Parallel](https://www.gnu.org/software/parallel/) already
provide a succinct means of performing shell commands in parallel. I have 2
reasons for this, with reason 2 being more important than reason 1.

1. While existing solutions are quite succinct, some simple aggregations can be
   even easier to write in a single frawk program.
2. While it is relatively straightforward to run a command in parallel across
   multiple input files, existing tools have a hard time achieving parallelism
   _within a single input file_.

frawk supports a file-per-worker model of parallelism using the `-pf` option,
but I think its support for record-level parallelism (under the `-pr` option) is
more interesting.

Consider the CSV format. Parallelizing CSV parsing is a difficult task because a
parser must take a different action based on whether characters like commas
occur inside a quoted field. Most
[existing](https://github.com/BurntSushi/rust-csv/blob/master/csv-core/src/reader.rs)
solutions I have come across construct a state machine that a parser steps
through on a byte-by-byte basis.

Recent approaches to parsing [JSON](https://arxiv.org/abs/1902.08318) and
[CSV](https://github.com/geofflangdale/simdcsv) instead perform a first pass to
write out a sequence locations of _structural characters_ within the input:
Relevant structural characters in CSV are `,`s that are not inside a quoted
field, `"` characters, `\n` and `\r\n` sequences. The remaining parsing task may
still need to step through a state machine of some kind, but this state machine
only executes once per structural character, not once per byte in the input.
Because this first phase can be implemented _extremely_ cheaply using SIMD
instructions, this approach achieves substantial end-to-end performance gains on
recent CPUs.

frawk implements this approach for scripts with CSV, TSV and
single-byte-separator inputs. Not only does this approach provide high
performance for all scripts that consume input in this form, the separation of
parsing into two phases provides a great opportunity to parallelize the reading
of a single CSV file. A single worker thread performs an initial pass on a chunk
of input data to discover structural characters, it then locates a relevant
record separator and sends that chunk off to a worker thread. That worker thread
can then finish the parsing task at its own pace.

This architecture doesn't scale perfectly --- I've seen diminishing marginal
returns after 4-6 workers depending on the machine --- but it scales fast enough
to process CSV files at >2GB/s on my laptop, which is much faster than I have
been able to process CSV otherwise. The [performance
doc](https://github.com/ezrosent/frawk/blob/master/info/performance.md) provides
measurements of the speedups that different frawk scripts achieve when run this
way, as well as comparisons to other tools performing the same task.

## The Meaning of Parallel frawk Programs

frawk supports a limited notion of parallelism suitable for performing simple
aggregations or transformations on textual data. The goal is frawk's parallelism
in its current form is to facilitate parallelizing frawk scripts that are
_already_ embarrassingly parallel. While there are many possible directions to
go from here, this strikes me as a modest but useful first step.

Like Awk, frawk executes programs as a sequence of "patterns" and "actions."
Actions are blocks of code that are executed if a pattern matches. Most patterns
are tested against each successive line of input, with the exception being
the `BEGIN` and `END` patterns whose corresponding actions are executed before
and after any input is read, respectively.

> To be precise, Awk scripts that only have a `BEGIN` pattern never read any
> input.

When frawk is passed the `pr` or `pf` command-line options, it is compiled in
_parallel mode_. In this mode, the frawk program is broken into three "stages":

1. The `BEGIN` block is executed by a single thread.
2. The main loop (i.e. pattern/action pairs aside from `BEGIN` and `END`) is
   executed independently in parallel by a configurable number of worker
   threads. Variables mentioned in both the `BEGIN` block and the main loop are
   copied to the worker threads.
3. The `END` block is executed by a single thread after all worker threads
   terminate. Variables mentioned in both the `END` block and the main loop are
   copied from each worker thread and _aggregated_ before being accessed by the
   thread executing the `END` block.

Here is a simple "select" script written in frawk that extracts the 2nd column
of an input source:
```awk
{ print $2; }
```

Suppose we knew we did not have to preserve the relative ordering of the input
document's rows: rerunning this script with the `-pr` option will run this
script in parallel. A similar benchmark in the [performance
doc](https://github.com/ezrosent/frawk/blob/master/info/performance.md) gets
close to a 2x speedup in record-oriented parallel mode, despite the fact that
writes to output files are all serialized, and all input records come from a
single file.

### Aggregations

_Implicit Aggregations_ Variables that are referenced in both the main loop and
`END` block of a script are implicitly aggregated across all worker threads.
Scalars are aggregated according to rules that are a bit arbitrary, but as we
shall see you always have the option of performing the aggregation explicitly.

* Scalars are aggregated differently based on their
  [type](https://github.com/ezrosent/frawk/blob/master/info/types.md). Integer
  and floating-point values are summed. String variables are aggregated by picking
  an arbitrary non-empty representative value from one of the worker threads.
* Maps are aggregated by performing a union of the underlying sets of key/value
  pairs, with overlapping values being aggregated according to the corresponding
  scalar rule.

This, among other things, means that simple aggregations like "sum"
```awk
{ SUM += $1 }
END { print SUM }
```
Or group-by:
```awk
{ HIST[$1}++ }
END {
    for (k in HIST) {
        print k, HIST[k];
    }
}
```
Produce the same output when run in serial and parallel modes, with the usual
caveats about map iteration ordering (undefined) and floating point addition
(it is not associative).

_Other Aggregations_ While useful, the aggregations that happen by default
are not universally applicable. Consider the embarrassingly parallel task of
finding the maximum (lexicographic) value of a particular column:

```awk
{
    if (NR==1) {
        max=$2;
    } else {
        max=max>=$2?max:$2;
    }
}
END {
    print max;
}
```

This script is no longer correct if it is run in parallel. In parallel, the
aggregation rules dictate that it will simply return _a_ maximum value observed
by a particular worker thread. To aggregate explicitly, worker threads are
provided with a `PID` control variable which takes on a positive integer value
in a contiguous numeric range, with each thread receiving a unique `PID` value.
This, combined with the implicit aggregation for maps, lets us write an
_explicit_ max aggregation.

```awk
{
    if (NR==1) {
        max[PID]=$2;
    } else {
        max[PID]=max[PID]>=$2?max[PID]:$2;
    }
}
END {
    for (pid in max) {
        if (!i++) {
            max_val = max[pid]
        } else {
            max_val = max_val>max[pid]?max_val:max[pid]
        }
    }
    print max_val;
}
```

Because the continuous map references are both annoying to write and inefficient
to execute, frawk has a `PREPARE` block which executes in the worker threads at
the end of its input:

```awk
{
    if (NR==1) {
        max=$2;
    } else {
        max=max>=$2?max:$2;
    }
}
PREPARE { max_map[PID] = max }
END {
    for (pid in max_map) {
        v = max_map[pid]
        max = max>v?:max:v
    }
    print max;
}
```

For a more involved example of an explicit aggregation, see the "Statistics"
benchmark in the [performance
doc](https://github.com/ezrosent/frawk/blob/master/info/performance.md).
