# Performance

## Disclaimer

One of frawk's goals is to be efficient. The abundance of such claims
notwithstanding, I've found it is very hard to precisely state the degree to
which an entire _programming language implementation_ is or is not efficient. I
have no doubt that there are programs in frawk that are slower than equivalent
programs in Rust or C, or even mawk or gawk (indeed, see the "Group By Key"
benchmark). In some cases this will be due to bugs or differences in the
quality of the language implementations, in others it will be due to bugs or
differences in the quality of the programs themselves, while in still others it
may be due to the fact that in some nontrivial sense one language allows you to
write the program more efficiently than another.

I've found that it can be very hard to distinguish between these three
categories of explanations when doing performance analysis. As such, I encourage
everyone reading this document to draw at most modest conclusions based on the
numbers below.  What I do hope this document demonstrates are some of frawk's
strengths when it comes to some common scripting tasks on CSV and TSV data. (Awk
and frawk can process other data formats as well, but in my experience
larger files are usually in CSV, TSV, or some similar standardized format).

## Benchmark Setup

All benchmark numbers report the minimum wall time across 5 iterations per
hardware configuration, along with the composite system and user times for that
invocation as reported by the `time` command. A computation running with wall
time of 2.5 seconds, and CPU time of 10.2s for user and 3.4s for system is
reported as "2.5s (10.2s + 3.4s)".  We also report throughput numbers, which
are computed as wall time divided by input file size.

This doc includes measurements for both parallel and serial invocations of
frawk, and it also provides numbers for frawk using its LLVM and its Cranelift
backend, with all optimizations enabled. Note that frawk adaptively chooses the
number of workers to launch for parallel invocations, so the ratio of CPU to
wall time can vary across invocations. XSV supports parallelism but I noticed
no performance benefit from this feature without first building an index of the
underlying CSV file (a process which can take longer than the benchmark task
itself without amortizing the cost over multiple runs).  Similarly, I do not
believe it is  possible to parallelize the other programs using generic tools
like `gnu parallel` without first partitioning the data-set into multiple
sub-files.

### `-itsv` vs `-F'\t'`

frawk allows you to specify the input format as TSV using the `itsv` option, but
it also provides support for traditional Awk field separators using `-F` or by
setting the `FS` variable.  These two are not the same; they end up invoking two
completely separate parsers under the hood. `-itsv` looks for escape sequences
(like `\t`) and converts them to their corresponding characters. `-F'\t'`, on
the other hand, does a more naive "split by tabs" algorithm. `-F'\t'` tends to
perform faster than `-itsv` because parsing is less complex, and field size
computations are easier to perform. The latter fact makes it easier to optimize
scripts that do not parse a large suffix of columns in the input (e.g. only
using columns 3 and 5 of a 30-column file). This is a feature of most of the
benchmarks in this document, and, as I understand it, it's a big part of why
tsv-utils is a consistent leader in performance here.

Because tsv-utils, mawk and gawk are all invoked without the extra escaping
behavior, the frawk invocations all use `-F'\t'` on TSV inputs. If you are
curious about how frawk performs using `itsv` but don't want to run these
benchmarks yourself,  it's my experience that `frawk -itsv` achieves similar
(albeit slightly higher) throughput to using `icsv` on an equivalent CSV file.

### Test Data
These benchmarks run on CSV and TSV versions of two data-sets:

* `all_train.csv` from the
  [HEPMASS dataset](https://archive.ics.uci.edu/ml/datasets/HEPMASS) in the UCI
  machine learning repository. This is a 7 million-row file and both CSV and TSV
  versions of the file are roughly 5.2GB, uncompressed.
* `TREE_GRM_ESTN.csv` from The Forest Service (data-sets available
  [here](https://apps.fs.usda.gov/fia/datamart/CSV/datamart_csv.html)). This
  file is a little over 36 million rows; the CSV version is 8.9GB, and the TSV
  version is 7.9GB.

### Hardware

This doc reports performance numbers from two machines, a new (late 2019) MacOS
laptop, and an older (mid-2016, Broadwell-based) workstation running Ubuntu.
They're called out as "MacOS" and "Linux" not because I take these numbers to be
benchmarks of those operating systems, but as a shorthand for the whole
configuration, software and hardware, used in those benchmarks. As the Tools
section makes clear, the versions of the tools I used are all slightly
different. Comparisons within the same configuration should be safe while
comparisons across configurations are less so.

**MacOS (Newer Hardware)** This is a 16-inch Macbook Pro running MacOS Big Sur
11.2.1 with an 8-core i9 clocked at 2.3GHz, boosting up to 4.8GHz. One thing to
keep in mind about newer Apple hardware is that the SSDs are very fast: read
throughput can top 2.5GB/s. None of these benchmarks are IO-bound on this machine,
although various portions of the input files might be cached by the OS.

**Linux (Older Hardware)** This is a dual-socket workstation with 2 8-core Xeon
2620v4 CPUs clocked at 2.2 GHz or so and boosting up to 2.9GHz on a single core,
with 64GB of RAM. Reads are serviced from an NVMe SSD that is pretty fast, but
not as fast as the SSD in the Mac. I do not think any of the benchmarks are IO-bound
on this machine. The machine is running Ubuntu 18.04 with a 4.15 kernel.

While the results are varied from benchmark to benchmark, I tend to find that
while frawk has good performance overall, it does noticeably better on the newer
hardware running MacOS. The absolute difference in these numbers are probably
due to having a newer CPU with a much higher boost clock, but I am less sure
about the relative performance differences between frawk and tsv-utils. One
contributing factor might be frawk's use of AVX2 driving the clock rate down to
a greater degree on the Broadwell-E CPU in Linux than the more recent CPU on the
Mac configuration. frawk's LLVM backend generally performs better than Cranelift
(this is as expected, as Cranelift does not implement as many optimizations as of
yet), but the performance is usually pretty close.

### Tools

These benchmarks report values for a number of tools, each with slightly
different intended use-cases and strengths. Not all of the tools are set up
well to handle each task, but each makes an appearance in some subset of the
benchmarks. All benchmarks include `frawk`, of course; in this case both the
`use_jemalloc` and `allow_avx2` features were enabled.

* [`mawk`](https://invisible-island.net/mawk/) is an Awk implementation due to
  Mike Brennan. It is focused on the "Awk book" semantics for the language, and
  on efficiency. I used mawk v1.3.4 on MacOS and Linux. The Linux copy is
  compiled from source with -march=native.
* [`gawk`](https://www.gnu.org/software/gawk/manual/) is GNU's Awk. It is
  actively maintained and includes several extensions to the Awk language (such
  as multidimensional arrays). It is the default Awk on many Linux machines. I
  used gawk v5.0.1 on MacOS and gawk v5.1 on Linux. All `gawk` invocations use the
  `-b` flag, which disables multibyte character support. This is a bit unfair to
  xsv, frawk, and tsv-utilities, which all support UTF-8 input and (e.g.) matching
  regular expressions in a UTF-8-aware manner. However, the benchmarks here do not
  make any special use of UTF-8, and gawk performs substantially better with this option
  enabled (sometimes >2x).
* [`xsv`](https://github.com/BurntSushi/xsv) is a command-line utility for
  performing common computations and transformations on CSV data. I used xsv
  v0.13.0 on both platforms
* [`tsv-utils`](https://github.com/eBay/tsv-utils) is a set of command-line
  utilities offering a similar feature-set to xsv, but with a focus on TSV data. I
  used tsv-utils `v2.1.1_linux-x86_64_ldc2` downloaded from the repo for MacOS
  and Linux.

There are plenty of other tools (including other Awks) I could put here (see
the benchmarks referenced on xsv and tsv-utils' pages for more), but I chose a
small set of tools that could execute some subset of the benchmarks
efficiently.

We benchmark on CSV and TSV data. Some utilities are not equipped for all
possible configurations, for example:

* `tsv-utils` is explicitly optimized for TSV and not CSV. We only run it on
  the TSV variants of these datasets, but note that the bundled `csv2tsv` takes
  longer -though not that much longer- than many of the benchmark workloads
  (28.7s for TREE_GRM_ESTN and 14.6s for all_train, with the new version v2.1.1).
* Benchmarks for the `TREE_GRM_ESTN` dataset are not run on CSV input with
  `mawk` or `gawk`, as it contains quoted fields and these utilities do
  not handle CSV escaping out of the box. We still run them on the TSV versions
  of these files using `\t` as a field separator.
* `tsv-utils` and `xsv` are both limited to a specific menu of common functions.
  When those functions cannot easily express the computation in question, we
  leave them out.

Scripts and output for these benchmarks are
[here](https://github.com/ezrosent/frawk/blob/master/info/scripts). They do not
include the raw data or the various binaries in question, but they should be
straightforward to adapt for a given set of installs on another machine. Outputs
from the Linux configuration lives are in the ".2" files.

## Ad-hoc number-crunching
Before we start benchmarking purpose-built tools, I want to emphasize the power
that Awk provides for basic number-crunching tasks. Not only can you efficiently
combine common tasks like filtering, group-by, and arithmetic, but you can build
complex expressions as well. Suppose I wanted to take the scaled, weighted sum
of the first column, and the maximum of column 4 and column 5, when column 8 had
a particular value. In Awk this is a pretty simple script:

```awk
function max(x,y) { return x<y?y:x; }
"GS" == $8 { accum += (0.5*$1+0.5*max($4+0,$5+0))/1000.0 }
END { print accum; }
```

> Note: the `+0`s ensure we are always performing a numeric comparison. In Awk
> this is unnecessary if all instances of column $4 and column $5 are numeric;
> in frawk this is required: max will perform a lexicographic comparison instead
> of a numeric one without the explicit conversion.

But I have to preprocess the data to run Awk on it, as Awk doesn't properly
support quoted CSV fields by default. Supposing I didn't want to do that, the
next thing I'd reach for is a short script in Python:
```python
import csv
import sys

def parse_float(s):
    try:
        return float(s)
    except ValueError:
        return 0.0

accum = 0
with open(sys.argv[1], "r") as f:
    csvr = csv.reader(f)
    for row in csvr:
        if row[7] == "GS":
            f1 = parse_float(row[0])
            f4 = parse_float(row[3])
            f5 = parse_float(row[4])
            accum += ((0.5*f1 + 0.5*max(f4,f5)) / 1000.0)
print(accum)
```
If I was concerned about the running time for the script, I would probably write
the script in Rust instead rather than spending time trying to optimize the
Python.  Here's one I came up with quickly using the
[`csv`](https://github.com/BurntSushi/rust-csv) crate:

```rust
extern crate csv;

use std::fs::File;
use std::io::BufReader;

fn get_field_default_0(record: &csv::StringRecord, field: usize) -> f64 {
    match record.get(field) {
        Some(x) => x.parse().unwrap_or(0.0),
        None => 0.0,
    }
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let filename = &args[1];
    let f = File::open(filename)?;
    let br = BufReader::new(f);
    let mut csvr = csv::Reader::from_reader(br);
    let mut accum = 0f64;
    for record in csvr.records() {
        let record = match record {
            Ok(record) => record,
            Err(err) => {
                eprintln!("failed to parse record: {}", err);
                std::process::abort()
            }
        };
        if record.get(7) == Some("GS") {
            let f1: f64 = get_field_default_0(&record, 0);
            let f4: f64 = get_field_default_0(&record, 3);
            let f5: f64 = get_field_default_0(&record, 4);
            let max = if f4 < f5 { f5 } else { f4 };
            accum += (0.5 * f1 + 0.5 * max) / 1000f64
        }.
    }
    println!("{}", accum);
    Ok(())
}
```

This takes 8.2 seconds to compile for a release build. Depending on the setting
this either is or isn't important. We'll leave it out for now, but keep in mind
that frawk's and python's runtimes include the time it takes to compile a
program.

**MacOS**

| Program | Format | Running Time | Throughput |
| -- | -- | -- | -- |
| Python | CSV | 2m48.7s (2m47.4s + 1.3s) | 53.02 MB/s |
| Rust | CSV | 25.9s (24.8s + 1.1s) | 345.57 MB/s |
| frawk (cranelift) | CSV | 19.9s (18.8s + 1.1s) | 450.13 MB/s |
| frawk (cranelift, parallel) | CSV | 4.9s (23.2s + 1.2s) | 1827.84 MB/s |
| frawk (llvm) | CSV | 19.6s (18.5s + 1.1s) | 457.12 MB/s |
| frawk (llvm, parallel) | CSV | 4.9s (22.9s + 1.2s) | 1842.90 MB/s |

**Linux**

| Program | Format | Running Time | Throughput |
| -- | -- | -- | -- |
| Python | CSV | 2m17.1s (2m15.3s + 1.8s) | 65.23 MB/s |
| Rust | CSV | 29.9s (28.4s + 1.6s) | 299.05 MB/s |
| frawk (cranelift) | CSV | 29.4s (27.3s + 2.2s) | 304.02 MB/s |
| frawk (cranelift, parallel) | CSV | 9.8s (35.5s + 3.5s) | 908.81 MB/s |
| frawk (llvm) | CSV | 28.4s (26.6s + 1.8s) | 315.46 MB/s |
| frawk (llvm, parallel) | CSV | 9.9s (35.2s + 3.8s) | 905.22 MB/s |

frawk is a good deal faster than the other options, particularly on the newer
hardware, or when run in parallel. Now, the Rust script could of course be
optimized substantially (frawk is implemented in Rust, after all).  But if you
need to do exploratory, ad-hoc computations like this, a frawk script is
probably going to be faster than the first few Rust programs you write.

With that said, for many tasks you do not need a full programming language,
and there are purpose-built tools for computing the particular value or
transformation on the dataset. The rest of these benchmarks compare frawk and
Awk to some of those.

## Sum two columns
_Sum columns 4 and 5 from `TREE_GRM_ESTN` and columns 6 and 18 for `all_train`_

Programs:
* Awk: `-F{,\t} {sum1 += ${6,4}; sum2 += ${18,5};} END { print sum1,sum2}`
* frawk: `-i{c,t}sv {sum1 += ${6,4}; sum2 += ${18,5};} END { print sum1,sum2}`
* tsv-utils: `tsv-summarize -H --sum {6,4},{18,5}`

Awk was only run on the TSV version of TREE_GRM_ESTN, as it has quote-escaped
columns, tsv-utils was only run on TSV versions of both files, and xsv was not
included because there is no way to persuade it to _just_ compute a sum.

As can be seen below, frawk in parallel mode was the fastest utility in terms
of wall-time on MacOS, but it is slightly slower on the Linux hardware; on a
per-core basis frawk was faster than mawk and gawk but slower than tsv-utils on
both configurations.

**MacOS**

| Program | Format | Running Time (TREE_GRM_ESTN) | Throughput (TREE_GRM_ESTN) | Running Time (all_train) | Throughput (all_train) |
| -- | -- | -- | -- | -- | -- |
| mawk | CSV | NA | NA | 10.8s (9.8s + 1.1s) | 477.98 MB/s |
| mawk | TSV | 42.0s (40.5s + 1.5s) | 187.81 MB/s | 10.8s (9.8s + 1.0s) | 478.38 MB/s |
| gawk | CSV | NA | NA | 10.4s (9.5s + 0.9s) | 500.19 MB/s |
| gawk | TSV | 14.0s (12.7s + 1.3s) | 562.60 MB/s | 10.4s (9.6s + 0.9s) | 496.07 MB/s |
| tsv-utils | TSV | 5.6s (5.0s + 0.6s) | 1397.24 MB/s | 2.7s (2.2s + 0.4s) | 1953.39 MB/s |
| frawk (llvm) | CSV | 18.0s (16.9s + 1.1s) | 496.91 MB/s | 3.6s (3.0s + 0.6s) | 1436.06 MB/s |
| frawk (llvm) | TSV | 10.0s (9.0s + 1.0s) | 790.27 MB/s | 3.1s (2.5s + 0.6s) | 1650.23 MB/s |
| frawk (llvm, parallel) | CSV | 4.9s (23.1s + 1.3s) | 1822.99 MB/s | 1.8s (4.7s + 0.7s) | 2846.86 MB/s |
| frawk (llvm, parallel) | TSV | 3.4s (12.4s + 1.0s) | 2332.73 MB/s | 1.7s (4.3s + 0.7s) | 3065.98 MB/s |
| frawk (cranelift) | CSV | 18.2s (17.0s + 1.1s) | 492.75 MB/s | 3.7s (3.0s + 0.6s) | 1405.66 MB/s |
| frawk (cranelift) | TSV | 10.3s (9.3s + 1.0s) | 763.36 MB/s | 3.1s (2.5s + 0.6s) | 1652.34 MB/s |
| frawk (cranelift, parallel) | CSV | 4.9s (23.3s + 1.3s) | 1807.89 MB/s | 1.8s (4.7s + 0.6s) | 2896.22 MB/s |
| frawk (cranelift, parallel) | TSV | 3.4s (12.4s + 1.0s) | 2350.80 MB/s | 1.7s (4.4s + 0.7s) | 3071.43 MB/s |

**Linux**

| Program | Format | Running Time (TREE_GRM_ESTN) | Throughput (TREE_GRM_ESTN) | Running Time (all_train) | Throughput (all_train) |
| -- | -- | -- | -- | -- | -- |
| mawk | CSV | NA | NA | 10.4s (9.3s + 1.1s) | 498.07 MB/s |
| mawk | TSV | 54.9s (53.0s + 1.9s) | 143.74 MB/s | 10.5s (9.1s + 1.4s) | 491.41 MB/s |
| gawk | CSV | NA | NA | 11.5s (10.2s + 1.3s) | 451.24 MB/s |
| gawk | TSV | 23.1s (21.7s + 1.4s) | 341.23 MB/s | 11.4s (10.4s + 0.9s) | 456.21 MB/s |
| tsv-utils | TSV | 7.5s (6.3s + 1.2s) | 1047.75 MB/s | 3.6s (2.9s + 0.7s) | 1430.11 MB/s |
| frawk (llvm) | CSV | 26.0s (24.3s + 1.8s) | 343.79 MB/s | 5.4s (4.3s + 1.0s) | 960.04 MB/s |
| frawk (llvm) | TSV | 14.7s (13.2s + 1.5s) | 536.59 MB/s | 4.6s (3.8s + 0.9s) | 1119.42 MB/s |
| frawk (llvm, parallel) | CSV | 9.9s (34.1s + 4.6s) | 899.67 MB/s | 4.0s (9.0s + 2.5s) | 1307.69 MB/s |
| frawk (llvm, parallel) | TSV | 8.1s (21.0s + 3.0s) | 978.75 MB/s | 3.5s (8.3s + 2.0s) | 1462.83 MB/s |
| frawk (cranelift) | CSV | 26.0s (24.4s + 1.6s) | 344.47 MB/s | 5.3s (4.4s + 0.9s) | 973.02 MB/s |
| frawk (cranelift) | TSV | 15.3s (13.8s + 1.4s) | 516.87 MB/s | 4.8s (3.9s + 0.9s) | 1085.85 MB/s |
| frawk (cranelift, parallel) | CSV | 9.8s (33.7s + 4.4s) | 913.45 MB/s | 4.2s (8.7s + 2.2s) | 1233.25 MB/s |
| frawk (cranelift, parallel) | TSV | 8.4s (25.1s + 5.3s) | 943.75 MB/s | 3.8s (8.6s + 2.6s) | 1355.61 MB/s |

## Statistics
_Collect the sum, mean, minimum, maximum, minimum length, maximum length and
standard deviation of a numeric column, collect the maximum, minimum, maximum
length and minimum length of a string column_.

As in the "sum" benchmark, frawk and Awk have the same programs, but with frawk
using the `icsv` and `itsv` options. This benchmark is meant to mirror `xsv`'s
`summarize` command. The xsv invocation is `xsv summarize -s5,6 [-d\t]`. The
Awk program is more involved:

```awk
function min(x,y) { return x<y?x:y; }
function max(x,y) { return x<y?y:x; }
function step_sum(x) { SUM += x; }
function step_stddev(x, k,  xa2) { xa2 = (x - A) * (x - A); A = A + (x-A)/k; Q=Q+((k-1)/k)*xa2; }
NR==1  { h2 = $5; h1 = $6; }
NR > 1 {
    # f2 is numeric, f1 is a string
    f2=$5+0; f2Len = length($5);
    f1=$6; f1Len = length($6);
    if (NR==2) {
        min1=max1=f1;
        min2=max2=f2;
        min1L=max1L=f1Len;
        min2L=max2L=f2Len;
    } else {
        min1 = min(min1, f1)
        min2 = min(min2, f2)
        min1L = min(min1L, f1Len)
        min2L = min(min2L, f2Len)
        max1 = max(max1, f1)
        max2 = max(max2, f2)
        max1L = max(max1L, f1Len)
        max2L = max(max2L, f2Len)
    }
    step_sum(f2);
    step_stddev(f2, NR-1);
}
END {
    N=NR-1 # account for header
    print "field","sum","min","max","min_length","max_length","mean","stddev"
    print h2,SUM,min2,max2,min2L,max2L,(SUM/N), sqrt(Q/(N-1))
    print h1,"NA",min1,max1,min1L,max1L,"NA","NA"
}
```

I had to look up the algorithm for computing a running sample standard deviation
on Wikipedia. If this is all I wanted to compute, I'd probably go with xsv.
This script will not parallelize automatically: to get the correct answer we
will need to (a) manually aggregate the min and max values, as they are not sums
and (b) apply an algorithm (again, from Wikipedia) to combine the
subcomputations of the empirical variance. With that we get the formidable:

```awk
function min(x,y) { return x<y?x:y; }
function max(x,y) { return x<y?y:x; }
function step_stddev(x, k,  xa2) { xa2 = (x - A) * (x - A); A = A + (x-A)/k; Q=Q+((k-1)/k)*xa2; }
BEGIN {
    getline;
    h2 = $5; h1 = $6;
}
{
    # f2 is numeric, f1 is a string
    f2=$5+0; f2Len = length($5);
    f1=$6; f1Len = length($6);
    if (num_records++) {
        min1 = min(min1, f1)
        min2 = min(min2, f2)
        min1L = min(min1L, f1Len)
        min2L = min(min2L, f2Len)
        max1 = max(max1, f1)
        max2 = max(max2, f2)
        max1L = max(max1L, f1Len)
        max2L = max(max2L, f2Len)
    } else {
        min1=max1=f1;
        min2=max2=f2;
        min1L=max1L=f1Len;
        min2L=max2L=f2Len;
    }
    SUM += f2;
    step_stddev(f2, num_records);
}
PREPARE {
    min1M[PID]  = min1
    min2M[PID]  = min2
    min1lM[PID] = min1L
    min2lM[PID] = min2L

    max1M[PID]  = max1
    max2M[PID]  = max2
    max1lM[PID] = max1L
    max2lM[PID] = max2L

    records[PID] = num_records
    sums[PID] = SUM
    # (sub)sample variance.
    m2s[PID] = (Q/(num_records-1))

    if (PID != 1) {
        min1L = min2L = max1L = max2L = 0;
        min2 = max2 = 0
    }
}
END {
      for (pid in records) {
        min1 = min(min1, min1M[pid])
        min2 = min(min2, min2M[pid])
        min1L = min(min1L, min1lM[pid]);
        min2L = min(min2L, min2lM[pid]);

        max1 = max(max1, max1M[pid])
        max2 = max(max2, max2M[pid])
        max1L = max(max1L, max1lM[pid]);
        max2L = max(max2L, max2lM[pid]);
    }

    for (pid in records) {
        nb = records[pid]
        sb = sums[pid]
        mb = sb / nb
        m2b = m2s[pid]
        if (i == 1) {
            na = nb; ma = mb; sa = sb; m2a = m2b;
        } else {
            delta = mb - ma;
            ma = (sa + sb) / (na + nb)
            sa += sums[k]
            m2a = ((na*m2a) + (nb*m2b))/(na+nb) + ((delta*delta*na*nb)/((na+nb)*(na+nb)))
            na += nb
        }

    }

    stddev = sqrt(m2a)

    print "field","sum","min","max","min_length","max_length","mean","stddev"
    print h2,SUM,min2,max2,min2L,max2L,(SUM/num_records), stddev
    print h1,"NA",min1,max1,min1L,max1L,"NA","NA"
}
```

The numeric portion of this benchmark took a little over 10 seconds on MacOS,
and 15 seconds on Linux using tsv-utils, but I am omitting it from the table
because it does not support the given summary statistics on string fields. All
numbers are reported for TREE_GRM_ESTN.

As can be seen below, frawk performed this task more quickly than any of the
other benchmark programs, though the race is pretty close with xsv on the Linux
desktop using a single core and CSV format. Note that frawk is noticeably slower
using Cranelift when compared with LLVM in this benchmark; the other benchmark
programs show performance of the two backends much closer together.

**MacOS**

| Program | Format | Running Time | Throughput |
| -- | -- | -- | -- |
| gawk | TSV | 1m13.3s (1m11.9s + 1.4s) | 107.69 MB/s |
| mawk | TSV | 1m12.6s (1m11.1s + 1.5s) | 108.73 MB/s |
| frawk (cranelift) | CSV | 26.4s (25.3s + 1.1s) | 338.97 MB/s |
| frawk (cranelift) | TSV | 18.8s (17.8s + 1.0s) | 420.73 MB/s |
| frawk (cranelift, parallel) | CSV | 5.3s (29.0s + 1.3s) | 1686.23 MB/s |
| frawk (cranelift, parallel) | TSV | 3.5s (16.6s + 1.0s) | 2223.62 MB/s |
| frawk (llvm) | CSV | 20.0s (18.9s + 1.1s) | 447.41 MB/s |
| frawk (llvm) | TSV | 12.5s (11.5s + 1.0s) | 631.23 MB/s |
| frawk (llvm, parallel) | CSV | 5.1s (23.6s + 1.2s) | 1762.30 MB/s |
| frawk (llvm, parallel) | TSV | 3.6s (16.4s + 1.1s) | 2178.20 MB/s |
| xsv | CSV | 34.7s (33.6s + 1.1s) | 258.14 MB/s |
| xsv | TSV | 32.8s (31.8s + 0.9s) | 240.78 MB/s |

**Linux**

| Program | Format | Running Time | Throughput |
| -- | -- | -- | -- |
| gawk | TSV | 1m14.4s (1m12.8s + 1.6s) | 106.05 MB/s |
| mawk | TSV | 1m23.3s (1m21.4s + 1.9s) | 94.75 MB/s |
| frawk (cranelift) | CSV | 39.0s (37.1s + 2.0s) | 229.09 MB/s |
| frawk (cranelift) | TSV | 28.3s (26.7s + 1.6s) | 278.42 MB/s |
| frawk (cranelift, parallel) | CSV | 9.9s (42.2s + 4.8s) | 900.58 MB/s |
| frawk (cranelift, parallel) | TSV | 8.1s (32.3s + 5.1s) | 979.96 MB/s |
| frawk (llvm) | CSV | 29.1s (27.4s + 1.7s) | 307.05 MB/s |
| frawk (llvm) | TSV | 18.7s (17.0s + 1.7s) | 422.67 MB/s |
| frawk (llvm, parallel) | CSV | 10.0s (35.9s + 3.4s) | 891.33 MB/s |
| frawk (llvm, parallel) | TSV | 7.8s (26.3s + 3.7s) | 1010.97 MB/s |
| xsv | CSV | 34.2s (32.5s + 1.7s) | 261.54 MB/s |
| xsv | TSV | 31.7s (30.3s + 1.4s) | 248.89 MB/s |

## Select
_Select 3 fields from the all_train dataset._

This is a task that all the benchmark programs support. The Awk script looks
like `BEGIN { OFS={",","\t"} { print $1,$2,$8 }`. all_train does not have any
quoted fields, so gawk and mawk can use them with the `-F,` option. As before,
frawk uses the `icsv` and `itsv` options.

The xsv invocation looks like `xsv select [-d'\t'] 1,8,19`, and the tsv-utils
invocation is `tsv-select -f1,8,19`. All output is written to `/dev/null`, so
all times surely underestimate the true running time of such an operation.

> Note: We report parallel numbers for frawk here, but running the `select`
> script in parallel mode does not preserve the original ordering of the rows in
> the input file. While the ordering of some data-sets are not meaningful, the
> comparison is not apples-to-apples.

frawk performs this task slower than tsv-utils and faster than the other
benchmark programs. While the gap between frawk in parallel mode and tsv-utils
is probably in the noise on MacOS (it's still pretty large for the Linux
configuration), tsv-utils unquestionably performs better per core than frawk,
while preserving the input's row ordering.

**MacOS**

| Program | Format | Running Time | Throughput |
| -- | -- | -- | -- |
| gawk | CSV | 7.9s (7.0s + 0.9s) | 657.00 MB/s |
| gawk | TSV | 7.9s (7.1s + 0.9s) | 654.26 MB/s |
| mawk | CSV | 8.4s (7.4s + 1.0s) | 612.83 MB/s |
| mawk | TSV | 8.4s (7.4s + 1.0s) | 616.85 MB/s |
| frawk (cranelift) | CSV | 3.8s (3.3s + 0.7s) | 1358.81 MB/s |
| frawk (cranelift) | TSV | 3.3s (2.8s + 0.7s) | 1569.70 MB/s |
| frawk (cranelift, parallel) | CSV | 1.9s (5.1s + 0.7s) | 2745.72 MB/s |
| frawk (cranelift, parallel) | TSV | 1.7s (4.6s + 0.7s) | 2972.70 MB/s |
| frawk (llvm) | CSV | 3.8s (3.2s + 0.7s) | 1373.59 MB/s |
| frawk (llvm) | TSV | 3.2s (2.7s + 0.7s) | 1598.78 MB/s |
| frawk (llvm, parallel) | CSV | 1.9s (5.1s + 0.7s) | 2678.96 MB/s |
| frawk (llvm, parallel) | TSV | 1.8s (4.6s + 0.7s) | 2930.64 MB/s |
| tsv-utils | TSV | 2.0s (1.6s + 0.4s) | 2589.22 MB/s |
| xsv | CSV | 5.4s (4.8s + 0.6s) | 961.28 MB/s |
| xsv | TSV | 5.3s (4.7s + 0.6s) | 970.65 MB/s |

**Linux**

| Program | Format | Running Time | Throughput |
| -- | -- | -- | -- |
| gawk | CSV | 9.4s (8.3s + 1.1s) | 550.72 MB/s |
| gawk | TSV | 9.4s (8.4s + 1.0s) | 551.01 MB/s |
| mawk | CSV | 8.3s (7.0s + 1.3s) | 621.14 MB/s |
| mawk | TSV | 8.1s (7.0s + 1.1s) | 638.68 MB/s |
| frawk (cranelift) | CSV | 5.5s (4.8s + 1.0s) | 942.39 MB/s |
| frawk (cranelift) | TSV | 4.7s (4.0s + 1.0s) | 1100.39 MB/s |
| frawk (cranelift, parallel) | CSV | 4.0s (9.5s + 2.3s) | 1294.29 MB/s |
| frawk (cranelift, parallel) | TSV | 3.5s (8.5s + 2.0s) | 1476.18 MB/s |
| frawk (llvm) | CSV | 5.3s (4.6s + 1.0s) | 978.36 MB/s |
| frawk (llvm) | TSV | 4.6s (4.0s + 0.9s) | 1115.08 MB/s |
| frawk (llvm, parallel) | CSV | 4.1s (9.4s + 2.5s) | 1259.35 MB/s |
| frawk (llvm, parallel) | TSV | 3.7s (8.8s + 2.2s) | 1406.04 MB/s |
| tsv-utils | TSV | 2.9s (2.0s + 0.9s) | 1756.00 MB/s |
| xsv | CSV | 8.1s (7.2s + 0.9s) | 642.73 MB/s |
| xsv | TSV | 8.0s (7.2s + 0.8s) | 646.66 MB/s |

## Filter
_Print out all records from the `all_train` dataset matching a simple numeric
filter on two of the columns_

The Awk script computing this filter is `$4 > 0.000024 && $16 > 0.3`. Because
`all_train` has no quoted fields, mawk and gawk can both run using the `-F,` and
`-F\t` options. As before, frawk runs the same script with the `icsv` and `itsv`
options. The tsv-utils invocation is `tsv-filter -H --gt 4:0.000025 --gt 16:0.3
./all_train.tsv` (taken from the same benchmark on the tsv-utils repo).  xsv
does not support numeric filters, so it was omitted from this benchmark.

> Note: We report parallel numbers for frawk here, but the same caveat
> highlighted in the `select` benchmark holds here. The comparison is not
> apples-to-apples because frawk will not preserve the input's ordering of the
> rows when run in parallel mode.

Again, frawk is slower than tsv-utils and faster than everything else when
running serially. In parallel, frawk is slightly faster than tsv-utils in terms
of wall time on MacOS, and a still good deal slower in the Linux configuration.

**MacOS**

| Program | Format | Running Time | Throughput |
| -- | -- | -- | -- |
| gawk | CSV | 8.2s (7.3s + 0.9s) | 634.61 MB/s |
| gawk | TSV | 8.2s (7.4s + 0.9s) | 630.52 MB/s |
| mawk | CSV | 10.1s (9.1s + 1.0s) | 511.25 MB/s |
| mawk | TSV | 10.0s (9.0s + 1.0s) | 517.69 MB/s |
| frawk (cranelift) | CSV | 3.7s (3.2s + 0.7s) | 1390.93 MB/s |
| frawk (cranelift) | TSV | 3.1s (2.6s + 0.7s) | 1670.46 MB/s |
| frawk (cranelift, parallel) | CSV | 1.9s (5.2s + 0.8s) | 2684.52 MB/s |
| frawk (cranelift, parallel) | TSV | 1.7s (4.7s + 0.7s) | 2969.29 MB/s |
| frawk (llvm) | CSV | 3.6s (3.1s + 0.7s) | 1422.65 MB/s |
| frawk (llvm) | TSV | 3.1s (2.6s + 0.7s) | 1684.04 MB/s |
| frawk (llvm, parallel) | CSV | 1.9s (5.2s + 0.8s) | 2670.67 MB/s |
| frawk (llvm, parallel) | TSV | 1.8s (4.8s + 0.8s) | 2873.72 MB/s |
| tsv-utils | TSV | 2.3s (1.9s + 0.4s) | 2213.95 MB/s |


**Linux**

| Program | Format | Running Time | Throughput |
| -- | -- | -- | -- |
| gawk | CSV | 9.4s (8.4s + 1.0s) | 551.01 MB/s |
| gawk | TSV | 9.4s (8.4s + 0.9s) | 553.78 MB/s |
| mawk | CSV | 10.1s (8.9s + 1.2s) | 512.67 MB/s |
| mawk | TSV | 10.2s (9.1s + 1.1s) | 507.39 MB/s |
| frawk (cranelift) | CSV | 5.3s (4.7s + 1.0s) | 980.39 MB/s |
| frawk (cranelift) | TSV | 4.8s (3.8s + 1.3s) | 1083.13 MB/s |
| frawk (cranelift, parallel) | CSV | 4.3s (9.5s + 2.8s) | 1204.29 MB/s |
| frawk (cranelift, parallel) | TSV | 3.6s (8.5s + 2.3s) | 1432.09 MB/s |
| frawk (llvm) | CSV | 5.2s (4.5s + 1.0s) | 1002.21 MB/s |
| frawk (llvm) | TSV | 4.5s (4.0s + 0.9s) | 1146.69 MB/s |
| frawk (llvm, parallel) | CSV | 4.4s (9.6s + 2.6s) | 1180.40 MB/s |
| frawk (llvm, parallel) | TSV | 3.9s (8.8s + 2.3s) | 1337.75 MB/s |
| tsv-utils | TSV | 3.4s (2.7s + 0.7s) | 1532.53 MB/s |

## Group By Key
_Print the mean of field 2 grouped by the value in field 6 for TREE_GRM_ESTN_

The tsv-utils command was `tsv-summarize -H  --group-by 6 --mean 2`. The Awk
script, with the usual settings for gawk, mawk, and frawk, reads:

```awk
BEGIN { getline; }
{ N[$6]++; SUM[$6]+=$2; }
END {
    OFS="\t"
    for (k in N) {
        print k, ((SUM[k])/N[k]);
    }
}
```

This is a workload where gawk and frawk are very close in terms of
single-threaded performance. Neither program is nearly as fast as TSV-utils
though. Even if parallel frawk is able to catch up on MacOS, my guess is that
there are serious opportunitities for optimization of arrays here, though part
of the slowdown may be due to tsv-utils's superious handling of 'wide rows'
where we only read a small number of columns.

**MacOS**

| Program | Format | Running Time | Throughput |
| -- | -- | -- | -- |
| gawk | TSV | 14.8s (13.5s + 1.3s) | 534.48 MB/s |
| mawk | TSV | 42.1s (40.6s + 1.5s) | 187.61 MB/s |
| frawk (cranelift) | CSV | 22.1s (21.0s + 1.1s) | 405.69 MB/s |
| frawk (cranelift) | TSV | 15.1s (14.1s + 1.0s) | 523.84 MB/s |
| frawk (cranelift, parallel) | CSV | 5.3s (29.5s + 1.3s) | 1701.62 MB/s |
| frawk (cranelift, parallel) | TSV | 3.5s (16.6s + 1.0s) | 2225.50 MB/s |
| frawk (llvm) | CSV | 21.8s (20.7s + 1.1s) | 410.47 MB/s |
| frawk (llvm) | TSV | 14.8s (13.8s + 1.0s) | 534.66 MB/s |
| frawk (llvm, parallel) | CSV | 5.2s (29.0s + 1.3s) | 1720.27 MB/s |
| frawk (llvm, parallel) | TSV | 3.6s (16.5s + 1.0s) | 2221.74 MB/s |
| tsv-utils | TSV | 4.9s (4.3s + 0.6s) | 1614.16 MB/s |

**Linux**

| Program | Format | Running Time | Throughput |
| -- | -- | -- | -- |
| gawk | TSV | 23.5s (21.9s + 1.6s) | 335.70 MB/s |
| mawk | TSV | 50.6s (48.5s + 2.1s) | 155.86 MB/s |
| frawk (cranelift) | CSV | 32.4s (30.5s + 1.9s) | 276.18 MB/s |
| frawk (cranelift) | TSV | 22.9s (21.1s + 1.8s) | 343.91 MB/s |
| frawk (cranelift, parallel) | CSV | 10.2s (41.5s + 5.5s) | 881.24 MB/s |
| frawk (cranelift, parallel) | TSV | 7.9s (32.0s + 5.0s) | 1001.35 MB/s |
| frawk (llvm) | CSV | 31.8s (30.1s + 1.7s) | 281.08 MB/s |
| frawk (llvm) | TSV | 22.1s (20.5s + 1.6s) | 356.80 MB/s |
| frawk (llvm, parallel) | CSV | 10.4s (42.5s + 6.0s) | 859.72 MB/s |
| frawk (llvm, parallel) | TSV | 7.5s (27.3s + 2.4s) | 1054.19 MB/s |
| tsv-utils | TSV | 5.9s (4.6s + 1.3s) | 1333.27 MB/s |
