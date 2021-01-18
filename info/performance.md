# Performance

## Disclaimer

One of frawk's goals is to be efficient. The abundance of such claims
notwithstanding, I've found it is very hard to precisely state the degree to
which an entire _programming language implementation_ is or is not efficient. I
have no doubt that there are programs in frawk that are slower than equivalent
programs in Rust or C, or even mawk or gawk. In some cases this will be due to
bugs or differences in the quality of the language implementations, in others it
will be due to bugs or differences in the quality of the programs themselves,
while in still others it may be due to the fact that in some nontrivial sense
one language allows you to write the program more efficiently than another.

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
backend, with all optimizations enabled. The parallel invocations launch 3
workers. XSV supports parallelism but I noticed no performance benefit from this
feature without first building an index of the underlying CSV file (a process
which can take longer than the benchmark task itself without amortizing the cost
over multiple runs). Similarly, I do not believe it is  possible to parallelize
the other programs using generic tools like `gnu parallel` without first
partitioning the data-set into multiple sub-files.

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

**MacOS (Newer Hardware)** This is a 16-inch Macbook Pro running MacOS Catalina
10.15.4 with an 8-core i9 clocked at 2.3GHz, boosting up to 4.8GHz. One thing to
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
  on efficiency. I used mawk v1.3.4 on MacOS and v1.3.3 on Linux.
* [`gawk`](https://www.gnu.org/software/gawk/manual/) is GNU's Awk. It is
  actively maintained and includes several extensions to the Awk language (such
  as multidimensional arrays). It is the default Awk on many Linux machines. I
  used gawk v5.0.1 on MacOS and v4.1.4 on Linux.
* [`xsv`](https://github.com/BurntSushi/xsv) is a command-line utility for
  performing common computations and transformations on CSV data. I used xsv
  v0.13.0 on both platforms
* [`tsv-utils`](https://github.com/eBay/tsv-utils) is a set of command-line
  utilities offering a similar feature-set to xsv, but with a focus on TSV data. I
  used tsv-utils `v2.1.1_linux-x86_64_ldc2` downloaded from the repo for MacOS
  and Linux.

There are plenty of other tools I could put here (see the benchmarks referenced
on xsv and tsv-utils' pages for more), but I chose a small set of tools that
could execute some subset of the benchmarks efficiently.

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
| Program | Running Time | Throughput |
| -- | -- | -- |
| Python | CSV | 2m49.5s (2m48.1s + 1.4s) | 52.78 MB/s |
| Rust | CSV | 28.2s (27.1s + 1.1s) | 317.06 MB/s |
| frawk (cranelift) | CSV | 21.7s (20.6s + 1.1s) | 411.95 MB/s |
| frawk (cranelift, parallel) | CSV | 6.3s (24.1s + 1.2s) | 1414.30 MB/s |
| frawk (llvm) | CSV | 21.2s (20.1s + 1.1s) | 421.74 MB/s |
| frawk (llvm, parallel) | CSV | 6.1s (23.2s + 1.2s) | 1462.87 MB/s |

**Linux**
| Program | Running Time | Throughput |
| -- | -- | -- |
| Python | CSV | 2m18.6s (2m16.6s + 2.0s) | 64.56 MB/s |
| Rust | CSV | 30.5s (28.6s + 1.9s) | 293.63 MB/s |
| frawk (cranelift) | CSV | 29.8s (28.0s + 1.8s) | 300.59 MB/s |
| frawk (cranelift, parallel) | CSV | 9.3s (34.3s + 2.8s) | 959.29 MB/s |
| frawk (llvm) | CSV | 29.3s (27.4s + 1.9s) | 305.69 MB/s |
| frawk (llvm, parallel) | CSV | 9.3s (34.0s + 3.1s) | 961.67 MB/s |

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
| Program | Running Time (TREE_GRM_ESTN) | Throughput (TREE_GRM_ESTN) | Running Time (all_train) | Throughput (all_train) |
| -- | -- | -- | -- | -- |
| mawk | CSV | NA | NA | 11.0s (10.0s + 1.1s) | 469.32 MB/s |
| mawk | TSV | 42.8s (41.3s + 1.5s) | 184.30 MB/s | 11.0s (9.9s + 1.1s) | 471.62 MB/s |
| gawk | CSV | NA | NA | 24.9s (24.0s + 0.9s) | 207.69 MB/s |
| gawk | TSV | 23.8s (22.5s + 1.3s) | 331.58 MB/s | 25.0s (24.1s + 0.9s) | 207.15 MB/s |
| tsv-utils | TSV | 5.6s (4.9s + 0.6s) | 1417.83 MB/s | 2.7s (2.2s + 0.4s) | 1952.65 MB/s |
| frawk (llvm) | CSV | 19.3s (18.2s + 1.1s) | 462.53 MB/s | 7.8s (7.2s + 0.7s) | 661.11 MB/s |
| frawk (llvm) | TSV | 15.9s (14.9s + 1.0s) | 496.67 MB/s | 7.5s (6.9s + 0.6s) | 685.89 MB/s |
| frawk (llvm, parallel) | CSV | 5.5s (20.6s + 1.2s) | 1635.66 MB/s | 2.3s (8.5s + 0.7s) | 2228.24 MB/s |
| frawk (llvm, parallel) | TSV | 4.9s (18.4s + 1.1s) | 1613.50 MB/s | 2.4s (8.9s + 0.7s) | 2132.80 MB/s |
| frawk (cranelift) | CSV | 19.6s (18.5s + 1.1s) | 456.26 MB/s | 7.9s (7.2s + 0.7s) | 657.41 MB/s |
| frawk (cranelift) | TSV | 16.3s (15.2s + 1.0s) | 485.61 MB/s | 7.6s (7.0s + 0.7s) | 679.32 MB/s |
| frawk (cranelift, parallel) | CSV | 5.5s (20.8s + 1.2s) | 1624.96 MB/s | 2.4s (8.7s + 0.7s) | 2198.91 MB/s |
| frawk (cranelift, parallel) | TSV | 5.0s (19.0s + 1.1s) | 1572.35 MB/s | 2.4s (8.7s + 0.7s) | 2196.11 MB/s |

**Linux**
| Program | Running Time (TREE_GRM_ESTN) | Throughput (TREE_GRM_ESTN) | Running Time (all_train) | Throughput (all_train) |
| -- | -- | -- | -- | -- |
| mawk | CSV | NA | NA | 10.3s (8.9s + 1.4s) | 505.07 MB/s |
| mawk | TSV | 51.0s (49.0s + 2.0s) | 154.84 MB/s | 10.3s (9.0s + 1.3s) | 502.03 MB/s |
| gawk | CSV | NA | NA | 1m9.6s (1m8.6s + 1.0s) | 74.39 MB/s |
| gawk | TSV | 1m3.6s (1m2.0s + 1.6s) | 124.13 MB/s | 1m9.4s (1m8.3s + 1.1s) | 74.64 MB/s |
| tsv-utils | TSV | 7.6s (6.3s + 1.3s) | 1036.19 MB/s | 3.8s (2.9s + 0.9s) | 1360.96 MB/s |
| frawk (llvm) | CSV | 26.1s (24.1s + 2.1s) | 342.30 MB/s | 8.2s (7.4s + 0.8s) | 633.60 MB/s |
| frawk (llvm) | TSV | 19.6s (18.0s + 1.6s) | 402.45 MB/s | 8.1s (7.0s + 1.1s) | 642.73 MB/s |
| frawk (llvm, parallel) | CSV | 10.0s (33.7s + 5.0s) | 891.96 MB/s | 4.6s (13.2s + 3.6s) | 1121.84 MB/s |
| frawk (llvm, parallel) | TSV | 6.9s (24.7s + 2.8s) | 1138.11 MB/s | 4.0s (12.7s + 2.7s) | 1283.70 MB/s |
| frawk (cranelift) | CSV | 26.1s (24.5s + 1.6s) | 342.97 MB/s | 8.4s (7.1s + 1.2s) | 619.06 MB/s |
| frawk (cranelift) | TSV | 20.3s (18.7s + 1.6s) | 388.96 MB/s | 8.1s (6.9s + 1.2s) | 637.74 MB/s |
| frawk (cranelift, parallel) | CSV | 10.0s (34.3s + 4.6s) | 896.60 MB/s | 4.3s (12.9s + 3.2s) | 1207.94 MB/s |
| frawk (cranelift, parallel) | TSV | 6.9s (24.8s + 2.6s) | 1151.05 MB/s | 4.0s (12.5s + 2.7s) | 1297.53 MB/s |

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
    n_pids = length(records)
    for (i=1; i<=n_pids; i++) {
        min1 = min(min1, min1M[i])
        min2 = min(min2, min2M[i])
        min1L = min(min1L, min1lM[i]);
        min2L = min(min2L, min2lM[i]);

        max1 = max(max1, max1M[i])
        max2 = max(max2, max2M[i])
        max1L = max(max1L, max1lM[i]);
        max2L = max(max2L, max2lM[i]);
    }

    for (i=1; i<=n_pids; i++) {
        nb = records[i]
        sb = sums[i]
        mb = sb / nb
        m2b = m2s[i]
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
| Program | Running Time | Throughput |
| -- | -- | -- |
| frawk (cranelift) | CSV | 26.6s (25.5s + 1.1s) | 335.78 MB/s |
| frawk (cranelift) | TSV | 23.1s (22.1s + 1.0s) | 341.45 MB/s |
| frawk (cranelift, parallel) | CSV | 7.0s (26.7s + 1.2s) | 1276.46 MB/s |
| frawk (cranelift, parallel) | TSV | 6.4s (24.0s + 1.2s) | 1237.90 MB/s |
| frawk (llvm) | CSV | 21.2s (20.1s + 1.1s) | 421.87 MB/s |
| frawk (llvm) | TSV | 17.7s (16.7s + 1.0s) | 446.23 MB/s |
| frawk (llvm, parallel) | CSV | 6.1s (22.8s + 1.2s) | 1464.06 MB/s |
| frawk (llvm, parallel) | TSV | 5.3s (19.9s + 1.1s) | 1477.56 MB/s |
| gawk | TSV | 1m34.9s (1m33.4s + 1.5s) | 83.15 MB/s |
| mawk | TSV | 1m13.7s (1m12.2s + 1.5s) | 107.03 MB/s |
| tsv-utils | TSV | 10.0s (9.3s + 0.6s) | 792.81 MB/s |
| xsv | CSV | 35.0s (33.9s + 1.1s) | 255.41 MB/s |
| xsv | TSV | 33.0s (32.0s + 1.0s) | 239.23 MB/s |

**Linux**
| Program | Running Time | Throughput |
| -- | -- | -- |
| frawk (cranelift) | CSV | 37.1s (35.5s + 1.6s) | 241.36 MB/s |
| frawk (cranelift) | TSV | 30.7s (29.2s + 1.5s) | 256.90 MB/s |
| frawk (cranelift, parallel) | CSV | 9.6s (36.0s + 2.3s) | 930.65 MB/s |
| frawk (cranelift, parallel) | TSV | 7.9s (29.3s + 2.3s) | 993.03 MB/s |
| frawk (llvm) | CSV | 29.6s (27.4s + 2.1s) | 302.69 MB/s |
| frawk (llvm) | TSV | 22.5s (21.1s + 1.4s) | 351.16 MB/s |
| frawk (llvm, parallel) | CSV | 9.5s (33.9s + 3.5s) | 940.14 MB/s |
| frawk (llvm, parallel) | TSV | 7.0s (25.6s + 1.9s) | 1131.90 MB/s |
| gawk | TSV | 2m10.3s (2m8.7s + 1.6s) | 60.56 MB/s |
| mawk | TSV | 1m16.2s (1m14.2s + 1.9s) | 103.62 MB/s |
| tsv-utils | TSV | 15.2s (13.9s + 1.3s) | 519.94 MB/s |
| xsv | CSV | 34.9s (33.4s + 1.5s) | 256.52 MB/s |
| xsv | TSV | 32.3s (30.7s + 1.6s) | 244.54 MB/s |

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
| Program | Running Time | Throughput |
| -- | -- | -- |
| frawk (cranelift) | CSV | 4.2s (5.0s + 1.0s) | 1232.67 MB/s |
| frawk (cranelift) | TSV | 4.1s (4.9s + 1.0s) | 1267.67 MB/s |
| frawk (cranelift, parallel) | CSV | 2.1s (8.6s + 1.1s) | 2469.45 MB/s |
| frawk (cranelift, parallel) | TSV | 1.8s (7.5s + 1.0s) | 2845.29 MB/s |
| frawk (llvm) | CSV | 4.1s (5.0s + 1.1s) | 1252.34 MB/s |
| frawk (llvm) | TSV | 4.0s (4.8s + 1.0s) | 1292.35 MB/s |
| frawk (llvm, parallel) | CSV | 2.1s (8.7s + 1.1s) | 2422.09 MB/s |
| frawk (llvm, parallel) | TSV | 1.8s (7.5s + 1.0s) | 2828.20 MB/s |
| gawk | CSV | 23.2s (22.4s + 0.9s) | 222.74 MB/s |
| gawk | TSV | 23.2s (22.4s + 0.9s) | 222.76 MB/s |
| mawk | CSV | 8.5s (7.5s + 1.0s) | 607.37 MB/s |
| mawk | TSV | 8.6s (7.5s + 1.0s) | 604.67 MB/s |
| tsv-utils | TSV | 1.9s (1.5s + 0.4s) | 2669.30 MB/s |
| xsv | CSV | 5.5s (4.8s + 0.6s) | 942.39 MB/s |
| xsv | TSV | 5.4s (4.8s + 0.6s) | 957.55 MB/s |

**Linux**
| Program | Running Time | Throughput |
| -- | -- | -- |
| frawk (cranelift) | CSV | 5.6s (5.7s + 1.7s) | 919.14 MB/s |
| frawk (cranelift) | TSV | 5.5s (5.5s + 1.8s) | 935.24 MB/s |
| frawk (cranelift, parallel) | CSV | 5.2s (13.5s + 5.1s) | 986.56 MB/s |
| frawk (cranelift, parallel) | TSV | 4.9s (12.6s + 5.0s) | 1064.86 MB/s |
| frawk (llvm) | CSV | 5.6s (5.5s + 1.9s) | 921.10 MB/s |
| frawk (llvm) | TSV | 5.3s (5.3s + 1.8s) | 969.38 MB/s |
| frawk (llvm, parallel) | CSV | 5.1s (12.8s + 5.5s) | 1019.58 MB/s |
| frawk (llvm, parallel) | TSV | 4.9s (12.8s + 5.4s) | 1058.34 MB/s |
| gawk | CSV | 1m11.4s (1m10.1s + 1.3s) | 72.51 MB/s |
| gawk | TSV | 1m11.6s (1m10.4s + 1.2s) | 72.36 MB/s |
| mawk | CSV | 8.1s (6.6s + 1.5s) | 637.82 MB/s |
| mawk | TSV | 8.1s (6.8s + 1.3s) | 639.71 MB/s |
| tsv-utils | TSV | 2.9s (2.0s + 0.9s) | 1811.91 MB/s |
| xsv | CSV | 8.3s (7.3s + 1.1s) | 623.08 MB/s |
| xsv | TSV | 8.2s (7.2s + 1.0s) | 630.98 MB/s |

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
| Program | Running Time | Throughput |
| -- | -- | -- |
| frawk (cranelift) | CSV | 7.5s (9.2s + 1.3s) | 688.80 MB/s |
| frawk (cranelift) | TSV | 7.3s (9.0s + 1.3s) | 709.57 MB/s |
| frawk (cranelift, parallel) | CSV | 2.2s (9.3s + 0.9s) | 2399.65 MB/s |
| frawk (cranelift, parallel) | TSV | 2.2s (9.4s + 1.0s) | 2379.80 MB/s |
| frawk (llvm) | CSV | 7.5s (9.1s + 1.3s) | 691.56 MB/s |
| frawk (llvm) | TSV | 7.3s (9.0s + 1.3s) | 710.64 MB/s |
| frawk (llvm, parallel) | CSV | 2.2s (9.3s + 1.0s) | 2383.08 MB/s |
| frawk (llvm, parallel) | TSV | 2.1s (8.9s + 0.9s) | 2490.83 MB/s |
| gawk | CSV | 16.4s (15.5s + 0.9s) | 316.36 MB/s |
| gawk | TSV | 16.4s (15.6s + 0.9s) | 314.95 MB/s |
| mawk | CSV | 10.3s (9.2s + 1.1s) | 502.61 MB/s |
| mawk | TSV | 10.2s (9.2s + 1.0s) | 505.76 MB/s |
| tsv-utils | TSV | 2.3s (1.9s + 0.4s) | 2240.78 MB/s |

**Linux**
| Program | Running Time | Throughput |
| -- | -- | -- |
| frawk (cranelift) | CSV | 8.1s (8.6s + 2.3s) | 637.74 MB/s |
| frawk (cranelift) | TSV | 7.9s (8.2s + 2.4s) | 651.54 MB/s |
| frawk (cranelift, parallel) | CSV | 4.7s (14.7s + 4.5s) | 1093.19 MB/s |
| frawk (cranelift, parallel) | TSV | 4.1s (13.5s + 3.9s) | 1252.34 MB/s |
| frawk (llvm) | CSV | 8.1s (8.6s + 2.4s) | 638.37 MB/s |
| frawk (llvm) | TSV | 8.0s (8.3s + 2.5s) | 645.21 MB/s |
| frawk (llvm, parallel) | CSV | 4.3s (13.5s + 4.2s) | 1199.82 MB/s |
| frawk (llvm, parallel) | TSV | 4.1s (13.9s + 3.6s) | 1256.29 MB/s |
| gawk | CSV | 42.0s (40.8s + 1.3s) | 123.21 MB/s |
| gawk | TSV | 41.9s (40.4s + 1.4s) | 123.73 MB/s |
| mawk | CSV | 9.7s (8.5s + 1.2s) | 533.53 MB/s |
| mawk | TSV | 9.9s (8.7s + 1.3s) | 521.39 MB/s |
| tsv-utils | TSV | 3.5s (2.7s + 0.8s) | 1484.64 MB/s |

## Group By Key
_Print the mean of field 2 grouped by the value in field 6 for TREE_GRM_ESTN_

The tsv-utils command was `tsv-summarize -H  --group-by 6 --mean 2`. The Awk
script, with the usual settings for gawk, mawk, and frawk, reads:

```awk
NR > 1 { N[$6]++; SUM[$6]+=$2; }
END {
    OFS="\t"
    for (k in N) {
        print k, ((SUM[k])/N[k]);
    }
}
```

Once again, tsv-utils is substantially faster than a single-threaded frawk and
parallel frawk on Linux, but is slower than frawk in parallel mode on MacOS.
frawk, in all configurations, is faster at this task than mawk or gawk.

**MacOS**
| Program | Running Time | Throughput |
| -- | -- | -- |
| frawk (cranelift) | CSV | 22.8s (21.7s + 1.1s) | 392.07 MB/s |
| frawk (cranelift) | TSV | 19.8s (18.8s + 1.0s) | 399.29 MB/s |
| frawk (cranelift, parallel) | CSV | 6.7s (25.6s + 1.2s) | 1331.56 MB/s |
| frawk (cranelift, parallel) | TSV | 6.3s (23.6s + 1.2s) | 1257.63 MB/s |
| frawk (llvm) | CSV | 22.2s (21.1s + 1.1s) | 402.73 MB/s |
| frawk (llvm) | TSV | 19.2s (18.2s + 1.0s) | 411.11 MB/s |
| frawk (llvm, parallel) | CSV | 6.5s (24.6s + 1.2s) | 1381.75 MB/s |
| frawk (llvm, parallel) | TSV | 6.1s (22.9s + 1.2s) | 1295.62 MB/s |
| gawk | TSV | 27.8s (26.5s + 1.3s) | 283.81 MB/s |
| mawk | TSV | 44.0s (42.3s + 1.8s) | 179.20 MB/s |
| tsv-utils | TSV | 4.9s (4.3s + 0.6s) | 1608.90 MB/s |

**Linux**
| Program | Running Time | Throughput |
| -- | -- | -- |
| frawk (cranelift) | CSV | 29.7s (28.1s + 1.6s) | 300.81 MB/s |
| frawk (cranelift) | TSV | 24.5s (23.0s + 1.5s) | 322.00 MB/s |
| frawk (cranelift, parallel) | CSV | 9.3s (34.7s + 2.6s) | 957.45 MB/s |
| frawk (cranelift, parallel) | TSV | 7.2s (27.1s + 1.8s) | 1091.51 MB/s |
| frawk (llvm) | CSV | 29.3s (27.8s + 1.5s) | 305.39 MB/s |
| frawk (llvm) | TSV | 24.2s (22.6s + 1.6s) | 326.10 MB/s |
| frawk (llvm, parallel) | CSV | 9.4s (34.7s + 2.7s) | 949.82 MB/s |
| frawk (llvm, parallel) | TSV | 7.3s (26.9s + 2.3s) | 1074.71 MB/s |
| gawk | TSV | 1m11.6s (1m10.0s + 1.6s) | 110.19 MB/s |
| mawk | TSV | 47.3s (45.5s + 1.8s) | 166.87 MB/s |
| tsv-utils | TSV | 5.8s (4.6s + 1.2s) | 1363.21 MB/s |
