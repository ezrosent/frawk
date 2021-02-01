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
backend, with all optimizations enabled. The parallel invocations launch 3
workers (note this is not optimal for some benchmarks, but I don't want to tune
the frawk configuration aggressively just to improve benchmark numbers). XSV
supports parallelism but I noticed no performance benefit from this feature
without first building an index of the underlying CSV file (a process which can
take longer than the benchmark task itself without amortizing the cost over
multiple runs). Similarly, I do not believe it is  possible to parallelize the
other programs using generic tools like `gnu parallel` without first
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
| Python | CSV | 2m49.4s (2m47.9s + 1.4s) | 52.82 MB/s |
| Rust | CSV | 28.4s (27.3s + 1.1s) | 314.80 MB/s |
| frawk (cranelift) | CSV | 21.2s (20.1s + 1.1s) | 421.70 MB/s |
| frawk (cranelift, parallel) | CSV | 6.1s (23.2s + 1.2s) | 1464.06 MB/s |
| frawk (llvm) | CSV | 20.9s (19.7s + 1.1s) | 428.48 MB/s |
| frawk (llvm, parallel) | CSV | 6.0s (22.7s + 1.2s) | 1490.41 MB/s |

**Linux**
| Program | Format | Running Time | Throughput |
| -- | -- | -- | -- |
| Python | CSV | 2m16.2s (2m14.2s + 2.0s) | 65.67 MB/s |
| Rust | CSV | 30.1s (28.4s + 1.6s) | 297.66 MB/s |
| frawk (cranelift) | CSV | 27.9s (26.2s + 1.7s) | 320.12 MB/s |
| frawk (cranelift, parallel) | CSV | 9.4s (34.5s + 3.0s) | 951.03 MB/s |
| frawk (llvm) | CSV | 27.9s (25.8s + 2.0s) | 321.20 MB/s |
| frawk (llvm, parallel) | CSV | 9.5s (33.9s + 3.5s) | 943.21 MB/s |

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
| mawk | CSV | NA | NA | 11.1s (10.1s + 1.1s) | 465.31 MB/s |
| mawk | TSV | 42.7s (41.2s + 1.5s) | 184.80 MB/s | 11.1s (10.0s + 1.1s) | 468.59 MB/s |
| gawk | CSV | NA | NA | 10.3s (9.3s + 0.9s) | 504.67 MB/s |
| gawk | TSV | 14.1s (12.8s + 1.3s) | 557.75 MB/s | 10.3s (9.4s + 0.9s) | 504.82 MB/s |
| tsv-utils | TSV | 5.7s (5.0s + 0.7s) | 1394.03 MB/s | 2.6s (2.2s + 0.4s) | 1954.13 MB/s |
| frawk (llvm) | CSV | 19.3s (18.2s + 1.1s) | 462.34 MB/s | 7.9s (7.2s + 0.7s) | 658.92 MB/s |
| frawk (llvm) | TSV | 15.7s (14.7s + 1.0s) | 503.84 MB/s | 7.7s (7.0s + 0.7s) | 671.22 MB/s |
| frawk (llvm, parallel) | CSV | 5.4s (20.5s + 1.2s) | 1642.57 MB/s | 2.3s (8.6s + 0.7s) | 2210.17 MB/s |
| frawk (llvm, parallel) | TSV | 4.8s (18.1s + 1.1s) | 1635.91 MB/s | 2.4s (8.9s + 0.7s) | 2142.51 MB/s |
| frawk (cranelift) | CSV | 19.5s (18.3s + 1.1s) | 459.38 MB/s | 8.0s (7.3s + 0.7s) | 647.47 MB/s |
| frawk (cranelift) | TSV | 15.9s (14.9s + 1.0s) | 497.42 MB/s | 7.8s (7.1s + 0.7s) | 667.75 MB/s |
| frawk (cranelift, parallel) | CSV | 5.5s (20.7s + 1.2s) | 1633.87 MB/s | 2.4s (8.9s + 0.7s) | 2158.58 MB/s |
| frawk (cranelift, parallel) | TSV | 4.9s (18.3s + 1.1s) | 1626.13 MB/s | 2.4s (8.9s + 0.7s) | 2157.68 MB/s |

**Linux**
| Program | Format | Running Time (TREE_GRM_ESTN) | Throughput (TREE_GRM_ESTN) | Running Time (all_train) | Throughput (all_train) |
| -- | -- | -- | -- | -- | -- |
| mawk | CSV | NA | NA | 10.3s (9.1s + 1.1s) | 505.02 MB/s |
| mawk | TSV | 54.8s (53.0s + 1.7s) | 144.12 MB/s | 10.7s (9.3s + 1.4s) | 482.75 MB/s |
| gawk | CSV | NA | NA | 11.4s (10.2s + 1.2s) | 452.46 MB/s |
| gawk | TSV | 22.9s (21.3s + 1.7s) | 344.19 MB/s | 11.5s (10.2s + 1.3s) | 450.10 MB/s |
| tsv-utils | TSV | 7.5s (6.5s + 1.1s) | 1048.03 MB/s | 3.9s (2.9s + 1.0s) | 1329.17 MB/s |
| frawk (llvm) | CSV | 25.1s (23.6s + 1.5s) | 355.78 MB/s | 8.2s (7.0s + 1.1s) | 632.75 MB/s |
| frawk (llvm) | TSV | 19.6s (18.1s + 1.4s) | 403.21 MB/s | 8.1s (6.8s + 1.3s) | 638.37 MB/s |
| frawk (llvm, parallel) | CSV | 9.8s (32.7s + 4.9s) | 914.76 MB/s | 4.6s (13.9s + 3.2s) | 1135.12 MB/s |
| frawk (llvm, parallel) | TSV | 6.8s (24.6s + 2.5s) | 1154.25 MB/s | 4.1s (12.7s + 2.8s) | 1261.19 MB/s |
| frawk (cranelift) | CSV | 25.5s (23.9s + 1.7s) | 350.21 MB/s | 8.1s (7.1s + 1.0s) | 637.19 MB/s |
| frawk (cranelift) | TSV | 20.0s (18.2s + 1.8s) | 393.75 MB/s | 7.9s (7.1s + 0.9s) | 651.70 MB/s |
| frawk (cranelift, parallel) | CSV | 9.7s (33.2s + 4.6s) | 921.83 MB/s | 4.5s (13.7s + 3.2s) | 1139.12 MB/s |
| frawk (cranelift, parallel) | TSV | 6.9s (24.9s + 2.4s) | 1150.05 MB/s | 4.0s (12.8s + 2.6s) | 1301.44 MB/s |

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
| Program | Format | Running Time | Throughput |
| -- | -- | -- | -- |
| gawk | TSV | 1m12.8s (1m11.4s + 1.4s) | 108.39 MB/s |
| mawk | TSV | 1m14.2s (1m12.6s + 1.6s) | 106.33 MB/s |
| frawk (cranelift) | CSV | 27.4s (26.2s + 1.2s) | 326.99 MB/s |
| frawk (cranelift) | TSV | 23.9s (22.9s + 1.0s) | 330.57 MB/s |
| frawk (cranelift, parallel) | CSV | 7.2s (27.5s + 1.3s) | 1241.04 MB/s |
| frawk (cranelift, parallel) | TSV | 6.4s (24.0s + 1.3s) | 1238.10 MB/s |
| frawk (llvm) | CSV | 21.7s (20.5s + 1.2s) | 412.57 MB/s |
| frawk (llvm) | TSV | 17.8s (16.8s + 1.0s) | 442.63 MB/s |
| frawk (llvm, parallel) | CSV | 6.2s (23.3s + 1.2s) | 1431.96 MB/s |
| frawk (llvm, parallel) | TSV | 5.5s (20.4s + 1.1s) | 1441.13 MB/s |
| tsv-utils | TSV | 10.0s (9.3s + 0.6s) | 792.49 MB/s |
| xsv | CSV | 34.9s (33.8s + 1.1s) | 256.24 MB/s |
| xsv | TSV | 33.4s (32.5s + 1.0s) | 236.01 MB/s |

**Linux**
| Program | Format | Running Time | Throughput |
| -- | -- | -- | -- |
| gawk | TSV | 1m14.3s (1m12.7s + 1.6s) | 106.23 MB/s |
| mawk | TSV | 1m25.2s (1m23.4s + 1.8s) | 92.58 MB/s |
| frawk (cranelift) | CSV | 38.0s (36.3s + 1.7s) | 235.42 MB/s |
| frawk (cranelift) | TSV | 32.4s (30.8s + 1.5s) | 243.73 MB/s |
| frawk (cranelift, parallel) | CSV | 9.6s (35.9s + 2.3s) | 934.54 MB/s |
| frawk (cranelift, parallel) | TSV | 8.0s (29.7s + 2.3s) | 981.06 MB/s |
| frawk (llvm) | CSV | 28.8s (27.1s + 1.8s) | 310.43 MB/s |
| frawk (llvm) | TSV | 23.3s (21.8s + 1.5s) | 339.08 MB/s |
| frawk (llvm, parallel) | CSV | 9.6s (34.8s + 2.8s) | 936.50 MB/s |
| frawk (llvm, parallel) | TSV | 7.0s (25.6s + 2.0s) | 1127.54 MB/s |
| tsv-utils | TSV | 15.2s (14.0s + 1.1s) | 520.90 MB/s |
| xsv | CSV | 34.1s (32.5s + 1.6s) | 262.28 MB/s |
| xsv | TSV | 31.8s (30.6s + 1.2s) | 248.09 MB/s |

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
| gawk | CSV | 8.8s (7.8s + 1.0s) | 589.33 MB/s |
| gawk | TSV | 9.0s (8.0s + 1.0s) | 572.96 MB/s |
| mawk | CSV | 9.7s (8.5s + 1.2s) | 535.35 MB/s |
| mawk | TSV | 9.4s (8.3s + 1.1s) | 548.56 MB/s |
| frawk (cranelift) | CSV | 3.9s (3.3s + 0.7s) | 1337.75 MB/s |
| frawk (cranelift) | TSV | 3.7s (3.2s + 0.7s) | 1384.24 MB/s |
| frawk (cranelift, parallel) | CSV | 2.2s (7.9s + 0.9s) | 2361.35 MB/s |
| frawk (cranelift, parallel) | TSV | 2.0s (7.0s + 0.9s) | 2629.98 MB/s |
| frawk (llvm) | CSV | 4.1s (3.4s + 0.8s) | 1254.47 MB/s |
| frawk (llvm) | TSV | 4.1s (3.5s + 0.8s) | 1260.27 MB/s |
| frawk (llvm, parallel) | CSV | 2.2s (7.8s + 0.9s) | 2390.78 MB/s |
| frawk (llvm, parallel) | TSV | 2.1s (7.3s + 1.0s) | 2513.80 MB/s |
| tsv-utils | TSV | 2.0s (1.6s + 0.5s) | 2526.07 MB/s |
| xsv | CSV | 6.2s (5.5s + 0.7s) | 828.55 MB/s |
| xsv | TSV | 6.2s (5.4s + 0.7s) | 838.34 MB/s |

**Linux**
| Program | Format | Running Time | Throughput |
| -- | -- | -- | -- |
| gawk | CSV | 9.3s (8.4s + 1.0s) | 555.03 MB/s |
| gawk | TSV | 9.5s (8.4s + 1.1s) | 544.24 MB/s |
| mawk | CSV | 8.3s (6.8s + 1.4s) | 627.54 MB/s |
| mawk | TSV | 8.2s (6.8s + 1.4s) | 629.14 MB/s |
| frawk (cranelift) | CSV | 5.2s (4.4s + 1.1s) | 987.87 MB/s |
| frawk (cranelift) | TSV | 5.1s (4.3s + 1.1s) | 1008.66 MB/s |
| frawk (cranelift, parallel) | CSV | 5.4s (12.9s + 4.7s) | 958.08 MB/s |
| frawk (cranelift, parallel) | TSV | 5.0s (12.4s + 4.4s) | 1033.62 MB/s |
| frawk (llvm) | CSV | 5.1s (4.3s + 1.1s) | 1016.97 MB/s |
| frawk (llvm) | TSV | 5.1s (4.3s + 1.0s) | 1022.40 MB/s |
| frawk (llvm, parallel) | CSV | 5.4s (12.6s + 4.8s) | 957.37 MB/s |
| frawk (llvm, parallel) | TSV | 4.9s (12.2s + 4.4s) | 1049.97 MB/s |
| tsv-utils | TSV | 2.9s (2.0s + 0.9s) | 1815.72 MB/s |
| xsv | CSV | 8.2s (7.1s + 1.1s) | 629.14 MB/s |
| xsv | TSV | 8.1s (7.1s + 1.0s) | 638.52 MB/s |

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
| gawk | CSV | 9.1s (8.1s + 1.0s) | 568.37 MB/s |
| gawk | TSV | 9.0s (8.0s + 1.0s) | 575.06 MB/s |
| mawk | CSV | 11.4s (10.2s + 1.2s) | 454.09 MB/s |
| mawk | TSV | 11.5s (10.3s + 1.2s) | 451.24 MB/s |
| frawk (cranelift) | CSV | 7.3s (6.8s + 0.8s) | 707.44 MB/s |
| frawk (cranelift) | TSV | 7.1s (6.5s + 0.8s) | 732.76 MB/s |
| frawk (cranelift, parallel) | CSV | 2.2s (8.3s + 0.8s) | 2328.43 MB/s |
| frawk (cranelift, parallel) | TSV | 2.2s (8.3s + 0.8s) | 2303.57 MB/s |
| frawk (llvm) | CSV | 7.3s (6.8s + 0.8s) | 708.02 MB/s |
| frawk (llvm) | TSV | 7.1s (6.5s + 0.8s) | 732.35 MB/s |
| frawk (llvm, parallel) | CSV | 2.2s (7.9s + 0.8s) | 2351.70 MB/s |
| frawk (llvm, parallel) | TSV | 2.2s (8.1s + 0.8s) | 2351.70 MB/s |
| tsv-utils | TSV | 2.5s (2.0s + 0.5s) | 2062.30 MB/s |

**Linux**
| Program | Format | Running Time | Throughput |
| -- | -- | -- | -- |
| gawk | CSV | 9.4s (8.2s + 1.3s) | 549.55 MB/s |
| gawk | TSV | 9.5s (8.2s + 1.3s) | 544.75 MB/s |
| mawk | CSV | 10.2s (9.0s + 1.2s) | 507.59 MB/s |
| mawk | TSV | 10.4s (9.0s + 1.4s) | 499.66 MB/s |
| frawk (cranelift) | CSV | 7.7s (6.6s + 1.4s) | 676.65 MB/s |
| frawk (cranelift) | TSV | 7.5s (6.5s + 1.4s) | 692.58 MB/s |
| frawk (cranelift, parallel) | CSV | 4.6s (13.6s + 3.6s) | 1117.49 MB/s |
| frawk (cranelift, parallel) | TSV | 4.1s (13.0s + 3.1s) | 1253.25 MB/s |
| frawk (llvm) | CSV | 7.6s (6.7s + 1.4s) | 679.50 MB/s |
| frawk (llvm) | TSV | 7.4s (6.4s + 1.4s) | 695.93 MB/s |
| frawk (llvm, parallel) | CSV | 4.8s (14.2s + 3.6s) | 1080.64 MB/s |
| frawk (llvm, parallel) | TSV | 4.4s (13.7s + 3.1s) | 1184.73 MB/s |
| tsv-utils | TSV | 3.5s (2.6s + 0.9s) | 1487.20 MB/s |

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

This benchmark is notable because `gawk` (narrowly, on Linux, and substantially
on MacOS) outperforms `frawk` in a single-threaded configuration, while `frawk`
manages to achieve higher throughput in parallel mode. This points to some
likely optimization opportunities for frawk, particularly since neither program
is as fast as tsv-utils, even when single-threaded.

**MacOS**
| Program | Format | Running Time | Throughput |
| -- | -- | -- | -- |
| gawk | TSV | 15.0s (13.7s + 1.3s) | 525.37 MB/s |
| mawk | TSV | 43.2s (41.6s + 1.5s) | 182.69 MB/s |
| frawk (cranelift) | CSV | 23.9s (22.7s + 1.2s) | 374.40 MB/s |
| frawk (cranelift) | TSV | 20.8s (19.8s + 1.0s) | 378.86 MB/s |
| frawk (cranelift, parallel) | CSV | 7.1s (27.1s + 1.2s) | 1260.63 MB/s |
| frawk (cranelift, parallel) | TSV | 6.6s (24.7s + 1.3s) | 1202.08 MB/s |
| frawk (llvm) | CSV | 23.6s (22.5s + 1.2s) | 378.26 MB/s |
| frawk (llvm) | TSV | 20.5s (19.5s + 1.0s) | 384.73 MB/s |
| frawk (llvm, parallel) | CSV | 7.0s (26.7s + 1.2s) | 1274.28 MB/s |
| frawk (llvm, parallel) | TSV | 6.5s (24.5s + 1.3s) | 1210.19 MB/s |
| tsv-utils | TSV | 4.9s (4.3s + 0.7s) | 1596.20 MB/s |

**Linux**
| Program | Format | Running Time | Throughput |
| -- | -- | -- | -- |
| gawk | TSV | 23.4s (21.7s + 1.7s) | 337.47 MB/s |
| mawk | TSV | 50.1s (48.4s + 1.7s) | 157.47 MB/s |
| frawk (cranelift) | CSV | 30.1s (28.2s + 1.8s) | 297.33 MB/s |
| frawk (cranelift) | TSV | 24.8s (23.3s + 1.5s) | 317.75 MB/s |
| frawk (cranelift, parallel) | CSV | 9.3s (34.5s + 2.7s) | 960.32 MB/s |
| frawk (cranelift, parallel) | TSV | 7.3s (26.9s + 2.2s) | 1082.68 MB/s |
| frawk (llvm) | CSV | 29.9s (28.2s + 1.7s) | 299.54 MB/s |
| frawk (llvm) | TSV | 24.2s (22.5s + 1.7s) | 326.07 MB/s |
| frawk (llvm, parallel) | CSV | 9.5s (34.9s + 2.7s) | 945.91 MB/s |
| frawk (llvm, parallel) | TSV | 7.1s (26.4s + 1.9s) | 1106.98 MB/s |
| tsv-utils | TSV | 5.7s (4.4s + 1.3s) | 1381.10 MB/s |
