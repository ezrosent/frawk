# Performance

## Disclaimer

One of frawk's goals is to be efficient. Despite how common it is to do, I've
found it is very hard to precisely state the degree to which an entire
_programming language implementation_ is or is not efficient. I have no doubt
that there are programs in frawk that are slower than an equivalent program in
Rust, or C, or even mawk or gawk. Some of these programs will represent bugs or
differences in the quality of the implementation, some will represent one
program being better-written than another, and still more may demonstrate that
in some nontrivial sense one language allows you to write the program more
efficiently than another.

I've found that it can be very hard to distinguish between these 3, so I
encourage everyone reading this document to draw at most modest conclusions
based on these numbers.  What I do hope this document demonstrates are some of
frawk's strengths when it comes to some common scripting tasks on CSV and TSV
data. Awk and frawk can process other data formats as well, but in my experience
the larger files are usually in some standardized format.

## Benchmark Setup

All benchmark numbers report the minimum wall time across 5 iterations. None of
these utilities make a noticeable use of multi-core parallelism in single
invocations.

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

Numbers are reported from a 16-inch Macbook Pro running MacOS Catalina 10.15.4
with an 8-core i9 clocked at 2.3GHz, boosting up to 4.8GHz. One thing to keep
in mind about newer Apple hardware is that the SSDs are very fast: read
throughput can top 2.5GB/s. None of these benchmarks are IO-bound on this
machine, although various portions of the input files might be cached by the OS.

### Tools
These benchmarks report values for a number of tools, each with slightly
different intended use-cases and strengths. Not all of the tools are set up well
to handle each intended task, but each makes an appearance in some subset of the
benchmarks. All benchmarks include `frawk`, of course; in this case both the
`use_jemalloc` and `allow_avx2` features were enabled.

* [`mawk`](https://invisible-island.net/mawk/) is an Awk implementation due to
  Mike Brennan. It is focused on the "Awk book" semantics for the language, and
  on efficiency. I used mawk v1.3.4.
* [`gawk`](https://www.gnu.org/software/gawk/manual/) is GNU's Awk. It is
  actively maintained and includes several extensions to the Awk language (such
  as multidimensional arrays). It is the default Awk on many Linux machines. I
  used gawk v5.0.1.
* [`xsv`](https://github.com/BurntSushi/xsv) is a command-line utility for
  performing common computations and transformations on CSV data. I used xsv
  v0.13.0.
* [`tsv-utils`](https://github.com/eBay/tsv-utils) is a set of command-line
  utilities offering a similar feature-set to xsv, but with a focus on TSV data. I
  used tsv-utils `v1.6.1_osx-x86_64_ldc2`, downloaded from the repo.

There are plenty of other tools I could put here (see the benchmarks referenced
on xsv and tsv-utils' pages for more), but I chose a small set of tools that
could execute some subset of the benchmarks efficiently.

We benchmark on CSV and TSV data. Some utilities are not equipped for all
possible configurations, for example:

* `tsv-utils` is explicitly optimized for TSV and not CSV. We only run it on the
  TSV variants of these datasets, but note that the bundled `csv2tsv` takes
  longer than many of the benchmark workloads (38s for TREE_GRM_ESTN and 24s for
  all_train).
* Benchmarks for the `TREE_GRM_ESTN` dataset are not run on CSV input with
  `mawk` or `gawk`, as it contains quoted fields and these utilities do
  not handle CSV escaping out of the box. We still run them on the TSV versions
  of these files, using `\t` as a field separator.
* `tsv-utils` and `xsv` are both limited to a specific menu of common functions.
  When those functions cannot easily express the computation in question, we
  leave them out.

Scripts and output for these benchmarks are
[here](https://github.com/ezrosent/frawk/blob/master/info/scripts). They do not
include the raw data or the various binaries in question, but they should be
straightforward to adapt for a given set of installs on another machine.

## Ad-hoc number-crunching
Before we start benchmarking purpose-built tools, I want to emphasize the power
that Awk provides for basic number-crunching tasks. Not only can you efficiently
combine common tasks like filtering, group-by, and arithmetic, but you can build
complex expressions as well. Suppose I wanted to take the scaled, weighted sum
of the first column, and the maximum of column 4 and column 5 when column 8 had
a particular value. In Awk this is a pretty simple script:

```
function max(x,y) { return x<y?y:x; }
"GS" == $8 { accum += (0.5*$1+0.5*max($4+0,$5+0))/1000.0 }
END { print accum; }
```

> Note: Where the `+0`s ensure we are always performing a numeric comparison. In
> Awk this is unnecessary if all instances of column $4 and column $5 are
> numeric; in frawk this is required: max will perform a lexicographic
> comparison instead of a numeric one without the explicit conversion.

But I have to preprocess the data to run Awk on it, as Awk doesn't properly
support quoted CSV fields by default. Supposing I didn't want to do that, the
next thing I'd reach for is to write a short script in Python:
```
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
the script in Rust instead, rather than spending time trying to optimize the
Python.  Here's one I came up with quickly using the
[`csv`](https://github.com/BurntSushi/rust-csv) crate:

```
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

This takes 8.2 seconds to compile a release build. Depending on the setting this
either is or isn't important. We'll leave it out for now, but keep in mind that
frawk's and python's runtimes include the time it takes to compile a program.

| Program | Running Time (TREE_GRM_ESTN.csv) |
| -- | -- |
| Python | 2m47.6s |
| Rust | 28.2s |
| frawk | 19.4s |

frawk is a good deal faster than the other options. Now, the Rust script could
of course be optimized substantially (frawk is implemented in Rust, after all).
But if you need to do exploratory, ad-hoc computations like this, a frawk script
is probably going to be faster than the first few Rust programs you write.

With that said, a lot of the time you do not need a full programming language,
and there are purpose-built tools for computing the particular value or
transformation on the dataset. The rest of these benchmarks compare frawk and
Awk to some of those.

## Sum two columns
_Sum columns 4 and 5 from TREE_GRM_ESTN and columns 6 and 18 for all_train_

Programs:
* Awk: `-F{,\t} {sum1 += ${6,4}; sum2 += ${18,5};} END { print sum1,sum2}`
* frawk: `-i{c,t}sv {sum1 += ${6,4}; sum2 += ${18,5};} END { print sum1,sum2}`
* tsv-utils: `tsv-summarize -H --sum {6,4},{18,5}`

Awk was only run on the TSV version of TREE_GRM_ESTN, as it has quote-escaped
columns; tsv-utils was only run on TSV versions of both files, xsv was not
included because there is no way to persuade it to _just_ compute a sum.

| Program | Format | Running Time (TREE_GRM_ESTN) | Running Time (all_train) |
| -- | -- | -- | -- |
| mawk | TSV | 42.5s | 10.9s |
| mawk | CSV | NA | 10.9s |
| gawk | TSV | 25.8s | 24.1s |
| gawk | CSV | NA | 25.8s |
| tsv-utils | TSV | 6.0s | 2.7s |
| frawk | TSV | 14.1s | 7.3s |
| frawk | CSV | 17.2s | 7.5s |


## Statistics
_Collect the sum, mean, minimum, maximum, minimum length, maximum length and
standard deviation of a numeric column, collect the maximum, minimum, maximum
length and minimum length of a string column_.

As in the "sum" benchmark, frawk and Awk have the same programs, but with frawk
using the `icsv` and `itsv` options. This benchmark is meant to mirror `xsv`'s
`summarize` command. The xsv invocation is `xsv summarize -s5,6 [-d\t]`. The
Awk program is more involved:

```
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

The numeric portion of this benchmark took a little over 10 seconds using
tsv-utils, but I am omitting it from the table because it does not support
the given summary statistics on string fields. All numbers are reported for
TREE_GRM_ESTN.

| Program | Format | Running Time |
| -- | -- | -- |
| mawk | TSV | 1m13.5s |
| gawk | TSV | 1m34.9s |
| xsv | TSV | 32.9s |
| xsv | CSV | 34.4s |
| frawk | TSV | 16.9s |
| frawk | CSV | 20.1s |


## Select
_Select 3 fields from the all_train dataset._

This is a task that all the benchmark programs support. The Awk script looks
like `BEGIN { OFS={",","\t"} { print $1,$2,$8 }`. all_train does not have any
quoted fields, so gawk and mawk can use them with the `-F,` option. As before,
frawk uses the `icsv` and `itsv` options.

The xsv invocation looks like `xsv select [-d'\t'] 1,8,19`, and the tsv-utils
invocation is `tsv-select -f1,8,19`. All output is written to `/dev/null`, so
all times surely underestimate the true running time of such an operation.

| Program | Format | Running Time |
| -- | -- | -- |
| mawk | TSV | 8.4s |
| mawk | CSV | 8.5s |
| gawk | TSV | 24.2s |
| gawk | CSV | 24.1s |
| xsv | TSV | 5.3s |
| xsv | CSV | 5.3s |
| tsv-utils | TSV | 1.9s |
| frawk | TSV | 3.6s |
| frawk | CSV | 3.8s |

## Filter
_Print out all records from the `all_train` dataset matching a simple numeric
filter on two of the columns_

The Awk script computing this filter is `$4 > 0.000024 && $16 > 0.3`. Because
`all_train` has no quoted fields, mawk and gawk can both run using the `-F,` and
`-F\t` options. As before, frawk runs the same script with the `icsv` and `itsv`
options. The tsv-utils invocation is `tsv-filter -H --gt 4:0.000025 --gt 16:0.3
./all_train.tsv` (taken from the same benchmark on the tsv-utils repo).  xsv
does not support numeric filters, so it was omitted from this benchmark.

| Program | Format | Running Time |
| -- | -- | -- |
| mawk | TSV | 10.1s |
| mawk | CSV | 10.2s |
| gawk | TSV | 17.5s |
| gawk | CSV | 17.2s |
| tsv-utils | TSV | 2.4s |
| frawk | TSV | 6.6s |
| frawk | CSV | 7.0s |
