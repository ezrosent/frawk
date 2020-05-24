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
frawk's strengths when it comes to some common scripting tasks.

## Benchmark Setup

All benchmark numbers report the minimum wall time across 5 iterations. None of
these utilities make a huge use of parallelism (it seems that `xsv` does a
little, but it gets most of its improvements after it has built an index).

### Test Data
These benchmarks run on CSV and TSV versions of two data-sets:

* `all_train.csv` from the
  [HEPMASS dataset](https://archive.ics.uci.edu/ml/datasets/HEPMASS) in the UCI
  machine learning repository. This is a 7 million-row file and both CSV and TSV
  versions of the file are roughly 5.2GB, uncompressed.
* `TREE_GRM_ESTN.csv` from Forest Service (data-sets available
  [here](https://apps.fs.usda.gov/fia/datamart/CSV/datamart_csv.html)). This
  file is a little over 36 million rows; the CSV version is 8.9GB, and the TSV
  version is 7.9GB.

### Hardware
Numbers are reported from a 16-inch Macbook Pro running MacOS Catalina 10.15.4
with an 8-core i9. One thing to keep in mind about newer Apple hardware is that
the SSDs are very fast: read throughput can top 2.5GB/s. None of these
benchmarks are IO-bound on this machine, although various portions of the input
files might be cached by the OS.

### Tools
These benchmarks report values for a number of tools, each with slightly
different intended use-cases and strengths. Not all of the tools are set up well
to handle each intended task, but each makes an appearance in some subset of the
benchmarks. All benchmarks include `frawk`, of course; in this case both the
`use_jemalloc` and `allow_avx2` options were set.

* [`mawk`](https://invisible-island.net/mawk/) is an Awk implementation due to
  Mike Brennan. It is focused on the "Awk book" semantics for the language, and
  on efficiency.
* [`gawk`](https://www.gnu.org/software/gawk/manual/) is GNU's Awk. It is
  actively maintained and includes several extensions to the Awk language (such
  as multidimensional arrays). It is the default Awk on many Linux machines.
* [`xsv`](https://github.com/BurntSushi/xsv) is a command-line utility for
  performing common computations and transformations on CSV data.
* [`tsv-utils`](https://github.com/eBay/tsv-utils) is a set of command-line
  utilities offering a similar feature-set to xsv, but with a focus on TSV data.

There are plenty of other tools I could put here (see the benchmarks referenced
on xsv and tsv-utils' pages for more), but I chose a small set of tools that
could execute some subset of the benchmarks efficiently.

We benchmark on CSV and TSV data. Some utilities are not equipped for this, for
example:

* `tsv-utils` is explicitly optimized for TSV and not CSV. We only run it on the
  TSV variants of these datasets, but note that the bundled `csv2tsv` takes far
  longer than any of the benchmark workloads (roughly 40 seconds per file).
* Benchmarks for the `TREE_GRM_ESTN` dataset are not run on CSV input with
  `mawk` or `gawk`, as this file contains quoted fields and these utilities do
  not handle CSV escaping out of the box. We still run them on the TSV versions
  of these files, using `\t` as a field separator.
* `tsv-utils` and `xsv` are both limited to a specific menu of common functions.
  When those functions cannot express the computation in question, we leave them
  out.


<!-- mawk/gawk xsv, tsv-utils, -->
<!--
1. Sum of a column
2. All stats of a column (xsv)
3. Group-by
4. Filtering
5. Select (something IO-heavy)
5. Weighted average of two columns and the max of two more.
  -> write a custom script in python and Rust to do this
-->

## Ad-hoc number-crunching
Before we start benchmarking purpose-built tools, I want to emphasize the power
that Awk provides for basic number-crunching tasks. Not only can you efficiently
combine common tasks like filtering, group-by, and arithmetic, but you can build
fairly complex expressions as well. Suppose I wanted to take the weighted sum of
the first column, and the maximum of column 4 and column 5 when column 8 had a
particular value. In Awk this is a pretty simple script:

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
next thing I'd reach for is to write a short script in Rust. Here's one I came
up with quickly using the `csv` crate:
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
frawk's runtime includes the time it takes to compile a program.

| Program | Running Time (TREE_GRM_ESTN.csv) |
| -- | -- |
| Custom Rust | 28.1s |
| frawk | 19.5s |

frawk is a good deal faster. Now, the Rust script could of course be optimized
substantially; indeed, frawk is faster because it implements many such
optimizations. But if you need to do exploratory, ad-hoc computations like this,
a frawk script is probably going to be faster than the first few Rust programs
you write.


## Sum two columns
_Sum two numeric columns_
 

## Statistics
_Collect the sum, mean, minimum, maximum, minimum length, amximum length and
standard deviation of a numeric column, collect the maximum, minimum, maximum
length and minimum length of a string column_.

The numeric portion of this benchmark took a little over 10 seconds using
tsv-utils, but I am omitting it from the table because it does not support
the given summary statistics on string fields.


## Select
_Select 3 fields from the all_train dataset._

## Filter
_Print out all fields from the all_trains dataset matching a simple numeric
filter on two of the columns_

xsv does not support numeric filters, so it was omitted from this benchmark.
