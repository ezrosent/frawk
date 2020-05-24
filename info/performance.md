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
Before the remainder of 

## Statistics

## Filtering

## Group-By

