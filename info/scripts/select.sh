# Select fields 1,8,19 from the all_train dataset
MAWK=../mawk
GAWK="../gawk -b"
TSV_UTILS_BIN=../bin
XSV=xsv
FRAWK=../frawk

CSV1=../all_train.csv
CSV2=../TREE_GRM_ESTN.csv
TSV1=../all_train.tsv
TSV2=../TREE_GRM_ESTN.tsv

AWK_SCRIPT_CSV='BEGIN { OFS="," } { print $1,$8,$19 }'
AWK_SCRIPT_TSV='BEGIN { OFS="\t" } { print $1,$8,$19 }'

# write to tmp file so as to not pollute the output

for i in {1..5}; do
	set -x
	time $MAWK -F,    "$AWK_SCRIPT_CSV" "$CSV1" > /dev/null
	time $MAWK -F'\t' "$AWK_SCRIPT_TSV" "$TSV1" > /dev/null
	time $GAWK -F,    "$AWK_SCRIPT_CSV" "$CSV1" > /dev/null
	time $GAWK -F'\t' "$AWK_SCRIPT_TSV" "$TSV1" > /dev/null
	time $FRAWK -bllvm -icsv "$AWK_SCRIPT_CSV" "$CSV1" > /dev/null
	time $FRAWK -bllvm -itsv "$AWK_SCRIPT_TSV" "$TSV1" > /dev/null
	time $FRAWK -bllvm -icsv -pr -j3 "$AWK_SCRIPT_CSV" "$CSV1" > /dev/null
	time $FRAWK -bllvm -itsv -pr -j3 "$AWK_SCRIPT_TSV" "$TSV1" > /dev/null
	time $FRAWK -bcranelift -icsv "$AWK_SCRIPT_CSV" "$CSV1" > /dev/null
	time $FRAWK -bcranelift -itsv "$AWK_SCRIPT_TSV" "$TSV1" > /dev/null
	time $FRAWK -bcranelift -icsv -pr -j3 "$AWK_SCRIPT_CSV" "$CSV1" > /dev/null
	time $FRAWK -bcranelift -itsv -pr -j3 "$AWK_SCRIPT_TSV" "$TSV1" > /dev/null
	time xsv select 1,8,19  "$CSV1" > /dev/null
	time xsv select -d'\t' 1,8,19  "$TSV1" > /dev/null
	time $TSV_UTILS_BIN/tsv-select -f 1,8,19 "$TSV1" > /dev/null
	set +x
done
