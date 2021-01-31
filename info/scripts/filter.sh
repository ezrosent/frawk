# Filter fields in all_train based on numeric values
MAWK=../mawk
GAWK="../gawk -b"
TSV_UTILS_BIN=../bin
XSV=xsv
FRAWK=../frawk

CSV1=../all_train.csv
CSV2=../TREE_GRM_ESTN.csv
TSV1=../all_train.tsv
TSV2=../TREE_GRM_ESTN.tsv

AWK_SCRIPT='$4 > 0.000024 && $16 > 0.3'

# write to tmp file so as to not pollute the output

for i in {1..5}; do
	set -x
	time $MAWK -F,    "$AWK_SCRIPT" "$CSV1" > /dev/null
	time $MAWK -F'\t' "$AWK_SCRIPT" "$TSV1" > /dev/null
	time $GAWK -F,    "$AWK_SCRIPT" "$CSV1" > /dev/null
	time $GAWK -F'\t' "$AWK_SCRIPT" "$TSV1" > /dev/null
	time $FRAWK -bllvm -icsv "$AWK_SCRIPT" "$CSV1"  > /dev/null
	time $FRAWK -bllvm -itsv "$AWK_SCRIPT" "$TSV1"  > /dev/null
	time $FRAWK -bllvm -icsv -pr -j3 "$AWK_SCRIPT" "$CSV1"  > /dev/null
	time $FRAWK -bllvm -itsv -pr -j3 "$AWK_SCRIPT" "$TSV1"  > /dev/null
	time $FRAWK -bcranelift -icsv "$AWK_SCRIPT" "$CSV1" > /dev/null
	time $FRAWK -bcranelift -itsv "$AWK_SCRIPT" "$TSV1" > /dev/null
	time $FRAWK -bcranelift -icsv -pr -j3 "$AWK_SCRIPT" "$CSV1" > /dev/null
	time $FRAWK -bcranelift -itsv -pr -j3 "$AWK_SCRIPT" "$TSV1" > /dev/null
	time $TSV_UTILS_BIN/tsv-filter -H --gt 4:0.000025 --gt 16:0.3 "$TSV1" > /dev/null
	set +x
done
