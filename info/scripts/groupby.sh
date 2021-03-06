MAWK=../mawk
GAWK="../gawk -b"
TSV_UTILS_BIN=../bin
FRAWK=../frawk

CSV=../TREE_GRM_ESTN.csv
TSV=../TREE_GRM_ESTN.tsv

AWK_SCRIPT='
BEGIN { getline; }
{ N[$6]++; SUM[$6]+=$2; }
END {
    OFS="\t"
    for (k in N) {
        print k, ((SUM[k])/N[k]);
    }
}'

# write to tmp file so as to not pollute the output
SCRIPT_FILE=$(mktemp)
echo "$AWK_SCRIPT" > "$SCRIPT_FILE"

for i in {1..5}; do
	set -x
	time $MAWK -F'\t' -f "$SCRIPT_FILE" "${TSV}"
	time $GAWK -F'\t' -f "$SCRIPT_FILE" "${TSV}"
	time $FRAWK -bllvm -F'\t' -f "$SCRIPT_FILE" "${TSV}"
	time $FRAWK -bllvm -icsv -f "$SCRIPT_FILE" "${CSV}"
	time $FRAWK -bllvm -F'\t' -pr  -f "$SCRIPT_FILE" "${TSV}"
	time $FRAWK -bllvm -icsv -pr  -f "$SCRIPT_FILE" "${CSV}"
	time $FRAWK -bcranelift -F'\t' -f "$SCRIPT_FILE" "${TSV}"
	time $FRAWK -bcranelift -icsv -f "$SCRIPT_FILE" "${CSV}"
	time $FRAWK -bcranelift -F'\t' -pr  -f "$SCRIPT_FILE" "${TSV}"
	time $FRAWK -bcranelift -icsv -pr  -f "$SCRIPT_FILE" "${CSV}"
	time $TSV_UTILS_BIN/tsv-summarize -H --group-by 6 --mean 2  ${TSV}
	set +x
done

rm "$SCRIPT_FILE"
