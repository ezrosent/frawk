MAWK=mawk
GAWK=gawk
TSV_UTILS_BIN=../bin
FRAWK=frawk

CSV=../TREE_GRM_ESTN.csv
TSV=../TREE_GRM_ESTN.tsv

AWK_SCRIPT='
NR > 1 { N[$6]++; SUM[$6]+=$2; }
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
	time $FRAWK -itsv -f "$SCRIPT_FILE" "${TSV}"
	time $FRAWK -icsv -f "$SCRIPT_FILE" "${CSV}"
	time $FRAWK -itsv -pr -j4 -f "$SCRIPT_FILE" "${TSV}"
	time $FRAWK -icsv -pr -j4 -f "$SCRIPT_FILE" "${CSV}"
	time $TSV_UTILS_BIN/tsv-summarize -H --group-by 6 --mean 2  ${TSV}
	set +x
done

rm "$SCRIPT_FILE"
