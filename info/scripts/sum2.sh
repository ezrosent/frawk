# Sum 2 numeric fields

MAWK=mawk
GAWK=gawk
TSV_UTILS_BIN=../bin
XSV=xsv
FRAWK=frawk

CSV1=../all_train.csv
CSV2=../TREE_GRM_ESTN.csv
TSV1=../all_train.tsv
TSV2=../TREE_GRM_ESTN.tsv

for i in {1..5}; do
	set -x
	time $MAWK -F, '{sum1 += $6; sum2 += $18;} END { print sum1,sum2}' ${CSV1}
	time $MAWK -F'\t' '{sum1 += $4; sum2 += $5;} END { print sum1,sum2}' ${TSV2}
	time $MAWK -F'\t' '{sum1 += $6; sum2 += $18;} END { print sum1,sum2}' ${TSV1}

	time $GAWK -F, '{sum1 += $6; sum2 += $18;} END { print sum1,sum2}' ${CSV1}
	time $GAWK -F'\t' '{sum1 += $6; sum2 += $18;} END { print sum1,sum2}' ${TSV1}
	time $GAWK -F'\t' '{sum1 += $4; sum2 += $5;} END { print sum1,sum2}' ${TSV2}

	# Columns escaped
	# time $MAWK -F, '{sum1 += $4; sum2 += $5;} END { print sum1,sum2}' ${CSV2}
	# time $GAWK -F, '{sum1 += $4; sum2 += $5;} END { print sum1,sum2}' ${CSV2}

	time $TSV_UTILS_BIN/tsv-summarize -H --sum 6,18 ${TSV1}
	time $TSV_UTILS_BIN/tsv-summarize -H --sum 4,5 ${TSV2}

	time $FRAWK -icsv '{sum1 += $6; sum2 += $18;} END { print sum1,sum2}' ${CSV1}
	time $FRAWK -icsv '{sum1 += $4; sum2 += $5;} END { print sum1,sum2}' ${CSV2}
	time $FRAWK -itsv '{sum1 += $6; sum2 += $18;} END { print sum1,sum2}' ${TSV1}
	time $FRAWK -itsv '{sum1 += $4; sum2 += $5;} END { print sum1,sum2}' ${TSV2}

	time $FRAWK -icsv -pr -j4 '{sum1 += $6; sum2 += $18;} END { print sum1,sum2}' ${CSV1}
	time $FRAWK -icsv -pr -j4 '{sum1 += $4; sum2 += $5;} END { print sum1,sum2}' ${CSV2}
	time $FRAWK -itsv -pr -j4 '{sum1 += $6; sum2 += $18;} END { print sum1,sum2}' ${TSV1}
	time $FRAWK -itsv -pr -j4 '{sum1 += $4; sum2 += $5;} END { print sum1,sum2}' ${TSV2}

	set +x
done

