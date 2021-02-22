# Compute the statistics computed by XSV on a string column and numeric column
MAWK=../mawk
GAWK="../gawk -b"
TSV_UTILS_BIN=../bin
XSV=xsv
FRAWK=../frawk

CSV1=../all_train.csv
CSV2=../TREE_GRM_ESTN.csv
TSV1=../all_train.tsv
TSV2=../TREE_GRM_ESTN.tsv

AWK_SCRIPT='function min(x,y) { return x<y?x:y; }
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
}'

PARALLEL_FRAWK_SCRIPT='function min(x,y) { return x<y?x:y; }
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
      for (pid in records) {
        min1 = min(min1, min1M[pid])
        min2 = min(min2, min2M[pid])
        min1L = min(min1L, min1lM[pid]);
        min2L = min(min2L, min2lM[pid]);

        max1 = max(max1, max1M[pid])
        max2 = max(max2, max2M[pid])
        max1L = max(max1L, max1lM[pid]);
        max2L = max(max2L, max2lM[pid]);
    }

    for (pid in records) {
        nb = records[pid]
        sb = sums[pid]
        mb = sb / nb
        m2b = m2s[pid]
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
'

# write to tmp file so as to not pollute the output
SCRIPT_FILE=$(mktemp)
PARALLEL_SCRIPT_FILE=$(mktemp)
echo "$AWK_SCRIPT" > "$SCRIPT_FILE"
echo "$PARALLEL_FRAWK_SCRIPT" > "$PARALLEL_SCRIPT_FILE"

for i in {1..5}; do
    set -x
    time $MAWK -F'\t' -f "$SCRIPT_FILE" "${TSV2}"
    time $GAWK -F'\t' -f "$SCRIPT_FILE" "${TSV2}"
    time $FRAWK -bllvm -F'\t' -f "$SCRIPT_FILE" "${TSV2}"
    time $FRAWK -bllvm -icsv -f "$SCRIPT_FILE" "${CSV2}"
    time $FRAWK -bllvm -pr  -F'\t' -f "$PARALLEL_SCRIPT_FILE" "${TSV2}"
    time $FRAWK -bllvm -pr  -icsv -f "$PARALLEL_SCRIPT_FILE" "${CSV2}"
    time $FRAWK -bcranelift -F'\t' -f "$SCRIPT_FILE" "${TSV2}"
    time $FRAWK -bcranelift -icsv -f "$SCRIPT_FILE" "${CSV2}"
    time $FRAWK -bcranelift -pr  -F'\t' -f "$PARALLEL_SCRIPT_FILE" "${TSV2}"
    time $FRAWK -bcranelift -pr  -icsv -f "$PARALLEL_SCRIPT_FILE" "${CSV2}"
    time $XSV stats -s5,6 "${CSV2}"
    time $XSV stats -s5,6 -d'\t' "${TSV2}"
    # caveate: doing a lot less work here.
    time $TSV_UTILS_BIN/tsv-summarize -H --sum 5 --mean 5 --min 5 --max 5 --stdev 5 --mean 5 ${TSV2}
    set +x
done

rm "$SCRIPT_FILE"
