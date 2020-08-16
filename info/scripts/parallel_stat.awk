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
    n_pids = 1
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
    m2s[PID] = Q

    if (!(PID == 1)) {
        min1 = min2 = min1L = min2L = max1 = max2 = max1L = max2L = 0;
    }
}
END {
    for (k in min1M) { min1 = min(min1, min1M[k]); }
    for (k in min2M) { min2 = min(min2, min2M[k]); }
    for (k in min1lM) { min1L = min(min1L, min1lM[k]); }
    for (k in min2lM) { min2L = min(min2L, min2lM[k]); }

    for (k in max1M) { max1 = max(max1, max1M[k]); }
    for (k in max2M) { max2 = max(max2, max2M[k]); }
    for (k in max1lM) { max1L = max(max1L, max1lM[k]); }
    for (k in max2lM) { max2L = max(max2L, max2lM[k]); }

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
            m2a = m2a + m2s[k] + (delta*delta) * ((na*nb)/(na+nb))
            na += nb           
        }

    }

    stddev = sqrt(m2a/(num_records-1))

    print "field","sum","min","max","min_length","max_length","mean","stddev"
    print h2,SUM,min2,max2,min2L,max2L,(SUM/num_records), stddev
    print h1,"NA",min1,max1,min1L,max1L,"NA","NA"
}
