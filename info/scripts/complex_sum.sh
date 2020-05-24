# Sum 2 numeric fields
RUST=./complex_sum/target/release/complex_sum
PYTHON="python3 ./complex_sum.py"
FRAWK=frawk

CSV=../TREE_GRM_ESTN.csv

for i in {1..5}; do
	set -x
	time $RUST "$CSV"
	time $PYTHON "$CSV"
	time $FRAWK -icsv 'function max(x,y) { return x<y?y:x; } "GS" == $8 { accum += (0.5*$1+0.5*max($4+0,$5+0))/1000.0 } END { print accum; }' "$CSV"
	set +x
done

