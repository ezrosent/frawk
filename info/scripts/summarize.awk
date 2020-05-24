# Summarize the raw output contained in the ".out" files
$1 == "+" { prog=$2; inp=$NF }

$1 == "real" { 
	key=prog " " inp
	times[key] = times[key] " " $2
}
END {
	for (k in times) {
		print k, times[k]
	}
}

