# Summarize the raw output contained in the ".out" files
$1 == "+" {
	prog=$2; inp=$NF
	if (/-j4/) {
		prog="frawk(parallel)"
	}
}

$1 == "real" { 
	key=prog " " inp
	real=$2
	getline
	user=$2
	getline
	sys=$2
	times[key] = times[key] " " sprintf("%s (%s + %s)", real, user, sys)
}
END {
	for (k in times) {
		print k, times[k]
	}
}

