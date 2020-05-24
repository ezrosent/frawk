import csv
import sys

def parse_float(s):
    try:
        return float(s)
    except ValueError:
        return 0.0


accum = 0
with open(sys.argv[1], "r") as f:
    csvr = csv.reader(f)
    for row in csvr:
        if row[7] == "GS":
            f1 = parse_float(row[0])
            f4 = parse_float(row[3])
            f5 = parse_float(row[4])
            accum += ((0.5*f1 + 0.5*max(f4,f5)) / 1000.0)

print(accum)
