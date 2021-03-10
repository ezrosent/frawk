//! TODO: come back to the 4 test cases that couldn't be generated automatically
//! TODO: look for tests doing file io.
//! TODO: get a better way for
use assert_cmd::Command;
use std::fs::{read_to_string, File};
use std::io::Write;
use tempfile::tempdir;

#[cfg(feature = "llvm_backend")]
const BACKEND_ARGS: &'static [&'static str] = &["-binterp", "-bllvm", "-bcranelift"];
#[cfg(not(feature = "llvm_backend"))]
const BACKEND_ARGS: &'static [&'static str] = &["-binterp", "-bcranelift"];

#[test]
fn t_0() {
   let prog: String = r###"{ print }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_0.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_0a() {
   let prog: String = r###"{i = i+1; print i, NR}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_0a.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_1() {
   let prog: String = r###"BEGIN	{FS=":"}
	{print $1, $2, $3}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_1_x() {
   let prog: String = r###"{i="count" $1 $2; print i , $0}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_1_x.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_2() {
   let prog: String = r###"BEGIN	{OFS="==="}
	{print $1, $2, $3}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_2.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_2_x() {
   let prog: String = r###"{i=2; j=$3; $1=i;print i,j,$1}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_2_x.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_3() {
   let prog: String = r###"$1 == "5" || $1 == "4""###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_3.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_3_x() {
   let prog: String = r###"{
x = $1
while (x > 1) {
	print x
	x = x / 10
}
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_3_x.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_4() {
   let prog: String = r###"$1 ~ /5/ || $1 ~ /4/"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_4.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_4_x() {
   let prog: String = r###"{i=$(1); print i}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_4_x.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_5_x() {
   let prog: String = r###"{$(1) = "xxx"; print $1,$0}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_5_x.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_6() {
   let prog: String = r###"/a|b|c/	{
	i = $1
	print
	while (i >= 1) {
		print "	", i
		i = i / 10
	}
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_6.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_6a() {
   let prog: String = r###"/a|b|c/	{
	print
	for (i = $1; i >= 1; )
		print " ", i /= 10
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_6a.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_6b() {
   let prog: String = r###"/a|b|c/	{
	print
	for (i = $1; (i /= 10)>= 1; )
		print " ", i
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_6b.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_6_x() {
   let prog: String = r###"{print NF,$0}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_6_x.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_8_x() {
   let prog: String = r###"{$2=$1; print}

# this should produce a blank for an empty input line
# since it has created fields 1 and 2."###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_8_x.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_8_y() {
   let prog: String = r###"{$1=$2; print}

# this should print nothing for an empty input line
# since it has only referred to $2, not created it,
# and thus only $1 exists (and it's null).

# is this right???"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_8_y.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_a() {
   let prog: String = r###"	{if (amount[$2] "" == "") item[++num] = $2;
	 amount[$2] += $1
	}
END	{for (i=1; i<=num; i++)
		print item[i], amount[item[i]]
	}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_a.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_aeiou() {
   let prog: String = r###"/^[^aeiouy]*[aeiou][^aeiouy][aeiouy][aeiouy]*[^aeiouy]*$/"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_aeiou.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_aeiouy() {
   let prog: String = r###"/^[^aeiouy]*a[^aeiouy]*e[^aeiouy]*i[^aeiouy]*o[^aeiouy]*u[^aeiouy]*y[^aeiouy]*$/"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_aeiouy.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_array() {
   let prog: String = r###"	{ x[NR] = $0 }

END {
	i = 1
	while (i <= NR) {
		print x[i]
		split (x[i], y)
		usage = y[1]
		name = y[2]
		print "   ", name, usage
		i++
	}
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_array.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_array1() {
   let prog: String = r###"{for(i=1; i<=NF; i++) {
	if (x[$i] == "")
		y[++n] = $i
	x[$i]++
 }
}
END {
	for (i=0; i<n; i++)
		print (y[i], x[y[i]])
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_array1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_array2() {
   let prog: String = r###"$2 ~ /^[a-l]/	{ x["a"] = x["a"] + 1 }
$2 ~ /^[m-z]/	{ x["m"] = x["m"] + 1 }
$2 !~ /^[a-z]/	{ x["other"] = x["other"] + 1 }
END { print NR, x["a"], x["m"], x["other"] }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_array2.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_assert() {
   let prog: String = r###"# tests whether function returns sensible type bits

function assert(cond) { # assertion
    if (!cond) print "   >>> assert failed <<<"
}

function i(x) { return x }

{ m = length($1); n = length($2); n = i(n); assert(m > n) }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_assert.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_avg() {
   let prog: String = r###"{s = s + $1; c = c + 1}
END {
print "sum=", s, " count=", c
print "avg=", s/c
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_avg.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_be() {
   let prog: String = r###"# some question of what FILENAME ought to be before execution.
# current belief:  "-", or name of first file argument.
# this may not be sensible.

BEGIN { print FILENAME }
END { print NR }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_be.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_beginexit() {
   let prog: String = r###"BEGIN {
	while (getline && n++ < 10)
		print
	exit
}
{ print }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_beginexit.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_beginnext() {
   let prog: String = r###"BEGIN {
	while (getline && n++ < 10)
		print
	print "tenth"
}
{ print }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_beginnext.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_break() {
   let prog: String = r###"{
for (i=1; i <= NF; i++)
	if ($i ~ /^[a-z]+$/) {
		print $i " is alphabetic"
		break
	}
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_break.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_break1() {
   let prog: String = r###"	{ x[NR] = $0 }
END {
	for (i = 1; i <= NR; i++) {
		print i, x[i]
		if (x[i] ~ /shen/)
			break
	}
	print "got here"
	print i, x[i]
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_break1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_break2() {
   let prog: String = r###"	{ x[NR] = $0 }
END {
	for (i=1; i <= NR; i++) {
		print i, x[i]
		if (x[i] ~ /shen/)
			break
	}
	print "got here"
	print i, x[i]
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_break2.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_break3() {
   let prog: String = r###"{	for (i = 1; i <= NF; i++) {
		for (j = 1; j <= NF; j++)
			if (j == 2)
				break;
		print "inner", i, j
	}
	print "outer", i, j
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_break3.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_bug1() {
   let prog: String = r###"# this program fails if awk is created without separate I&D
# prints garbage if no $3
{ print $1, $3 }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_bug1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_builtins() {
   let prog: String = r###"/^[0-9]/ { print $1,
	length($1),
	log($1),
	sqrt($1),
	int(sqrt($1)),
	exp($1 % 10) }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_builtins.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_b_x() {
   let prog: String = r###"{$6=":::" ; print $6; print NF, $0}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_b_x.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_cat() {
   let prog: String = r###"{print $2 " " $1}
{print $1 " " "is", $2}
{print $2 FS "is" FS $1}
{print length($1 $2), length($1) + length($2)}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_cat.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_cat1() {
   let prog: String = r###"{print x $0}	# should precede by zero"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_cat1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_cat2() {
   let prog: String = r###"{$1 = $1 "*"; print}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_cat2.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_cmp() {
   let prog: String = r###"$2 > $1"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_cmp.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_coerce() {
   let prog: String = r###"END {	print i, NR
	if (i < NR)
		print i, NR
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_coerce.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_coerce2() {
   let prog: String = r###"{
	print index(1, $1)
	print substr(123456789, 1, 3)
	print 1 in x
	print 1 23 456
	print 123456789 ~ 123, 123456789 ~ "abc"
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_coerce2.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_comment() {
   let prog: String = r###"# this is a comment line
# so is this
/#/	{ print "this one has a # in it: " $0	# comment
	print "again:" $0
	}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_comment.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_comment1() {
   let prog: String = r###"#comment
       #
BEGIN { x = 1 }
/abc/ { print $0 }
#comment
END { print NR }
#comment"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_comment1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_concat() {
   let prog: String = r###"{ x = $1; print x (++i) }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_concat.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_cond() {
   let prog: String = r###"{ print (substr($2,1,1) > substr($2,2,1)) ? $1 : $2 }
{ x = substr($1, 1, 1); y = substr($1, 2, 1); z = substr($1, 3, 1)
  print (x > y ? (x > z ? x : z) : y > z ? y : z) }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_cond.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_contin() {
   let prog: String = r###"{
for (i = 1; i <= NF; i++) {
	if ($i ~ /^[0-9]+$/)
		continue;
	print $i, " is non-numeric"
	next
}
print $0, "is all numeric"
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_contin.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_count() {
   let prog: String = r###"END { print NR }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_count.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_crlf() {
   let prog: String = r###"# checks whether lines with crlf are parsed ok

{print  \
 }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_crlf.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_cum() {
   let prog: String = r###"{i = i + $1; print i}
END {
print i
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_cum.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_delete0() {
   let prog: String = r###"NF > 0 { 
  n = split($0, x)
  if (n != NF)
	printf("split screwed up %d %d\n", n, NF)
  delete x[1]
  k = 0
  for (i in x)
	k++
  if (k != NF-1)
	printf "delete miscount %d elems should be %d at line %d\n", k, NF-1, NR 
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_delete0.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_delete1() {
   let prog: String = r###"{ split("1 1.2 abc", x)
  x[$1]++
  delete x[1]
  delete x[1.2]
  delete x["abc"]
  delete x[$1]
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_delete1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_delete2() {
   let prog: String = r###"NR < 50 { n = split($0, x)
  for (i = 1; i <= n; i++)
  for (j = 1; j <= n; j++)
	y[i,j] = n * i + j
  for (i = 1; i <= n; i++)
	delete y[i,i]
  k = 0
  for (i in y)
	k++
  if (k != int(n^2-n))
	printf "delete2 miscount %d vs %d at %d\n", k, n^2-n, NR
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_delete2.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_delete3() {
   let prog: String = r###"{ x[$1] = $1
  delete x[$1]
  n = 0
  for (i in x) n++
  if (n != 0)
	print "error", n, "at", NR
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_delete3.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_do() {
   let prog: String = r###"NF > 0 {
	t = $0
	gsub(/[ \t]+/, "", t)
	n = split($0, y)
	if (n > 0) {
		i = 1
		s = ""
		do {
			s = s $i
		} while (i++ < NF)
	}
	if (s != t)
		print "bad at", NR
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_do.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_d_x() {
   let prog: String = r###"BEGIN {FS=":" ; OFS=":"}
{print NF "	",$0}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_d_x.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_e() {
   let prog: String = r###"$1 < 10 || $2 ~ /bwk/ "###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_e.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_else() {
   let prog: String = r###"{ if($1>1000) print "yes"
  else print "no"
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_else.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_exit() {
   let prog: String = r###"{ print }
$1 < 5000 { exit NR }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_exit.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_exit1() {
   let prog: String = r###"BEGIN {
	print "this is before calling myabort"
	myabort(1)
	print "this is after calling myabort"
} 
function myabort(n) {
	print "in myabort - before exit", n
	exit 2
	print "in myabort - after exit"
}
END {
	print "into END"
	myabort(2)
	print "should not see this"
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_exit1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_f() {
   let prog: String = r###"{print $2, $1}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_f.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_f0() {
   let prog: String = r###"$1 ~ /x/ {print $0}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_f0.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_f1() {
   let prog: String = r###"{$1 = 1; print}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_f1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_f2() {
   let prog: String = r###"{$1 = 1; print $0}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_f2.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_f3() {
   let prog: String = r###"{$1 = NR; print}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_f3.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_f4() {
   let prog: String = r###"{$1 = NR; print $0}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_f4.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_for() {
   let prog: String = r###"{ for (i=1; i<=NF; i++)
	print i, $i
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_for.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_for1() {
   let prog: String = r###"{
	i = 1
	for (;;) {
		if (i > NF)
			next
		print i, $i
		i++
	}
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_for1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_for2() {
   let prog: String = r###"{
	for (i=1;;i++) {
		if (i > NF)
			next
		print i, $i
	}
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_for2.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_for3() {
   let prog: String = r###"{ for (i = 1; length($i) > 0; i++)
	print i, $i
}
{ for (i = 1;
	 length($i) > 0;
	 i++)
	print $i
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_for3.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_format4() {
   let prog: String = r###"BEGIN {
text=sprintf ("%125s", "x")
print length (text)
print text
xxx=substr (text,1,105)
print length (xxx)
print xxx
exit
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_format4.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_fun() {
   let prog: String = r###"function g() { return "{" f() "}" }
function f() { return $1 }
 { print "<" g() ">" }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_fun.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_fun0() {
   let prog: String = r###"function f(a) { print "hello"; return a }
{ print "<" f($1) ">" }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_fun0.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_fun1() {
   let prog: String = r###"function f(a,b,c) { print "hello" }
NR < 3 { f(1,2,3) }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_fun1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_fun2() {
   let prog: String = r###"function f(n) {
	while (n < 10) {
		print n
		n = n + 1
	}
}
function g(n) {
	print "g", n
}
{ f($1); g($1); print n }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_fun2.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_fun3() {
   let prog: String = r###"function f(n) { while ((n /= 10) > 1)  print n }
function g(n) { print "g", n }
{ f($1); g($1) }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_fun3.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_fun4() {
   let prog: String = r###"function f(a, n) {
	for (i=1; i <= n; i++)
		print "	" a[i]
}

{	print
	n = split($0, x)
	f(x, n)
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_fun4.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_fun5() {
   let prog: String = r###"function f(a) {
	return split($0, a)
}
{
	print
	n = f(x)
	for (i = 1; i <= n; i++)
		print "	" x[i]
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_fun5.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_f_x() {
   let prog: String = r###"$1>0 {print $1, sqrt($1)}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_f_x.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_getline1() {
   let prog: String = r###"{ x = $1
  for (i = 1; i <= 3; i++)
	if (getline)
		x = x " " $1
  print x
  x = ""
}
END {
  if (x != "") print x
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_getline1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_getval() {
   let prog: String = r###"{ # tests various resetting of $1, $0, etc.

	$1 = length($1) + length($2)
	print $0 + 0

}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_getval.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_gsub() {
   let prog: String = r###"{gsub(/[aeiou]/,"foo"); print}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_gsub.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_gsub1() {
   let prog: String = r###"{gsub(/$/,"x"); print}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_gsub1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_gsub3() {
   let prog: String = r###"length($1) {gsub(substr($1,1,1),"(&)"); print}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_gsub3.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_gsub4() {
   let prog: String = r###"length($1) == 0 { next }

{gsub("[" $1 "]","(&)"); print}
{gsub("[" $1 "]","(\\&)"); print}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_gsub4.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_if() {
   let prog: String = r###"{if($1 || $2) print}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_if.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_in() {
   let prog: String = r###"BEGIN {
	x["apple"] = 1;
	x["orange"] = 2;
	x["lemon"] = 3;
	for (i in x)
		print i, x[i] | "sort"
	close("sort")
	exit
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_in.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_in1() {
   let prog: String = r###"	{ if (amount[$2] == "")
		name[++n] = $2
	  amount[$2] += $1
	}
END	{ for (i in name)
		print i, name[i], amount[name[i]] | "sort"
	}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_in1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_in2() {
   let prog: String = r###"	{ x[substr($2, 1, 1)] += $1 }
END	{ for (i in x)
		print i, x[i]
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_in2.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_in3() {
   let prog: String = r###"	{ x[NR] = $0 }
END {
	for (i in x)
		if (x[i] ~ /shen/)
			break
	print i, x[i]
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_in3.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_incr() {
   let prog: String = r###"{ ++i; --j; k++; l-- }
END { print NR, i, j, k, l }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_incr.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_incr2() {
   let prog: String = r###"{ s = 0
  for (i=1; i <= NF; )
	if ($(i) ~ /^[0-9]+$/)
		s += $(i++)
	else
		i++
  print s
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_incr2.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_incr3() {
   let prog: String = r###"{ s = 0
  for (i=1; i <= NF;  s += $(i++))
	;
  print s
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_incr3.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_index() {
   let prog: String = r###"{	n = length
	d = 0
	for (i = 1; i <= n; i++)
		if ((k = index($0, substr($0, i))) != i) {
			d = 1
			break;
		}
	if (d)
		print $0, "has duplicate letters"
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_index.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_intest() {
   let prog: String = r###"{
	line = substr($0, index($0, " "))
	print line
	n = split(line, x)
	if ($1 in x)
		print "yes"
	else
		print "no"
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_intest.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_intest2() {
   let prog: String = r###"{
	line = substr($0, index($0, " "))
	print line
	n = split(line, x)
	x[$0, $1] = $0
	print x[$0, $1]
	print "<<<"
for (i in x) print i, x[i]
	print ">>>"
	if (($0,$1) in x)
		print "yes"
	if ($1 in x)
		print "yes"
	else
		print "no"
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_intest2.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_i_x() {
   let prog: String = r###"$1+0 > 0 {i=i+log($1); print i,log($1)}
END {print i}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_i_x.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_j_x() {
   let prog: String = r###"{i=i+sqrt($1); print i,sqrt($1)}
END {print sqrt(i),i}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_j_x.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_longstr() {
   let prog: String = r###"BEGIN{
x = "111111111122222222233333333334444444444555555555566666666667777777777888888888899999999990000000000"
printf "%s\n", x
exit
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_longstr.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_makef() {
   let prog: String = r###"{$3 = 2*$1; print $1, $2, $3}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_makef.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_match() {
   let prog: String = r###"$2 ~ /ava|bwk/"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_match.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_match1() {
   let prog: String = r###"NF > 0 && match($NF, $1) {
	print $0, RSTART, RLENGTH
	if (RLENGTH != length($1))
		printf "match error at %d: %d %d\n",
			NR, RLENGTH, RSTART >"/dev/tty"
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_match1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_max() {
   let prog: String = r###"length > max	{ max = length; x = $0}
END { print max, x }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_max.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_mod() {
   let prog: String = r###"NR % 2 == 1"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_mod.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_monotone() {
   let prog: String = r###"/^a?b?c?d?e?f?g?h?i?j?k?l?m?n?o?p?q?r?s?t?u?v?w?x?y?z?$|^z?y?x?w?v?u?t?s?r?q?p?o?n?m?l?k?j?i?h?g?f?e?d?c?b?a?$/"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_monotone.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_nameval() {
   let prog: String = r###"	{ if (amount[$2] == "")
		name[++n] = $2
	  amount[$2] += $1
	}
END	{ for (i = 1; i <= n; i++)
		print name[i], amount[name[i]]
	}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_nameval.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_next() {
   let prog: String = r###"$1 > 5000	{ next }
{ print }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_next.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_NF() {
   let prog: String = r###"{ OFS = "|"; print NF; NF = 2; print NF; print; $5 = "five"; print NF; print }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_NF.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_not() {
   let prog: String = r###"$2 !~ /ava|bwk/
!($1 < 2000)
!($2 ~ /bwk/)
!$2 ~ /bwk/"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_not.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_null0() {
   let prog: String = r###"BEGIN { FS = ":" }
{	if (a) print "a", a
	if (b == 0) print "b", b
	if ( c == "0") print "c", c
	if (d == "") print "d", d
	if (e == 1-1) print "e", e
}
$1 == 0	{print "$1 = 0"}
$1 == "0"	{print "$1 = quoted 0"}
$1 == ""	{print "$1 = null string"}
$5 == 0	{print "$5 = 0"}
$5 == "0"	{print "$5 = quoted 0"}
$5 == ""	{print "$5 = null string"}
$1 == $3 {print "$1 = $3"}
$5 == $6 {print "$5 = $6"}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_null0.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_ofmt() {
   let prog: String = r###"BEGIN	{OFMT="%.5g"}
	{print $1+0}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_ofmt.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_ofs() {
   let prog: String = r###"BEGIN	{ OFS = " %% "; ORS = "##" }
	{ print $1, $2; print }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_ofs.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_ors() {
   let prog: String = r###"BEGIN	{ORS="abc"}
	{print $1, $2, $3}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_ors.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_pat() {
   let prog: String = r###"/a/ || /b/
/a/ && /b/
/a/ && NR > 10
/a/ || NR > 10"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_pat.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_pipe() {
   let prog: String = r###"BEGIN {print "read /usr/bwk/awk/t.pipe" | "cat"}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_pipe.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_pp() {
   let prog: String = r###"/a/,/b/"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_pp.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_pp1() {
   let prog: String = r###"/bwk/,/bwk/	{ print $2, $1 }
/ava/,/ava/	{ print $2, $1 }
/pjw/,/pjw/	{ print $2, $1 }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_pp1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_pp2() {
   let prog: String = r###"/bwk/,/scj/	{ print "1: ", $0 }
/bwk/, /bsb/	{ print "2: ", $0 }
/mel/, /doug/	{ print "3: ", $0 }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_pp2.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_printf() {
   let prog: String = r###"{
 printf "%%: %s ... %s \t", $2, $1
 x = sprintf("%8d %10.10s", $1, $2)
 print x
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_printf.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_quote() {
   let prog: String = r###"{print "\"" $1 "\""}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_quote.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_randk() {
   let prog: String = r###"{
	k = 2
	n = NF
	i = 1
	while ( i <= n ) {
		if ( rand() < k/n ) {
			print i
			k--
		}
		n--
		i++
	}
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_randk.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_re1() {
   let prog: String = r###"/[a-cg-j1-3]/	{ print $0 " matches /[a-cg-j1-3]/" }
/[^aeiou]/	{ print $0 " matches /[^aeiou]/" }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_re1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_re1a() {
   let prog: String = r###"BEGIN { r1 = "[a-cg-j1-3]"
	r2 = "[^aeiou]"
}

$0 ~ r1	{ print $0 " matches /[a-cg-j1-3]/" }
$0 ~ r2	{ print $0 " matches /[^aeiou]/" }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_re1a.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_re3() {
   let prog: String = r###"{	r1 = $1
	r2 = $1 ":"
}

length(r1) && $0 ~ r1	{ print $0 " matches " r1 }
length(r1) && $0 ~ r2	{ print $0 " matches " r2 }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_re3.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_re4() {
   let prog: String = r###"BEGIN {	r1 = "xxx"
	r2 = "xxx" ":"
	r3 = ":" r2
	r4 = "a"
}

$0 ~ r1	{ print $0 " matches " r1 }
$0 ~ r2	{ print $0 " matches " r2 }
$0 ~ r3	{ print $0 " matches " r3 }
$0 ~ r4	{ print $0 " matches " r4 }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_re4.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_re5() {
   let prog: String = r###"BEGIN {	for (i = 0; i <= 9; i++) r[i] = i }

{ for (i in r) if ($0 ~ r[i]) print }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_re5.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_re7() {
   let prog: String = r###"/^([0-9]+\.?[0-9]*|\.[0-9]+)((e|E)(\+|-)?[0-9]+)?$/"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_re7.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_rec() {
   let prog: String = r###"{ print sqrt($1) }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_rec.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_redir1() {
   let prog: String = r###"$1%2==1	{print >"foo.odd"}
$1%2==0	{print >"foo.even"}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_redir1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_reFS() {
   let prog: String = r###"BEGIN	{ FS = "\t+" }
	{ print $1, $2 }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_reFS.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_reg() {
   let prog: String = r###"/[^\[\]]/
!/^\[/
!/^[\[\]]/
/[\[\]]/"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_reg.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_roff() {
   let prog: String = r###"NF > 0	{
	for (i = 1; i <= NF; i++) {
		n = length($i)
		if (n + olen >= 60) {
			print oline
			olen = n
			oline = $i
		} else {
			oline = oline " " $i
			olen += n
		}
	}
}

NF == 0 {
	print oline
	olen = 0
}

END {
	if (olen > 0)
		print oline
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_roff.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_sep() {
   let prog: String = r###"BEGIN	{ FS = "1"; print "field separator is", FS }
NF>1	{ print $0 " has " NF " fields" }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_sep.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_seqno() {
   let prog: String = r###"{print NR, $0}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_seqno.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_set0() {
   let prog: String = r###"{$0 = $1; print; print NF, $0; print $2}
{$(0) = $1; print; print NF, $0; print $2}
{ i = 1; $(i) = $i+1; print }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_set0.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_set0a() {
   let prog: String = r###"{$0 = $2; print; print NF, $0; print $1}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_set0a.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_set0b() {
   let prog: String = r###"{x=$1 = $0 = $2; print }
{$0 = $2 = $1; print }
{$(0) = $(2) = $(1); print }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_set0b.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_set1() {
   let prog: String = r###"function f(x) { x = 1; print x }
{ f($0)
  f($1) }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_set1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_set2() {
   let prog: String = r###"{ n = length($0) % 2
  $n = $2
  print
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_set2.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_set3() {
   let prog: String = r###"{ i = 1; $i = $i/10; print }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_set3.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_split1() {
   let prog: String = r###"BEGIN	{ z = "stuff" }
{ split ($0, x); print x[3], x[2], x[1] }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_split1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_split2() {
   let prog: String = r###"{ split ($0, x); print x[2], x[1] }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_split2.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_split2a() {
   let prog: String = r###"BEGIN {
  a[1]="a b"
  print split(a[1],a),a[1],a[2]
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_split2a.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_split3() {
   let prog: String = r###"{ a = $0 " " $0 " " $0
  if ($1 != "")
      n = split (a, x, "[" $1 "]")
  print n, x[1], x[2], x[3], x[4] }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_split3.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_split4() {
   let prog: String = r###"{ a = $0 " " $0 " " $0 " " 123
  n = split (a, x, /[ \t][ \t]*/)
  print n, x[1], x[2], x[3], x[4]
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_split4.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_split8() {
   let prog: String = r###"{
	n = split ($0, x, /[ 	]+/)
	print n
	if (n != NF)
		print "split botch at ", NR, n, NF
	for (i=1; i<=n; i++)
		if ($i != x[i])
			print "different element at ", i, x[i], $i
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_split8.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_split9() {
   let prog: String = r###"{
	n = split ($0, x, FS)
	if (n != NF)
		print "botch at ", NR, n, NF
	for (i=1; i<=n; i++)
		if ($i != x[i])
			print "diff at ", i, x[i], $i
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_split9.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_split9a() {
   let prog: String = r###"BEGIN { FS = "a" }
{
	n = split ($0, x, FS)
	if (n != NF)
		print "botch at ", NR, n, NF
	for (i=1; i<=n; i++)
		if ($i != x[i])
			print "diff at ", i, x[i], $i
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_split9a.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_stately() {
   let prog: String = r###"/^(al|ak|az|ar|ca|co|ct|de|fl|ga|hi|io|il|in|ia|ks|ky|la|me|md|ma|mi|mn|ms|mo|mt|nb|nv|nh|nj|nm|ny|nc|nd|oh|ok|or|pa|ri|sc|sd|tn|tx|ut|vt|va|wa|wv|wi|-|wy)*$/"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_stately.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_strcmp() {
   let prog: String = r###"$2 >= "ava" && $2 <= "bwk" || $2 >= "pjw""###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_strcmp.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_strcmp1() {
   let prog: String = r###"$1 != 1 && $1 != 2 && $1 != 3 && $1 != 4 && $1 != 5"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_strcmp1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_strnum() {
   let prog: String = r###"BEGIN { print 1E2 "", 12e-2 "", e12 "", 1.23456789 "" }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_strnum.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_sub0() {
   let prog: String = r###"
{sub(/[aeiou]/, "foo"); print}
{sub("[aeiou]", "foo"); print}

{gsub(/[aeiou]/, "foo"); print}
{gsub("[aeiou]", "foo"); print}

{sub(/[aeiou]/, "&foo"); print}
{sub("[aeiou]", "&foo"); print}

{gsub(/[aeiou]/, "&foo"); print}
{gsub("[aeiou]", "&foo"); print}

{sub(/[aeiou]/, "\&foo"); print}
{sub("[aeiou]", "\&foo"); print}

{gsub(/[aeiou]/, "\&foo"); print}
{gsub("[aeiou]", "\&foo"); print}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_sub0.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_sub1() {
   let prog: String = r###"{sub(/.$/,"x"); print}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_sub1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_sub2() {
   let prog: String = r###"{sub(/.$/,"&&"); print}
{sub(/.$/,"&\\&&"); print}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_sub2.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_sub3() {
   let prog: String = r###"length($1) {sub(substr($1,1,1),"(&)"); print}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_sub3.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_substr() {
   let prog: String = r###"substr($2, 1, 1) ~ /[abc]/
substr($2, length($2)) !~ /[a-z]/
substr($2, length($2)) ~ /./"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_substr.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_substr1() {
   let prog: String = r###"NR % 2 { print substr($0, 0, -1) }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_substr1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_time() {
   let prog: String = r###"BEGIN {
	FS = "-"
}
/sh$/ {
	n++
	l = length($NF)
	s += l
	ck %= l
	totck += ck
	print
}
END {
	if (n > 0) {
		printf "%d %d %d %fn\n", totck, n, s, s/n
	}
	else
		print "n is zero"
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_time.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_vf() {
   let prog: String = r###"BEGIN { i = 1 }
{print $(i+i)}
{print $(1)}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_vf.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_vf1() {
   let prog: String = r###"{	print
	i = 1
	while (i <= NF) {
		print "	" $i
		i = i + 1
	}
}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_vf1.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_vf2() {
   let prog: String = r###"{ print $NF++; print $NF }"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_vf2.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_vf3() {
   let prog: String = r###"BEGIN { i=1; j=2 }
{$i = $j; print}"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_vf3.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

#[test]
fn t_x() {
   let prog: String = r###"/x/"###.into();
   let tmpdir = tempdir().unwrap();
   let data_fname = tmpdir.path().join("test.data");
   {
      let mut file = File::create(data_fname.clone()).unwrap();
      write!(file, "{}", TEST_DATA).unwrap();
   }
   let data_str = data_fname.clone().into_os_string().into_string().unwrap();
   let expected: String = String::from(include_str!("test_output/t_x.txt"));
   for backend_arg in BACKEND_ARGS {
   Command::cargo_bin("frawk").unwrap().arg(String::from(*backend_arg)).arg(prog.clone())
   .arg(data_str.clone())
   .arg(data_str.clone())
   .assert().stdout(expected.clone());
   }
}

const TEST_DATA: &'static str = r#"/dev/rrp3:

17379	mel
16693	bwk	me
16116	ken	him	someone else
15713	srb
11895	lem
10409	scj
10252	rhm
 9853	shen
 9748	a68
 9492	sif
 9190	pjw
 8912	nls
 8895	dmr
 8491	cda
 8372	bs
 8252	llc
 7450	mb
 7360	ava
 7273	jrv
 7080	bin
 7063	greg
 6567	dict
 6462	lck
 6291	rje
 6211	lwf
 5671	dave
 5373	jhc
 5220	agf
 5167	doug
 5007	valerie
 3963	jca
 3895	bbs
 3796	moh
 3481	xchar
 3200	tbl
 2845	s
 2774	tgs
 2641	met
 2566	jck
 2511	port
 2479	sue
 2127	root
 1989	bsb
 1989	jeg
 1933	eag
 1801	pdj
 1590	tpc
 1385	cvw
 1370	rwm
 1316	avg
 1205	eg
 1194	jam
 1153	dl
 1150	lgm
 1031	cmb
 1018	jwr
  950	gdb
  931	marc
  898	usg
  865	ggr
  822	daemon
  803	mihalis
  700	honey
  624	tad
  559	acs
  541	uucp
  523	raf
  495	adh
  456	kec
  414	craig
  386	donmac
  375	jj
  348	ravi
  344	drw
  327	stars
  288	mrg
  272	jcb
  263	ralph
  253	tom
  251	sjb
  248	haight
  224	sharon
  222	chuck
  213	dsj
  201	bill
  184	god
  176	sys
  166	meh
  163	jon
  144	dan
  143	fox
  123	dale
  116	kab
   95	buz
   80	asc
   79	jas
   79	trt
   64	wsb
   62	dwh
   56	ktf
   54	lr
   47	dlc
   45	dls
   45	jwf
   44	mash
   43	ars
   43	vgl
   37	jfo
   32	rab
   31	pd
   29	jns
   25	spm
   22	rob
   15	egb
   10	hm
   10	mhb
    6	aed
    6	cpb
    5	evp
    4	ber
    4	men
    4	mitch
    3	ast
    3	jfr
    3	lax
    3	nel
    2	blue
    2	jfk
    2	njas
    1	122sec
    1	ddwar
    1	gopi
    1	jk
    1	learn
    1	low
    1	nac
    1	sidor
1root:EMpNB8Zp56:0:0:Super-User,,,,,,,:/:/bin/sh
2roottcsh:*:0:0:Super-User running tcsh [cbm]:/:/bin/tcsh
3sysadm:*:0:0:System V Administration:/usr/admin:/bin/sh
4diag:*:0:996:Hardware Diagnostics:/usr/diags:/bin/csh
5daemon:*:1:1:daemons:/:/bin/sh
6bin:*:2:2:System Tools Owner:/bin:/dev/null
7nuucp:BJnuQbAo:6:10:UUCP.Admin:/usr/spool/uucppublic:/usr/lib/uucp/uucico
8uucp:*:3:5:UUCP.Admin:/usr/lib/uucp:
9sys:*:4:0:System Activity Owner:/usr/adm:/bin/sh
10adm:*:5:3:Accounting Files Owner:/usr/adm:/bin/sh
11lp:*:9:9:Print Spooler Owner:/var/spool/lp:/bin/sh
12auditor:*:11:0:Audit Activity Owner:/auditor:/bin/sh
13dbadmin:*:12:0:Security Database Owner:/dbadmin:/bin/sh
14bootes:dcon:50:1:Tom Killian (DO NOT REMOVE):/tmp:
15cdjuke:dcon:51:1:Tom Killian (DO NOT REMOVE):/tmp:
16rfindd:*:66:1:Rfind Daemon and Fsdump:/var/rfindd:/bin/sh
17EZsetup:*:992:998:System Setup:/var/sysadmdesktop/EZsetup:/bin/csh
18demos:*:993:997:Demonstration User:/usr/demos:/bin/csh
19tutor:*:994:997:Tutorial User:/usr/tutor:/bin/csh
20tour:*:995:997:IRIS Space Tour:/usr/people/tour:/bin/csh
21guest:nfP4/Wpvio/Rw:998:998:Guest Account:/usr/people/guest:/bin/csh
224Dgifts:0nWRTZsOMt.:999:998:4Dgifts Account:/usr/people/4Dgifts:/bin/csh
23nobody:*:60001:60001:SVR4 nobody uid:/dev/null:/dev/null
24noaccess:*:60002:60002:uid no access:/dev/null:/dev/null
25nobody:*:-2:-2:original nobody uid:/dev/null:/dev/null
26rje:*:8:8:RJE Owner:/usr/spool/rje:
27changes:*:11:11:system change log:/:
28dist:sorry:9999:4:file distributions:/v/adm/dist:/v/bin/sh
29man:*:99:995:On-line Manual Owner:/:
30phoneca:*:991:991:phone call log [tom]:/v/adm/log:/v/bin/sh
1r oot EMpNB8Zp56 0 0 Super-User,,,,,,, / /bin/sh
2r oottcsh * 0 0 Super-User running tcsh [cbm] / /bin/tcsh
3s ysadm * 0 0 System V Administration /usr/admin /bin/sh
4d iag * 0 996 Hardware Diagnostics /usr/diags /bin/csh
5d aemon * 1 1 daemons / /bin/sh
6b in * 2 2 System Tools Owner /bin /dev/null
7n uucp BJnuQbAo 6 10 UUCP.Admin /usr/spool/uucppublic /usr/lib/uucp/uucico
8u ucp * 3 5 UUCP.Admin /usr/lib/uucp 
9s ys * 4 0 System Activity Owner /usr/adm /bin/sh
10 adm * 5 3 Accounting Files Owner /usr/adm /bin/sh
11 lp * 9 9 Print Spooler Owner /var/spool/lp /bin/sh
12 auditor * 11 0 Audit Activity Owner /auditor /bin/sh
13 dbadmin * 12 0 Security Database Owner /dbadmin /bin/sh
14 bootes dcon 50 1 Tom Killian (DO NOT REMOVE) /tmp 
15 cdjuke dcon 51 1 Tom Killian (DO NOT REMOVE) /tmp 
16 rfindd * 66 1 Rfind Daemon and Fsdump /var/rfindd /bin/sh
17 EZsetup * 992 998 System Setup /var/sysadmdesktop/EZsetup /bin/csh
18 demos * 993 997 Demonstration User /usr/demos /bin/csh
19 tutor * 994 997 Tutorial User /usr/tutor /bin/csh
20 tour * 995 997 IRIS Space Tour /usr/people/tour /bin/csh
21 guest nfP4/Wpvio/Rw 998 998 Guest Account /usr/people/guest /bin/csh
22 4Dgifts 0nWRTZsOMt. 999 998 4Dgifts Account /usr/people/4Dgifts /bin/csh
23 nobody * 60001 60001 SVR4 nobody uid /dev/null /dev/null
24 noaccess * 60002 60002 uid no access /dev/null /dev/null
25 nobody * -2 -2 original nobody uid /dev/null /dev/null
26 rje * 8 8 RJE Owner /usr/spool/rje 
27 changes * 11 11 system change log / 
28 dist sorry 9999 4 file distributions /v/adm/dist /v/bin/sh
29 man * 99 995 On-line Manual Owner / 
30 phoneca * 991 991 phone call log [tom] /v/adm/log /v/bin/sh
"#;
