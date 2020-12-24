//! A translation of the `p.` tests in the test directory in the [one true awk
//! repo](https://github.com/onetrueawk/awk/tree/master/testdir) into Rust, modified for
//! differences in frawk behavior/semantics.
//!
//! Those changes are:
//! * A frawk parsing limitation leads `FNR==1, FNR==5` not to parse; we need parens around the
//! comparisons
//! * `length` is not syntactic sugar for `length($0)`.
//! * frawk prints more digits on floating point values by default.
//! * frawk's parser requires semicolons between a last statement and a `}` sometimes
//! * frawk's rules aronud comparing strings and numbers are different, e.g. "Russia < 1" is true
//!   in frawk but false in awk/mawk.

use assert_cmd::Command;
use std::fs::{read_to_string, File};
use std::io::Write;
use tempfile::tempdir;

#[cfg(feature = "llvm_backend")]
const BACKEND_ARGS: &'static [&'static str] = &["-b", "-O3"];
#[cfg(not(feature = "llvm_backend"))]
const BACKEND_ARGS: &'static [&'static str] = &["-b"];

const COUNTRIES: &'static str = r#"Russia	8650	262	Asia
Canada	3852	24	North America
China	3692	866	Asia
USA	3615	219	North America
Brazil	3286	116	South America
Australia	2968	14	Australia
India	1269	637	Asia
Argentina	1072	26	South America
Sudan	968	19	Africa
Algeria	920	18	Africa
"#;

fn unordered_output_equals(bs1: &[u8], bs2: &[u8]) {
    let mut lines1: Vec<_> = bs1.split(|x| *x == b'\n').collect();
    let mut lines2: Vec<_> = bs2.split(|x| *x == b'\n').collect();
    lines1.sort();
    lines2.sort();
    if lines1 != lines2 {
        let pretty_1: Vec<_> = lines1.into_iter().map(String::from_utf8_lossy).collect();
        let pretty_2: Vec<_> = lines2.into_iter().map(String::from_utf8_lossy).collect();
        assert!(
            false,
            "expected (in any order) {:?}, got {:?}",
            pretty_1, pretty_2
        );
    }
}

#[test]
fn p_test_1() {
    let expected = String::from(
        r#"Russia	8650	262	Asia
Canada	3852	24	North America
China	3692	866	Asia
USA	3615	219	North America
Brazil	3286	116	South America
Australia	2968	14	Australia
India	1269	637	Asia
Argentina	1072	26	South America
Sudan	968	19	Africa
Algeria	920	18	Africa
Russia	8650	262	Asia
Canada	3852	24	North America
China	3692	866	Asia
USA	3615	219	North America
Brazil	3286	116	South America
Australia	2968	14	Australia
India	1269	637	Asia
Argentina	1072	26	South America
Sudan	968	19	Africa
Algeria	920	18	Africa
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"{ print }"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_10() {
    let expected = String::from(
        r#"Australia	2968	14	Australia
Australia	2968	14	Australia
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"$1 == $4"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_11() {
    let expected = String::from(
        r#"Russia	8650	262	Asia
China	3692	866	Asia
India	1269	637	Asia
Russia	8650	262	Asia
China	3692	866	Asia
India	1269	637	Asia
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"/Asia/"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_12() {
    let expected = String::from(
        r#"Russia
China
India
Russia
China
India
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"$4 ~ /Asia/ { print $1 }"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_13() {
    let expected = String::from(
        r#"Canada
USA
Brazil
Australia
Argentina
Sudan
Algeria
Canada
USA
Brazil
Australia
Argentina
Sudan
Algeria
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"$4 !~ /Asia/ {print $1 }"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_14() {
    let expected = String::from(r#""#);
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"/\$/"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_15() {
    let expected = String::from(r#""#);
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"/\\/"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_16() {
    let expected = String::from(r#""#);
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"/^.$/"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_17() {
    let expected = String::from(r#""#);
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"$2 !~ /^[0-9]+$/"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_18() {
    let expected = String::from(r#""#);
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"/(apple|cherry) (pie|tart)/"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_19() {
    let expected = String::from(r#""#);
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"BEGIN	{ digits = "^[0-9]+$" }
$2 !~ digits"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_2() {
    let expected = String::from(
        r#"Russia 262
Canada 24
China 866
USA 219
Brazil 116
Australia 14
India 637
Argentina 26
Sudan 19
Algeria 18
Russia 262
Canada 24
China 866
USA 219
Brazil 116
Australia 14
India 637
Argentina 26
Sudan 19
Algeria 18
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"{ print $1, $3 }"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_20() {
    let expected = String::from(
        r#"China	3692	866	Asia
India	1269	637	Asia
China	3692	866	Asia
India	1269	637	Asia
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"$4 == "Asia" && $3 > 500"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_21() {
    let expected = String::from(
        r#"Russia	8650	262	Asia
China	3692	866	Asia
India	1269	637	Asia
Russia	8650	262	Asia
China	3692	866	Asia
India	1269	637	Asia
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"$4 == "Asia" || $4 == "Europe""#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_21a() {
    let expected = String::from(
        r#"Russia	8650	262	Asia
China	3692	866	Asia
India	1269	637	Asia
Sudan	968	19	Africa
Algeria	920	18	Africa
Russia	8650	262	Asia
China	3692	866	Asia
India	1269	637	Asia
Sudan	968	19	Africa
Algeria	920	18	Africa
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"/Asia/ || /Africa/"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_22() {
    let expected = String::from(
        r#"Russia	8650	262	Asia
China	3692	866	Asia
India	1269	637	Asia
Russia	8650	262	Asia
China	3692	866	Asia
India	1269	637	Asia
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"$4 ~ /^(Asia|Europe)$/"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_23() {
    let expected = String::from(
        r#"Canada	3852	24	North America
China	3692	866	Asia
USA	3615	219	North America
Brazil	3286	116	South America
Canada	3852	24	North America
China	3692	866	Asia
USA	3615	219	North America
Brazil	3286	116	South America
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"/Canada/, /Brazil/"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_24() {
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    let expected = format!(
        r#"{filename} Russia	8650	262	Asia
{filename} Canada	3852	24	North America
{filename} China	3692	866	Asia
{filename} USA	3615	219	North America
{filename} Brazil	3286	116	South America
{filename} Russia	8650	262	Asia
{filename} Canada	3852	24	North America
{filename} China	3692	866	Asia
{filename} USA	3615	219	North America
{filename} Brazil	3286	116	South America
"#,
        filename = data_string.as_str()
    );
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"(FNR == 1), (FNR == 5) { print FILENAME, $0 }"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_25() {
    let expected = String::from(
        r#"    Russia   30.3
    Canada    6.2
     China  234.6
       USA   60.6
    Brazil   35.3
 Australia    4.7
     India  502.0
 Argentina   24.3
     Sudan   19.6
   Algeria   19.6
    Russia   30.3
    Canada    6.2
     China  234.6
       USA   60.6
    Brazil   35.3
 Australia    4.7
     India  502.0
 Argentina   24.3
     Sudan   19.6
   Algeria   19.6
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"{ printf "%10s %6.1f\n", $1, 1000 * $3 / $2 }"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_26() {
    let expected = String::from(
        r#"population of 6 Asian countries in millions is 3530.0
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"/Asia/	{ pop = pop + $3; n = n + 1 }
END	{ print "population of", n,\
		"Asian countries in millions is", pop }"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_26a() {
    let expected = String::from(
        r#"population of 6 Asian countries in millions is 3530.0
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"/Asia/	{ pop += $3; ++n }
END	{ print "population of", n,\
		"Asian countries in millions is", pop }"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_27() {
    let expected = String::from(
        r#"China 866
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"maxpop < $3	{ maxpop = $3; country = $1 }
END		{ print country, maxpop }"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_28() {
    let expected = String::from(
        r#"1:Russia	8650	262	Asia
2:Canada	3852	24	North America
3:China	3692	866	Asia
4:USA	3615	219	North America
5:Brazil	3286	116	South America
6:Australia	2968	14	Australia
7:India	1269	637	Asia
8:Argentina	1072	26	South America
9:Sudan	968	19	Africa
10:Algeria	920	18	Africa
11:Russia	8650	262	Asia
12:Canada	3852	24	North America
13:China	3692	866	Asia
14:USA	3615	219	North America
15:Brazil	3286	116	South America
16:Australia	2968	14	Australia
17:India	1269	637	Asia
18:Argentina	1072	26	South America
19:Sudan	968	19	Africa
20:Algeria	920	18	Africa
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"{ print NR ":" $0 }"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_29() {
    let expected = String::from(
        r#"Russia	8650	262	Asia
Canada	3852	24	North America
China	3692	866	Asia
United States	3615	219	North America
Brazil	3286	116	South America
Australia	2968	14	Australia
India	1269	637	Asia
Argentina	1072	26	South America
Sudan	968	19	Africa
Algeria	920	18	Africa
Russia	8650	262	Asia
Canada	3852	24	North America
China	3692	866	Asia
United States	3615	219	North America
Brazil	3286	116	South America
Australia	2968	14	Australia
India	1269	637	Asia
Argentina	1072	26	South America
Sudan	968	19	Africa
Algeria	920	18	Africa
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"	{ gsub(/USA/, "United States"); print }"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_3() {
    let expected = String::from(
        r#"[    Russia] [262             ]
[    Canada] [24              ]
[     China] [866             ]
[       USA] [219             ]
[    Brazil] [116             ]
[ Australia] [14              ]
[     India] [637             ]
[ Argentina] [26              ]
[     Sudan] [19              ]
[   Algeria] [18              ]
[    Russia] [262             ]
[    Canada] [24              ]
[     China] [866             ]
[       USA] [219             ]
[    Brazil] [116             ]
[ Australia] [14              ]
[     India] [637             ]
[ Argentina] [26              ]
[     Sudan] [19              ]
[   Algeria] [18              ]
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"{ printf "[%10s] [%-16d]\n", $1, $3 }"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_30() {
    let expected = String::from(
        r#"20 Russia	8650	262	Asia
28 Canada	3852	24	North America
19 China	3692	866	Asia
26 USA	3615	219	North America
29 Brazil	3286	116	South America
27 Australia	2968	14	Australia
19 India	1269	637	Asia
31 Argentina	1072	26	South America
19 Sudan	968	19	Africa
21 Algeria	920	18	Africa
20 Russia	8650	262	Asia
28 Canada	3852	24	North America
19 China	3692	866	Asia
26 USA	3615	219	North America
29 Brazil	3286	116	South America
27 Australia	2968	14	Australia
19 India	1269	637	Asia
31 Argentina	1072	26	South America
19 Sudan	968	19	Africa
21 Algeria	920	18	Africa
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"{ print length($0), $0 }"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_31() {
    let expected = String::from(
        r#"Australia
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"length($1) > max	{ max = length($1); name = $1 }
END			{ print name }"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_32() {
    let expected = String::from(
        r#"Rus 8650 262 Asia
Can 3852 24 North America
Chi 3692 866 Asia
USA 3615 219 North America
Bra 3286 116 South America
Aus 2968 14 Australia
Ind 1269 637 Asia
Arg 1072 26 South America
Sud 968 19 Africa
Alg 920 18 Africa
Rus 8650 262 Asia
Can 3852 24 North America
Chi 3692 866 Asia
USA 3615 219 North America
Bra 3286 116 South America
Aus 2968 14 Australia
Ind 1269 637 Asia
Arg 1072 26 South America
Sud 968 19 Africa
Alg 920 18 Africa
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"{ $1 = substr($1, 1, 3); print }"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_33() {
    let expected = String::from(
        r#" Rus Can Chi USA Bra Aus Ind Arg Sud Alg Rus Can Chi USA Bra Aus Ind Arg Sud Alg
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"	{ s = s " " substr($1, 1, 3) }
END	{ print s }"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_34() {
    let expected = String::from(
        r#"Russia 8.65 262 Asia
Canada 3.852 24 North America
China 3.692 866 Asia
USA 3.615 219 North America
Brazil 3.286 116 South America
Australia 2.968 14 Australia
India 1.269 637 Asia
Argentina 1.072 26 South America
Sudan 0.968 19 Africa
Algeria 0.92 18 Africa
Russia 8.65 262 Asia
Canada 3.852 24 North America
China 3.692 866 Asia
USA 3.615 219 North America
Brazil 3.286 116 South America
Australia 2.968 14 Australia
India 1.269 637 Asia
Argentina 1.072 26 South America
Sudan 0.968 19 Africa
Algeria 0.92 18 Africa
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"{ $2 /= 1000; print }"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_35() {
    let expected = String::from(
        r#"Russia	8650	262	Asia
Canada	3852	24	NA
China	3692	866	Asia
USA	3615	219	NA
Brazil	3286	116	SA
Australia	2968	14	Australia
India	1269	637	Asia
Argentina	1072	26	SA
Sudan	968	19	Africa
Algeria	920	18	Africa
Russia	8650	262	Asia
Canada	3852	24	NA
China	3692	866	Asia
USA	3615	219	NA
Brazil	3286	116	SA
Australia	2968	14	Australia
India	1269	637	Asia
Argentina	1072	26	SA
Sudan	968	19	Africa
Algeria	920	18	Africa
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"BEGIN			{ FS = OFS = "\t" }
$4 ~ /^North America$/	{ $4 = "NA" }
$4 ~ /^South America$/	{ $4 = "SA" }
			{ print }"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_36() {
    let expected = String::from(
        r#"Russia	8650	262	Asia	30.289017341040463
Canada	3852	24	North America	6.230529595015576
China	3692	866	Asia	234.56121343445287
USA	3615	219	North America	60.58091286307054
Brazil	3286	116	South America	35.30127814972611
Australia	2968	14	Australia	4.716981132075472
India	1269	637	Asia	501.9700551615445
Argentina	1072	26	South America	24.253731343283583
Sudan	968	19	Africa	19.628099173553718
Algeria	920	18	Africa	19.565217391304348
Russia	8650	262	Asia	30.289017341040463
Canada	3852	24	North America	6.230529595015576
China	3692	866	Asia	234.56121343445287
USA	3615	219	North America	60.58091286307054
Brazil	3286	116	South America	35.30127814972611
Australia	2968	14	Australia	4.716981132075472
India	1269	637	Asia	501.9700551615445
Argentina	1072	26	South America	24.253731343283583
Sudan	968	19	Africa	19.628099173553718
Algeria	920	18	Africa	19.565217391304348
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"BEGIN	{ FS = OFS = "\t" }
	{ $5 = 1000 * $3 / $2 ; print $1, $2, $3, $4, $5 }"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_37() {
    let expected = String::from(r#""#);
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"$1 "" == $2 """#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_38() {
    let expected = String::from(
        r#"China 866
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"{	if (maxpop < $3) {
		maxpop = $3
		country = $1
	}
}
END	{ print country, maxpop }"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_39() {
    let expected = String::from(
        r#"Russia
8650
262
Asia
Canada
3852
24
North
America
China
3692
866
Asia
USA
3615
219
North
America
Brazil
3286
116
South
America
Australia
2968
14
Australia
India
1269
637
Asia
Argentina
1072
26
South
America
Sudan
968
19
Africa
Algeria
920
18
Africa
Russia
8650
262
Asia
Canada
3852
24
North
America
China
3692
866
Asia
USA
3615
219
North
America
Brazil
3286
116
South
America
Australia
2968
14
Australia
India
1269
637
Asia
Argentina
1072
26
South
America
Sudan
968
19
Africa
Algeria
920
18
Africa
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"{	i = 1
	while (i <= NF) {
		print $i
		i++
	}
}"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_4() {
    let expected = String::from(
        r#"1 Russia	8650	262	Asia
2 Canada	3852	24	North America
3 China	3692	866	Asia
4 USA	3615	219	North America
5 Brazil	3286	116	South America
6 Australia	2968	14	Australia
7 India	1269	637	Asia
8 Argentina	1072	26	South America
9 Sudan	968	19	Africa
10 Algeria	920	18	Africa
11 Russia	8650	262	Asia
12 Canada	3852	24	North America
13 China	3692	866	Asia
14 USA	3615	219	North America
15 Brazil	3286	116	South America
16 Australia	2968	14	Australia
17 India	1269	637	Asia
18 Argentina	1072	26	South America
19 Sudan	968	19	Africa
20 Algeria	920	18	Africa
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"{ print NR, $0 }"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_40() {
    let expected = String::from(
        r#"Russia
8650
262
Asia
Canada
3852
24
North
America
China
3692
866
Asia
USA
3615
219
North
America
Brazil
3286
116
South
America
Australia
2968
14
Australia
India
1269
637
Asia
Argentina
1072
26
South
America
Sudan
968
19
Africa
Algeria
920
18
Africa
Russia
8650
262
Asia
Canada
3852
24
North
America
China
3692
866
Asia
USA
3615
219
North
America
Brazil
3286
116
South
America
Australia
2968
14
Australia
India
1269
637
Asia
Argentina
1072
26
South
America
Sudan
968
19
Africa
Algeria
920
18
Africa
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"{	for (i = 1; i <= NF; i++)
		print $i
}"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_41() {
    let expected = String::from(r#""#);
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"NR >= 10	{ exit }
END		{ if (NR < 10)
			print FILENAME " has only " NR " lines" }"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_42() {
    let expected = String::from(
        r#"Asian population in millions is 3530.0
African population in millions is 74.0
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"/Asia/		{ pop["Asia"] += $3 }
/Africa/	{ pop["Africa"] += $3 }
END		{ print "Asian population in millions is", pop["Asia"]
		  print "African population in millions is", pop["Africa"] }"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_43() {
    let expected = String::from(
        r#"Asia:27222.0
Australia:5936.0
Africa:3776.0
South America:8716.0
North America:14934.0
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        let output = Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"BEGIN	{ FS = "\t" }
	{ area[$4] += $2 }
END	{ for (name in area)
		print name ":" area[name]; }"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .output()
            .unwrap()
            .stdout;
        unordered_output_equals(expected.as_bytes(), &output[..]);
    }
}

#[test]
fn p_test_44() {
    let expected = String::from(
        r#"Russia! is 1.0
Canada! is 1.0
China! is 1.0
USA! is 1.0
Brazil! is 1.0
Australia! is 1.0
India! is 1.0
Argentina! is 1.0
Sudan! is 1.0
Algeria! is 1.0
Russia! is 1.0
Canada! is 1.0
China! is 1.0
USA! is 1.0
Brazil! is 1.0
Australia! is 1.0
India! is 1.0
Argentina! is 1.0
Sudan! is 1.0
Algeria! is 1.0
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"function fact(n) {
	if (n <= 1)
		return 1
	else
		return n * fact(n-1)
}
{ print $1 "! is " fact($1) }"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_45() {
    let expected = String::from(
        r#"Russia:8650

Canada:3852

China:3692

USA:3615

Brazil:3286

Australia:2968

India:1269

Argentina:1072

Sudan:968

Algeria:920

Russia:8650

Canada:3852

China:3692

USA:3615

Brazil:3286

Australia:2968

India:1269

Argentina:1072

Sudan:968

Algeria:920

"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"BEGIN	{ OFS = ":" ; ORS = "\n\n" }
	{ print $1, $2 }"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_46() {
    let expected = String::from(
        r#"Russia8650
Canada3852
China3692
USA3615
Brazil3286
Australia2968
India1269
Argentina1072
Sudan968
Algeria920
Russia8650
Canada3852
China3692
USA3615
Brazil3286
Australia2968
India1269
Argentina1072
Sudan968
Algeria920
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"	{ print $1 $2 }"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_47() {
    let expected = String::from(r#""#);
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let small_path = tmpdir.path().join("tempsmall");
    let big_path = tmpdir.path().join("tempbig");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    let small = small_path.clone().into_os_string().into_string().unwrap();
    let big = big_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(format!(
                r#"$3 > 100	{{ print >"{tempbig}" }}
$3 <= 100	{{ print >"{tempsmall}" }}"#,
                tempsmall = small,
                tempbig = big,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
        let got_small = read_to_string(small_path.clone()).unwrap();
        assert_eq!(
            got_small,
            r#"Canada	3852	24	North America
Australia	2968	14	Australia
Argentina	1072	26	South America
Sudan	968	19	Africa
Algeria	920	18	Africa
Canada	3852	24	North America
Australia	2968	14	Australia
Argentina	1072	26	South America
Sudan	968	19	Africa
Algeria	920	18	Africa
"#
        );
        let got_big = read_to_string(big_path.clone()).unwrap();
        assert_eq!(
            got_big,
            r#"Russia	8650	262	Asia
China	3692	866	Asia
USA	3615	219	North America
Brazil	3286	116	South America
India	1269	637	Asia
Russia	8650	262	Asia
China	3692	866	Asia
USA	3615	219	North America
Brazil	3286	116	South America
India	1269	637	Asia
"#
        );
    }
}

#[cfg(not(target_os = "windows"))]
#[test]
fn p_test_48() {
    let expected = String::from(
        r#"Africa:74.0
Asia:3530.0
Australia:28.0
North America:486.0
South America:284.0
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"BEGIN	{ FS = "\t" }
	{ pop[$4] += $3 }
END	{ for (c in pop)
		print c ":" pop[c] | "sort"; }"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_48a() {
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    let expected = format!("{filename} {filename} \n", filename = data_string.as_str());
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"BEGIN {
	for (i = 1; i < ARGC; i++)
		printf "%s ", ARGV[i]
	printf "\n"
	exit
}"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_48b() {
    let expected = String::from(
        r#"Russia	8650	262	Asia
USA	3615	219	North America
India	1269	637	Asia
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"BEGIN	{ srand(10); k = 3; n = 10 }
{	if (n <= 0) exit
	if (rand() <= k/n) { print; k-- }
	n--
}"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_49() {
    let expected = String::from(r#""#);
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"$1 == "include" { system("cat " $2) }"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_5() {
    let expected = String::from(
        r#"   COUNTRY   AREA   POP       CONTINENT
    Russia   8650   262            Asia
    Canada   3852    24   North America
     China   3692   866            Asia
       USA   3615   219   North America
    Brazil   3286   116   South America
 Australia   2968    14       Australia
     India   1269   637            Asia
 Argentina   1072    26   South America
     Sudan    968    19          Africa
   Algeria    920    18          Africa
    Russia   8650   262            Asia
    Canada   3852    24   North America
     China   3692   866            Asia
       USA   3615   219   North America
    Brazil   3286   116   South America
 Australia   2968    14       Australia
     India   1269   637            Asia
 Argentina   1072    26   South America
     Sudan    968    19          Africa
   Algeria    920    18          Africa
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"BEGIN	{ FS = "\t"
	  printf "%10s %6s %5s %15s\n", "COUNTRY", "AREA", "POP", "CONTINENT" }
	{ printf "%10s %6d %5d %15s\n", $1, $2, $3, $4 }"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[cfg(not(target_os = "windows"))]
#[test]
fn p_test_50() {
    let expected = String::from(
        r#"Africa:Sudan:38.0
Africa:Algeria:36.0
Asia:China:1732.0
Asia:India:1274.0
Asia:Russia:524.0
Australia:Australia:28.0
North America:USA:438.0
North America:Canada:48.0
South America:Brazil:232.0
South America:Argentina:52.0
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"BEGIN	{ FS = "\t" }
	{ pop[$4 ":" $1] += $3 }
END	{ for (cc in pop)
		print cc ":" pop[cc] | "sort -t: -k 1,1 -k 3nr"; }"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_51() {
    let expected = String::from(
        r#"
Russia	8650	262	Asia:
	                0

Canada	3852	24	North America:
	                0

China	3692	866	Asia:
	                0

USA	3615	219	North America:
	                0

Brazil	3286	116	South America:
	                0

Australia	2968	14	Australia:
	                0

India	1269	637	Asia:
	                0

Argentina	1072	26	South America:
	                0

Sudan	968	19	Africa:
	                0

Algeria	920	18	Africa:
	                0

Russia	8650	262	Asia:
	                0

Canada	3852	24	North America:
	                0

China	3692	866	Asia:
	                0

USA	3615	219	North America:
	                0

Brazil	3286	116	South America:
	                0

Australia	2968	14	Australia:
	                0

India	1269	637	Asia:
	                0

Argentina	1072	26	South America:
	                0

Sudan	968	19	Africa:
	                0

Algeria	920	18	Africa:
	                0
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"BEGIN	{ FS = ":" }
{	if ($1 != prev) {
		print "\n" $1 ":"
		prev = $1
	}
	printf "\t%-10s %6d\n", $2, $3
}"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_52() {
    let expected = String::from(
        r#"
Russia	8650	262	Asia:
	                0
	total     	      0

Canada	3852	24	North America:
	                0
	total     	      0

China	3692	866	Asia:
	                0
	total     	      0

USA	3615	219	North America:
	                0
	total     	      0

Brazil	3286	116	South America:
	                0
	total     	      0

Australia	2968	14	Australia:
	                0
	total     	      0

India	1269	637	Asia:
	                0
	total     	      0

Argentina	1072	26	South America:
	                0
	total     	      0

Sudan	968	19	Africa:
	                0
	total     	      0

Algeria	920	18	Africa:
	                0
	total     	      0

Russia	8650	262	Asia:
	                0
	total     	      0

Canada	3852	24	North America:
	                0
	total     	      0

China	3692	866	Asia:
	                0
	total     	      0

USA	3615	219	North America:
	                0
	total     	      0

Brazil	3286	116	South America:
	                0
	total     	      0

Australia	2968	14	Australia:
	                0
	total     	      0

India	1269	637	Asia:
	                0
	total     	      0

Argentina	1072	26	South America:
	                0
	total     	      0

Sudan	968	19	Africa:
	                0
	total     	      0

Algeria	920	18	Africa:
	                0
	total     	      0

World Total		      0
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"BEGIN	{ FS = ":" }
{
	if ($1 != prev) {
		if (prev) {
			printf "\t%-10s\t %6d\n", "total", subtotal
			subtotal = 0
		}
		print "\n" $1 ":"
		prev = $1
	}
	printf "\t%-10s %6d\n", $2, $3
	wtotal += $3
	subtotal += $3
}
END	{ printf "\t%-10s\t %6d\n", "total", subtotal
	  printf "\n%-10s\t\t %6d\n", "World Total", wtotal }"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_5a() {
    let expected = String::from(
        r#"   COUNTRY	  AREA	 POP'N	      CONTINENT
    Russia	  8650	   262	           Asia
    Canada	  3852	    24	  North America
     China	  3692	   866	           Asia
       USA	  3615	   219	  North America
    Brazil	  3286	   116	  South America
 Australia	  2968	    14	      Australia
     India	  1269	   637	           Asia
 Argentina	  1072	    26	  South America
     Sudan	   968	    19	         Africa
   Algeria	   920	    18	         Africa
    Russia	  8650	   262	           Asia
    Canada	  3852	    24	  North America
     China	  3692	   866	           Asia
       USA	  3615	   219	  North America
    Brazil	  3286	   116	  South America
 Australia	  2968	    14	      Australia
     India	  1269	   637	           Asia
 Argentina	  1072	    26	  South America
     Sudan	   968	    19	         Africa
   Algeria	   920	    18	         Africa
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"BEGIN	{ FS = "\t"
	  printf "%10s\t%6s\t%6s\t%15s\n", "COUNTRY", "AREA", "POP'N", "CONTINENT"}
	{ printf "%10s\t%6d\t%6d\t%15s\n", $1, $2, $3, $4}"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_6() {
    let expected = String::from(
        r#"20
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"END	{ print NR }"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_7() {
    let expected = String::from(
        r#"Russia	8650	262	Asia
China	3692	866	Asia
USA	3615	219	North America
Brazil	3286	116	South America
India	1269	637	Asia
Russia	8650	262	Asia
China	3692	866	Asia
USA	3615	219	North America
Brazil	3286	116	South America
India	1269	637	Asia
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"$3 > 100"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_8() {
    let expected = String::from(
        r#"Russia
China
India
Russia
China
India
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"$4 == "Asia" { print $1 }"#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_9() {
    let expected = String::from(
        r#"USA	3615	219	North America
Sudan	968	19	Africa
USA	3615	219	North America
Sudan	968	19	Africa
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(r#"$1 >= "S""#))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn p_test_table() {
    let expected = String::from(
        r#"Russia      8650   262   Asia         
Canada      3852    24   North America
China       3692   866   Asia         
USA         3615   219   North America
Brazil      3286   116   South America
Australia   2968    14   Australia    
India       1269   637   Asia         
Argentina   1072    26   South America
Sudan        968    19   Africa       
Algeria      920    18   Africa       
Russia      8650   262   Asia         
Canada      3852    24   North America
China       3692   866   Asia         
USA         3615   219   North America
Brazil      3286   116   South America
Australia   2968    14   Australia    
India       1269   637   Asia         
Argentina   1072    26   South America
Sudan        968    19   Africa       
Algeria      920    18   Africa       
"#,
    );
    let tmpdir = tempdir().unwrap();
    let data_path = tmpdir.path().join("test.countries");
    let data_string = data_path.clone().into_os_string().into_string().unwrap();
    {
        let mut file = File::create(data_path).unwrap();
        write!(file, "{}", COUNTRIES).unwrap();
    }
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from(
                r#"# table - simple table formatter

BEGIN {
    FS = "\t"; blanks = sprintf("%100s", " ")
    number = "^[+-]?([0-9]+[.]?[0-9]*|[.][0-9]+)$"
}

{   row[NR] = $0
    for (i = 1; i <= NF; i++) {
        if ($i ~ number)
            nwid[i] = max(nwid[i], length($i))
        wid[i] = max(wid[i], length($i))
    }
}

END {
    for (r = 1; r <= NR; r++) {
        n = split(row[r], d)
        for (i = 1; i <= n; i++) {
            sep = (i < n) ? "   " : "\n"
            if (d[i] ~ number)
                printf("%" wid[i] "s%s", numjust(i,d[i]), sep)
            else
                printf("%-" wid[i] "s%s", d[i], sep)
        }
    }
}

function max(x, y) { return (x > y) ? x : y }

function numjust(n, s) {   # position s in field n
    return s substr(blanks, 1, int((wid[n]-nwid[n])/2))
}"#,
            ))
            .arg(data_string.clone())
            .arg(data_string.clone())
            .assert()
            .stdout(expected.clone());
    }
}
