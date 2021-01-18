use assert_cmd::Command;
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

#[cfg(feature = "llvm_backend")]
const BACKEND_ARGS: &'static [&'static str] = &["-binterp", "-bllvm", "-bcranelift"];
#[cfg(not(feature = "llvm_backend"))]
const BACKEND_ARGS: &'static [&'static str] = &["-binterp", "-bcranelift"];

// A simple function that looks for the "constant folded" regex instructions in the generated
// output. This is a function that is possible to fool: test cases should be mindful of how it is
// implemented to ensure it is testing what is intended.
//
// We don't build this without llvm at the moment because we only fold constants on higher
// optimization levels.
#[cfg(feature = "llvm_backend")]
fn assert_folded(p: &str) {
    let prog: String = p.into();
    let out = String::from_utf8(
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(prog.clone())
            .arg(String::from("--dump-bytecode"))
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap();
    assert!(out.contains("MatchConst"))
}

// Compare two byte slices, up to reordering the lines of each.
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
fn constant_regex_folded() {
    // NB: 'function unused()' forces `x` to be global
    let prog: String = r#"function unused() { print x; }
BEGIN {
    x = "hi";
    x=ARGV[1];
    print("h" ~ x);
}"#
    .into();
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(prog.clone())
            .arg(String::from("h"))
            .assert()
            .stdout(String::from("1\n"));
    }
    #[cfg(feature = "llvm_backend")]
    {
        assert_folded(
            r#"function unused() { print x; }
    BEGIN { x = "hi"; print("h" ~ x); }"#,
        );
        assert_folded(r#"BEGIN { x = "hi"; x = "there"; print("h" ~ x); }"#);
    }
}

#[test]
fn simple_fi() {
    let input = r#"Item,Count
    carrots,2
    potato chips,3
    custard,1"#;
    let expected = "6.0 3\n";

    let tmpdir = tempdir().unwrap();
    let data_fname = tmpdir.path().join("numbers");
    {
        let mut file = File::create(data_fname.clone()).unwrap();
        file.write(input.as_bytes()).unwrap();
    }
    let prog: String = r#"{n+=$FI["Count"]} END { print n, NR; }"#.into();
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from("-icsv"))
            .arg(String::from("-H"))
            .arg(prog.clone())
            .arg(data_fname.clone().into_os_string().into_string().unwrap())
            .assert()
            .stdout(expected.clone());
    }
}

#[test]
fn mixed_map() {
    let expected = "hi 0.0 5\n1 1.0 3\n";
    let prog: String = r#"BEGIN {
m[1]=2
m["1"]++
m["hi"]=5
for (k in m) {
    print k,k+0,  m[k]
}}"#
    .into();
    for backend_arg in BACKEND_ARGS {
        let output = Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(prog.clone())
            .output()
            .unwrap()
            .stdout;
        unordered_output_equals(expected.as_bytes(), &output[..]);
    }
}

#[test]
fn iter_across_functions() {
    let input = ",,3,,4\n,,3,,6\n,,4,,5";
    let expected = "3 62.0\n4 30.0\n";

    let tmpdir = tempdir().unwrap();
    let data_fname = tmpdir.path().join("numbers");
    {
        let mut file = File::create(data_fname.clone()).unwrap();
        file.write(input.as_bytes()).unwrap();
    }
    let prog: String = r#"function update(h, k, v) {
            h[k] += v*v+v;
        }
        BEGIN {FS=",";}
        { update(h,$3,$5) }
        END {for (k in h) { print k, h[k]; }}"#
        .into();
    for backend_arg in BACKEND_ARGS {
        let output = Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(prog.clone())
            .arg(data_fname.clone().into_os_string().into_string().unwrap())
            .output()
            .unwrap()
            .stdout;
        unordered_output_equals(expected.as_bytes(), &output[..]);
    }
}

#[test]
fn nested_loops() {
    let expected = "0 0\n0 1\n0 2\n1 0\n1 1\n1 2\n2 0\n2 1\n2 2\n";
    let prog: String =
        "BEGIN { m[0]=0; m[1]=1; m[2]=2; for (i in m) for (j in m) print i,j; }".into();
    for backend_arg in BACKEND_ARGS {
        let output = Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(prog.clone())
            .output()
            .unwrap()
            .stdout;
        unordered_output_equals(expected.as_bytes(), &output[..]);
    }
}
