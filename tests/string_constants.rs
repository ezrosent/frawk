use assert_cmd::Command;
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

#[cfg(feature = "llvm_backend")]
const BACKEND_ARGS: &'static [&'static str] = &["-b", "-O3"];
#[cfg(not(feature = "llvm_backend"))]
const BACKEND_ARGS: &'static [&'static str] = &["-b"];

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
