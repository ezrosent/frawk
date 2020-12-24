use assert_cmd::Command;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

fn numbers_str(n: usize) -> (String, String) {
    let mut input = String::new();
    let mut sorted = String::new();
    let mut vs: Vec<_> = (0..n).collect();
    vs.shuffle(&mut thread_rng());
    for (i, n) in vs.into_iter().enumerate() {
        sorted.push_str(format!("{}\n", i).as_str());
        input.push_str(format!("{}\n", n).as_str());
    }
    (input, sorted)
}

const N: usize = 10_000;

#[cfg(feature = "llvm_backend")]
const BACKEND_ARGS: &'static [&'static str] = &["-b", "-O3"];
#[cfg(not(feature = "llvm_backend"))]
const BACKEND_ARGS: &'static [&'static str] = &["-b"];

#[cfg(not(target_os = "windows"))]
#[test]
fn sort_command_single_threaded() {
    let (input, expected) = numbers_str(N);
    let tmpdir = tempdir().unwrap();
    let data_fname = tmpdir.path().join("numbers");
    {
        let mut file = File::create(data_fname.clone()).unwrap();
        file.write(input.as_bytes()).unwrap();
    }
    let prog: String = r#"{ print $0 | "sort -n"; }"#.into();
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(prog.clone())
            .arg(data_fname.clone().into_os_string().into_string().unwrap())
            .assert()
            .stdout(expected.clone());
    }
}

#[cfg(not(target_os = "windows"))]
#[test]
fn sort_command_multi_threaded() {
    let (input, expected) = numbers_str(N);
    let tmpdir = tempdir().unwrap();
    let data_fname = tmpdir.path().join("numbers");
    {
        let mut file = File::create(data_fname.clone()).unwrap();
        file.write(input.as_bytes()).unwrap();
    }
    let prog: String = r#"{ print $0 | "sort -n"; }"#.into();
    for backend_arg in BACKEND_ARGS {
        Command::cargo_bin("frawk")
            .unwrap()
            .arg(String::from(*backend_arg))
            .arg(String::from("-pr"))
            .arg(String::from("-j4"))
            .arg(prog.clone())
            .arg(data_fname.clone().into_os_string().into_string().unwrap())
            .assert()
            .stdout(expected.clone());
    }
}
