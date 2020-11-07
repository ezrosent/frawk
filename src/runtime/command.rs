use std::io;
use std::process::{ChildStdin, ChildStdout, Command, Stdio};

use crate::runtime::Int;

fn prepare_command(bs: &[u8]) -> io::Result<Command> {
    let prog = match std::str::from_utf8(bs) {
        Ok(s) => s,
        Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidInput, e)),
    };
    if cfg!(target_os = "windows") {
        let mut cmd = Command::new("cmd");
        cmd.args(&["/C", prog]);
        Ok(cmd)
    } else {
        let mut cmd = Command::new("sh");
        cmd.args(&["-c", prog]);
        Ok(cmd)
    }
}

pub fn run_command(bs: &[u8]) -> Int {
    fn wrap_err(e: Option<i32>) -> Int {
        e.map(Int::from).unwrap_or(1)
    }
    fn run_command_inner(bs: &[u8]) -> io::Result<Int> {
        let status = prepare_command(bs)?.status()?;
        Ok(wrap_err(status.code()))
    }
    match run_command_inner(bs) {
        Ok(i) => i,
        Err(e) => wrap_err(e.raw_os_error()),
    }
}

pub fn command_for_write(bs: &[u8]) -> io::Result<ChildStdin> {
    let mut cmd = prepare_command(bs)?;
    let mut child = cmd.stdin(Stdio::piped()).stdout(Stdio::inherit()).spawn()?;
    Ok(child.stdin.take().unwrap())
}

pub fn command_for_read(bs: &[u8]) -> io::Result<ChildStdout> {
    let mut cmd = prepare_command(bs)?;
    let mut child = cmd.stdin(Stdio::inherit()).stdout(Stdio::piped()).spawn()?;
    Ok(child.stdout.take().unwrap())
}
