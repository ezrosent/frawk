use std::io;
use std::process::{ChildStdin, Command, Stdio};

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

pub fn command_for_write(bs: &[u8]) -> io::Result<ChildStdin> {
    let mut cmd = prepare_command(bs)?;
    let mut child = cmd.stdin(Stdio::piped()).stdout(Stdio::inherit()).spawn()?;
    Ok(child.stdin.take().unwrap())
}

pub fn command_for_read(bs: &[u8]) -> io::Result<impl io::Read> {
    let mut cmd = prepare_command(bs)?;
    let mut child = cmd.stdin(Stdio::inherit()).stdout(Stdio::piped()).spawn()?;
    Ok(child.stdout.take().unwrap())
}
