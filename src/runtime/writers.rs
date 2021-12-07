//! Writing data to output files
//!
//! This module has the following goals:
//!
//! 1. Support writes to a single file from multiple threads.
//! 2. Support batching writes both within a single thread and also across threads.
//!
//! Supporting all of these requirements, along with the ability to inject fakes for the file
//! system, leads to a fairly involved implementation. Callers build a Registry, which acts as a
//! thread-local cache of a shared mapping from file name to file handle. The core of the
//! implementation is in the implementation of these handles.
//!
//! File handles are each "clients" to a single thread issuing writes on their behalf. That thread
//! reads requests to write, flush, or even close that file and issues them in order. When the
//! thread receives adjacent write requests, it uses the write_vectored API to issue all of those
//! writes at once. This grants us nice batching semantics a la BufWriter without the additional
//! copies.
//!
//! (Aside: "thread per file" might become expensive if we want to support workloads with thousands
//! of open output files. In that case, we could replace each of these background threads with a
//! "task" a la futures/async.)
//!
//! Within a client, we batch writes similar to how a BufWriter would: copy incoming writes to a
//! local vector until we have buffered up to a given threshold. Once that threshold is reached, we
//! then send a reference to buffer on the channel to the receiving thread. While this write is
//! pending, the vector is kept alive in a separate buffer of "guards" that also contain an
//! ErrorCode which the receiver thread can use to signal that the pending write has been issued.
//! Buffers corresponding to writes that have completed are reused for future batches. This
//! protocol (as opposed to one that transfers ownership of the buffer to the thread performing the
//! writes) allows each client thread to avoid allocating new buffers continuously. It also
//! mitigates a "producer-consumer" allocation and freeing pattern, which can put a lot of strain
//! on some allocators.
//!
//! To facilitate easier testing, the functionality of the file system that we use is abstracted in
//! the `FileFactory` trait. The `testing` module contains an implementation of this trait that
//! writes all data in memory.

use std::collections::VecDeque;
use std::io::{self, Write};
use std::process::ChildStdin;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};

#[cfg(not(feature = "unstable"))]
use itertools::Itertools;

#[cfg(not(feature = "unstable"))]
use std::ops::Deref;

// TODO: get_handle() should return an error on failure to parse UTF8

// NB we only require mpsc semantics, but at time of writing there are a few open bugs on
// std::sync::mpsc, while crossbeam_channel is seeing more attention.
use crossbeam_channel::{bounded, Receiver, Sender};
use hashbrown::HashMap;

use crate::common::{CompileError, FileSpec, Notification, Result};
use crate::runtime::{command::command_for_write, Str};

/// The maximum number of pending requests in the per-file channels.
const IO_CHAN_SIZE: usize = 8;

/// The size of client-side batches.
const BUFFER_SIZE: usize = 64 << 10;

/// FileFactory abstracts over the portions of the file system used for the output of a frawk
/// program. It includes "file objects" as well as "stdout", which both implement the io::Write
/// trait.
///
/// The factories themselves must also be Clone and thread-safe, as they are passed to writer
/// threads at construction time.
pub trait FileFactory: Clone + 'static + Send + Sync {
    type Output: io::Write;
    type Stdout: io::Write;
    // TODO: make ChildStdin an associated type, to permit better testing
    fn cmd(&self, cmd: &[u8]) -> io::Result<ChildStdin> {
        command_for_write(cmd)
    }
    fn build(&self, path: &str, spec: FileSpec) -> io::Result<Self::Output>;
    // TODO maybe we shold support this returning an error.
    fn stdout(&self) -> Self::Stdout;
}

impl<W: io::Write, T: Fn(&str, FileSpec) -> io::Result<W> + Clone + 'static + Send + Sync>
    FileFactory for T
{
    type Output = W;
    type Stdout = grep_cli::StandardStream;
    fn build(&self, path: &str, spec: FileSpec) -> io::Result<W> {
        (&self)(path, spec)
    }
    fn stdout(&self) -> Self::Stdout {
        grep_cli::stdout(termcolor::ColorChoice::Auto)
    }
}

type FileWriter = std::fs::File;

fn open_file(path: &str, spec: FileSpec) -> io::Result<FileWriter> {
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .append(matches!(spec, FileSpec::Append))
        .open(path)?;
    Ok(file)
}

pub fn default_factory() -> impl FileFactory {
    open_file
}

pub fn factory_from_file(fname: &str) -> io::Result<impl FileFactory> {
    // Do a test open+truncate of the file.
    let _file = open_file(fname, FileSpec::Trunc)?;

    #[derive(Clone)]
    struct FileStdout(String);
    impl FileFactory for FileStdout {
        type Output = FileWriter;
        type Stdout = FileWriter;
        fn build(&self, path: &str, spec: FileSpec) -> io::Result<Self::Output> {
            open_file(path, spec)
        }
        fn stdout(&self) -> Self::Stdout {
            open_file(self.0.as_str(), FileSpec::Append).expect("failed to open stdout")
        }
    }
    Ok(FileStdout(fname.into()))
}

fn build_handle<W: io::Write, F: Fn(FileSpec) -> io::Result<W> + Send + 'static>(
    f: F,
    is_stdout: bool,
) -> RawHandle {
    let (sender, receiver) = bounded(IO_CHAN_SIZE);
    let error = Arc::new(Mutex::new(None));
    let receiver_error = error.clone();
    std::thread::spawn(move || receive_thread(receiver, receiver_error, f));
    RawHandle {
        error,
        sender,
        line_buffer: is_stdout && grep_cli::is_tty_stdout(),
    }
}

/// Registry is a thread-local handle on all files we have ever interacted with.
///
/// Note that handles are never removed, even after a file is closed. The single thread continues
/// to run and listen for new requests that might trigger a reopen.
pub struct Registry {
    global: Arc<dyn Root>,
    files: HashMap<Str<'static>, FileHandle>,
    cmds: HashMap<Str<'static>, FileHandle>,
    stdout: FileHandle,
}

impl Registry {
    pub fn from_factory(f: impl FileFactory) -> Registry {
        let root_impl = RootImpl::from_factory(f);
        let stdout = root_impl.get_stdout().into_handle();
        Registry {
            global: Arc::new(root_impl),
            files: Default::default(),
            cmds: Default::default(),
            stdout,
        }
    }

    pub fn get_handle<'a>(
        &mut self,
        name: Option<&Str<'a>>,
        fspec: FileSpec,
    ) -> Result<&mut FileHandle> {
        let name = if let Some(s) = name {
            s
        } else {
            return self.get_file(None);
        };
        match fspec {
            FileSpec::Cmd => self.get_cmd(name),
            FileSpec::Trunc | FileSpec::Append => self.get_file(Some(name)),
        }
    }

    pub fn close<'a>(&mut self, path_or_cmd: &Str<'a>) -> Result<()> {
        // TODO: implement a newtype for heterogeneous lookup. We shouldn't have to do the clone or
        // the unmoor here, but we need to because we cannot implement Borrow<Str<'a>> for
        // Borrow<Str<'static>> (conflicts with the blanket impl for Borrow).
        if let Some(fh) = self.files.get_mut(&path_or_cmd.clone().unmoor()) {
            fh.close()?;
            return Ok(());
        }
        if let Some(ch) = self.cmds.get_mut(&path_or_cmd.clone().unmoor()) {
            ch.close()?;
            return Ok(());
        }
        path_or_cmd.with_bytes(|bs| self.global.close(bs))
    }

    pub fn get_cmd<'a>(&mut self, cmd: &Str<'a>) -> Result<&mut FileHandle> {
        use hashbrown::hash_map::Entry;
        // borrowed by with_bytes closure.
        let global = &self.global;
        match self.cmds.entry(cmd.clone().unmoor()) {
            Entry::Occupied(o) => Ok(o.into_mut()),
            Entry::Vacant(v) => {
                Ok(v.insert(cmd.with_bytes(|bs| global.get_command(bs)).into_handle()))
            }
        }
    }

    pub fn get_file<'a>(&mut self, name: Option<&Str<'a>>) -> Result<&mut FileHandle> {
        match name {
            Some(path) => {
                use hashbrown::hash_map::Entry;
                // borrowed by with_bytes closure.
                let global = &self.global;
                match self.files.entry(path.clone().unmoor()) {
                    Entry::Occupied(o) => Ok(o.into_mut()),
                    Entry::Vacant(v) => {
                        let raw = path.with_bytes(|bs| match std::str::from_utf8(bs) {
                            Ok(s) => Ok(global.get_handle(s)),
                            Err(e) => err!("invalid UTF8 in filename: {}", e),
                        })?;
                        Ok(v.insert(raw.into_handle()))
                    }
                }
            }
            None => Ok(&mut self.stdout),
        }
    }

    pub fn destroy_and_flush_all_files(&mut self) -> Result<()> {
        let mut last_error = Ok(());
        for (_, mut fh) in self.files.drain().chain(self.cmds.drain()) {
            let res = fh.flush();
            if res.is_err() {
                last_error = res;
            }
        }
        last_error
    }
}

impl Clone for Registry {
    fn clone(&self) -> Registry {
        Registry {
            global: self.global.clone(),
            files: Default::default(),
            cmds: Default::default(),
            stdout: self.stdout.raw().into_handle(),
        }
    }
}

// We place Root behind a trait so that we can maintain static dispatch at the level of the
// receiver threads, while still avoiding an extra type parameter all the way up the stack.
trait Root: 'static + Send + Sync {
    fn get_command(&self, cmd: &[u8]) -> RawHandle;
    fn get_handle(&self, fname: &str) -> RawHandle;
    fn get_stdout(&self) -> RawHandle;
    // closes a file or command with name `fname`.
    fn close(&self, fname: &[u8]) -> Result<()>;
}

struct RootImpl<F> {
    handles: Mutex<HashMap<String, RawHandle>>,
    commands: Mutex<HashMap<Box<[u8]>, RawHandle>>,
    stdout_raw: RawHandle,
    file_factory: F,
}

impl<F: FileFactory> RootImpl<F> {
    fn from_factory(file_factory: F) -> RootImpl<F> {
        let local_factory = file_factory.clone();
        let stdout_raw = build_handle(
            move |_append| Ok(local_factory.stdout()),
            /*is_stdout*/ true,
        );
        RootImpl {
            handles: Default::default(),
            commands: Default::default(),
            stdout_raw,
            file_factory,
        }
    }
}

impl<F: FileFactory> Root for RootImpl<F> {
    fn close(&self, fname: &[u8]) -> Result<()> {
        let mut handle = None;
        {
            let cmds = self.commands.lock().unwrap();
            if let Some(h) = cmds.get(fname) {
                // We do this extra song and dance to avoid calling close with the lock held.
                handle = Some(h.clone());
            }
        }
        if let Some(h) = handle.take() {
            h.into_handle().close()?;
            return Ok(());
        }
        {
            let fname = if let Ok(s) = std::str::from_utf8(fname) {
                s
            } else {
                // If this file name is invalid UTF8, we haven't opened it; no need to return an
                // error.
                return Ok(());
            };
            let files = self.handles.lock().unwrap();
            if let Some(h) = files.get(fname) {
                handle = Some(h.clone());
            }
        }
        if let Some(h) = handle.take() {
            h.into_handle().close()?;
            return Ok(());
        }
        Ok(())
    }
    fn get_command(&self, cmd: &[u8]) -> RawHandle {
        let mut cmds = self.commands.lock().unwrap();
        if let Some(h) = cmds.get(cmd) {
            return h.clone();
        }
        let local_factory = self.file_factory.clone();
        let local_name = Box::<[u8]>::from(cmd);
        let global_name = local_name.clone();
        let handle = build_handle(
            move |_| local_factory.cmd(&*local_name),
            /*is_stdout=*/ true,
        );
        let _old = cmds.insert(global_name, handle.clone());
        debug_assert!(
            _old.is_none(),
            "duplicate insertion of command: {}",
            String::from_utf8_lossy(cmd)
        );
        handle
    }
    fn get_handle(&self, fname: &str) -> RawHandle {
        let mut handles = self.handles.lock().unwrap();
        if let Some(h) = handles.get(fname) {
            return h.clone();
        }
        let local_factory = self.file_factory.clone();
        let local_name = String::from(fname);
        let global_name = local_name.clone();
        let handle = build_handle(
            move |append| local_factory.build(local_name.as_str(), append),
            /*is_stdout=*/ false,
        );
        handles.insert(global_name, handle.clone());
        handle
    }
    fn get_stdout(&self) -> RawHandle {
        self.stdout_raw.clone()
    }
}

/// FileHandle contains thread-local state around writing to and closing an output file.
pub struct FileHandle {
    raw: RawHandle,
    // Why do we deal with Box<WriteGuard>s and not WriteGuards?
    //
    // We pass a reference to the ErrorCode within a WriteGuard to a write request. That address
    // must remain stable; if we stored a WriteGuard in a VecDeque directly we would not have that
    // guarantee. old_guards caches recent WriteGuards that have been discarded to avoid allocation
    // overheads in cases where we aren't using a fast malloc (in cases where we are, doing this
    // may still be marginally faster).
    old_guards: Vec<Box<WriteGuard>>,
    guards: VecDeque<Box<WriteGuard>>,
    cur_batch: Box<WriteGuard>,
}

impl FileHandle {
    fn raw(&self) -> RawHandle {
        self.raw.clone()
    }

    fn clear_guards(&mut self) -> Result<()> {
        let mut done_count = 0;
        for (i, guard) in self.guards.iter().enumerate() {
            match guard.status() {
                RequestStatus::ONGOING => break,
                RequestStatus::OK => done_count = i,
                RequestStatus::ERROR => return Err(self.read_error()),
            }
        }
        // TODO: off-by-one, should morally be ..=, but that doesn't handle the empty case.
        for _ in 0..done_count {
            let old = self.guards.pop_front().unwrap();
            if self.old_guards.len() < IO_CHAN_SIZE {
                self.old_guards.push(old);
            }
        }
        Ok(())
    }

    fn guard<'a>(&mut self) -> Box<WriteGuard> {
        if let Some(mut g) = self.old_guards.pop() {
            g.activate();
            g
        } else {
            Box::new(WriteGuard::default())
        }
    }

    fn read_error(&self) -> CompileError {
        // The receiver shut down before we did. That means something went wrong: probably an IO
        // error of some kind. In that case, the receiver thread stashed away the error it recieved
        // in raw.error for us to read it out. We don't optimize this path too aggressively because
        // IO errors in frawk scripts are fatal.
        const BAD_SHUTDOWN_MSG: &str = "internal error: (writer?) thread did not shut down cleanly";
        if let Ok(lock) = self.raw.error.lock() {
            match &*lock {
                Some(err) => err.clone(),
                None => CompileError(BAD_SHUTDOWN_MSG.into()),
            }
        } else {
            CompileError(BAD_SHUTDOWN_MSG.into())
        }
    }

    fn clear_batch(&mut self, upto: Option<usize>) -> Result<()> {
        if self.cur_batch.data.len() == 0 {
            return Ok(());
        }
        let (flush, upto) = if let Some(ix) = upto {
            (true, ix)
        } else {
            (false, self.cur_batch.data.len())
        };
        self.clear_guards()?;
        let mut next_batch = self.guard();
        self.cur_batch.peel(upto, &mut *next_batch);
        let req = self.cur_batch.request(flush);
        self.raw.sender.send(req).unwrap();
        std::mem::swap(&mut next_batch, &mut self.cur_batch);
        self.guards.push_back(next_batch);
        Ok(())
    }

    pub fn write_all<'a>(&mut self, ss: &[&Str<'a>], spec: FileSpec) -> Result<()> {
        let cur_len = self.cur_batch.data.len();
        let mut added_bytes = 0;
        let mut last_line = None;
        for s in ss.iter() {
            let bs = unsafe { &*s.get_bytes() };
            self.cur_batch.extend(&*bs, spec);
            if self.raw.line_buffer {
                if let Some(ix) = memchr::memchr(b'\n', bs) {
                    // +1 to include the newline
                    last_line = Some(cur_len + added_bytes + ix + 1);
                }
            }
            added_bytes += bs.len();
        }
        if (self.raw.line_buffer && last_line.is_some()) || (added_bytes + cur_len > BUFFER_SIZE) {
            self.clear_batch(last_line)?;
        }
        Ok(())
    }
    pub fn write<'a>(&mut self, s: &Str<'a>, spec: FileSpec) -> Result<()> {
        self.write_all(&[s], spec)
    }

    pub fn flush(&mut self) -> Result<()> {
        self.clear_batch(None)?;
        let (n, req) = Request::flush();
        self.raw.sender.send(req).unwrap();
        n.1.wait();
        self.guards.clear();
        if let RequestStatus::ERROR = n.0.read() {
            Err(self.read_error())
        } else {
            Ok(())
        }
    }

    pub fn close(&mut self) -> Result<()> {
        self.clear_batch(None)?;
        self.raw.sender.send(Request::Close).unwrap();
        Ok(())
    }
}

impl Drop for FileHandle {
    fn drop(&mut self) {
        let _ = self.flush();
        self.cur_batch.status.set_ok();
    }
}

/// A basic atomic error code type:
///
/// * 0 => "ONGOING"
/// * 1 => "OK"
/// * 2 => "ERROR"
///
/// ErrorCode is used in write and flush requests to signal if a request is still pending, has
/// completed successfully, or ran into an error. The "ERROR" case contains no additional error
/// information. FileHandle sets up a separate channel to transmit the full error; encountering an
/// "ERROR" status is merely a signal to check this location for a more detailed error message.
#[derive(Default)]
struct ErrorCode(AtomicUsize);

#[derive(Debug)]
enum RequestStatus {
    ONGOING = 0,
    OK = 1,
    ERROR = 2,
}

impl ErrorCode {
    fn read(&self) -> RequestStatus {
        match self.0.load(Ordering::Acquire) {
            0 => RequestStatus::ONGOING,
            1 => RequestStatus::OK,
            2 => RequestStatus::ERROR,
            _ => unreachable!(),
        }
    }
    fn set_ok(&self) {
        self.0.store(RequestStatus::OK as usize, Ordering::Release);
    }
    fn set_error(&self) {
        self.0
            .store(RequestStatus::ERROR as usize, Ordering::Release);
    }
}

/// Request is the "wire protocol" sent from client threads to the writing thread for a given file.
enum Request {
    // Because frawk has no separate call for opening an output file, we pass `append` along with
    // write requests. The append value does nothing unless this is the first write received (after
    // a close).
    Write {
        data: *const [u8],
        status: *const ErrorCode,
        spec: FileSpec,
        flush: bool,
    },
    Flush(Arc<(ErrorCode, Notification)>),
    Close,
}

// This isn't implemented automatically because of the raw pointers in Write. Those pointers are
// never mutated or reassigned, and the protocol guarantees that they remain valid for as long as
// the receiver thread has a reference to them.
unsafe impl Send for Request {}

impl Request {
    fn flush() -> (Arc<(ErrorCode, Notification)>, Request) {
        let notify = Arc::new((ErrorCode::default(), Notification::default()));
        let req = Request::Flush(notify.clone());
        (notify, req)
    }
    fn size(&self) -> usize {
        match self {
            // NB, aside from the invariants we maintain about the validity of `data`, grabbing the
            // length here should _always_ be safe. This is tracked by the {const_}slice_ptr_len
            // feature.
            Request::Write { data, .. } => unsafe { &**data }.len(),
            Request::Flush(_) | Request::Close => 0,
        }
    }
    fn set_code(&self, mut f: impl FnMut(&ErrorCode)) {
        match self {
            Request::Write { status, .. } => f(unsafe { &**status }),
            Request::Flush(n) => {
                f(&n.0);
                n.1.notify();
            }
            Request::Close => {}
        }
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        match self {
            Request::Write { status, .. } => {
                // We have to have set this as either ok, or an error.
                let status = unsafe { &**status }.read();
                assert!(!matches!(status, RequestStatus::ONGOING));
            }
            Request::Flush(n) => {
                assert!(n.1.has_been_notified());
            }
            Request::Close => {}
        }
    }
}

/// WriteGuard represents a pending write request.
#[derive(Default)]
struct WriteGuard {
    data: Vec<u8>,
    status: ErrorCode,
    spec: FileSpec,
}

impl WriteGuard {
    fn extend(&mut self, bs: &[u8], spec: FileSpec) {
        self.data.extend(bs);
        self.spec = spec;
    }

    fn peel(&mut self, bytes: usize, next: &mut WriteGuard) {
        if bytes < self.data.len() {
            next.data.extend(self.data[bytes..].iter().cloned());
            self.data.truncate(bytes);
        }
    }

    fn request(&self, flush: bool) -> Request {
        Request::Write {
            data: &self.data[..],
            status: &self.status,
            spec: self.spec,
            flush,
        }
    }

    fn status(&self) -> RequestStatus {
        self.status.read()
    }

    fn activate(&mut self) {
        self.status = ErrorCode::default();
        self.spec = FileSpec::Trunc;
        self.data.clear();
    }
}

impl Drop for WriteGuard {
    fn drop(&mut self) {
        let status = self.status();
        assert!(!matches!(status, RequestStatus::ONGOING));
    }
}

#[derive(Clone)]
struct RawHandle {
    error: Arc<Mutex<Option<CompileError>>>,
    sender: Sender<Request>,
    line_buffer: bool,
}

impl RawHandle {
    fn into_handle(self) -> FileHandle {
        FileHandle {
            cur_batch: Default::default(),
            raw: self,
            guards: Default::default(),
            old_guards: Default::default(),
        }
    }
}

// Implementation of the "server" thread issuing the writes.

#[derive(Default)]
struct WriteBatch {
    io_vec: Vec<io::IoSlice<'static>>,
    requests: Vec<Request>,
    n_writes: usize,
    write_bytes: usize,
    flush: bool,
    close: bool,
}

#[cfg(feature = "unstable")]
fn write_all(batch: &mut WriteBatch, w: &mut impl Write) -> io::Result</*close=*/ bool> {
    w.write_all_vectored(&mut batch.io_vec[..]).map(|_| true)
}

#[cfg(not(feature = "unstable"))]
fn write_all(batch: &mut WriteBatch, w: &mut impl Write) -> io::Result</*close=*/ bool> {
    batch
        .io_vec
        .iter()
        .map(|slice| {
            let buf: &[u8] = slice.deref();
            w.write_all(buf)
        })
        .fold_results(true, |a, _| a)
}

impl WriteBatch {
    fn n_writes(&self) -> usize {
        self.n_writes
    }

    fn issue(&mut self, w: &mut impl Write) -> io::Result</*close=*/ bool> {
        if self.write_bytes > 0 {
            write_all(self, w)?;
        }
        if self.flush || self.close {
            w.flush()?;
        }
        let close = self.close;
        self.clear();
        Ok(close)
    }

    fn get_spec(&self) -> FileSpec {
        for req in self.requests.iter() {
            if let Request::Write { spec, .. } = req {
                return *spec;
            }
        }
        Default::default()
    }
    fn push(&mut self, req: Request) -> bool {
        match &req {
            Request::Write { data, flush, .. } => {
                // TODO: this does not handle payloads larger than 4GB on windows, see
                // documentation for IoSlice. Should be an easy fix if this comes up.
                let io_slice = io::IoSlice::new(unsafe { &**data });
                self.write_bytes += io_slice.len();
                self.io_vec.push(io_slice);
                self.n_writes += 1;
                self.flush |= *flush;
            }
            Request::Flush(_) => self.flush = true,
            Request::Close => self.close = true,
        };
        self.requests.push(req);
        self.flush || self.close
    }
    fn clear_batch(&mut self, mut f: impl FnMut(&ErrorCode)) {
        self.io_vec.clear();
        for req in self.requests.drain(..) {
            req.set_code(&mut f)
        }
        self.close = false;
        self.flush = false;
        self.n_writes = 0;
        self.write_bytes = 0;
    }
    fn clear_error(&mut self) {
        self.clear_batch(ErrorCode::set_error)
    }
    fn clear(&mut self) {
        self.clear_batch(ErrorCode::set_ok)
    }
}

fn receive_thread<W: io::Write>(
    receiver: Receiver<Request>,
    error: Arc<Mutex<Option<CompileError>>>,
    f: impl Fn(FileSpec) -> io::Result<W>,
) {
    let mut batch = WriteBatch::default();
    if let Err(e) = receive_loop(&receiver, &mut batch, f) {
        // We got an error! install it in the `error` mutex.
        {
            let mut err = error.lock().unwrap();
            *err = Some(CompileError(format!("{}", e)));
        }
        // Now signal an error on any pending requests.
        batch.clear_error();
        // And send an error back for any more requests that come in.
        while let Ok(req) = receiver.recv() {
            req.set_code(ErrorCode::set_error)
        }
    }
}

fn receive_loop<W: io::Write>(
    receiver: &Receiver<Request>,
    batch: &mut WriteBatch,
    f: impl Fn(FileSpec) -> io::Result<W>,
) -> io::Result<()> {
    const MAX_BATCH_BYTES: usize = 1 << 20;
    const MAX_BATCH_SIZE: usize = 1 << 10;

    // Writer starts off closed. We use `f` to open it if a write appears.
    let mut writer = None;

    while let Ok(req) = receiver.recv() {
        // We build up a reasonably-sized batch of writes in the channel if it contains pending
        // operations in the channel.
        //
        // To simplify matters, we cut a batch short if we receive a "flush" or "close" request
        // (signaled by batch.push returning true).
        //
        // NB: this batching is redundant when writing to stdout (in which case grep_cli gives us a
        // buffered version of the writer), but we want it for the file IO case, where this
        // approach permits us fewer copies than the BufWriter approach.
        let mut batch_bytes = req.size();
        if !batch.push(req) {
            while let Ok(req) = receiver.try_recv() {
                batch_bytes += req.size();
                if batch.push(req)
                    || batch.n_writes() >= MAX_BATCH_SIZE
                    || batch_bytes >= MAX_BATCH_BYTES
                {
                    break;
                }
            }
        }
        if writer.is_none() {
            if batch.n_writes() == 0 {
                // check for a "flush/close-only batch", which we treat as a noop if the file is
                // closed.
                batch.clear();
                continue;
            }
            // We need to (re)open the file, the first write request will tell us whether or not
            // this is an append request.
            writer = Some(f(batch.get_spec())?);
        }
        if batch.issue(writer.as_mut().unwrap())? {
            writer = None;
        }
    }
    Ok(())
}

pub mod testing {
    use super::*;

    /// A file factory that writes all data in memory; used for unit testing.
    #[derive(Clone, Default)]
    pub struct FakeFs {
        pub stdout: FakeFile,
        named: Arc<Mutex<HashMap<String, FakeFile>>>,
    }

    impl FakeFs {
        pub fn get_handle(&self, path: &str) -> Option<FakeFile> {
            self.named.lock().unwrap().get(path).cloned()
        }
    }

    impl FileFactory for FakeFs {
        type Output = FakeFile;
        type Stdout = FakeFile;
        fn build(&self, path: &str, spec: FileSpec) -> io::Result<Self::Output> {
            let mut named = self.named.lock().unwrap();
            if let Some(file) = named.get(path) {
                file.reopen(spec);
                return Ok(file.clone());
            }
            let new_file = FakeFile::default();
            named.insert(path.into(), new_file.clone());
            Ok(new_file)
        }
        fn stdout(&self) -> Self::Stdout {
            self.stdout.clone()
        }
    }

    #[derive(Default)]
    struct FakeFileInner {
        data: Mutex<Vec<u8>>,
        poison: AtomicBool,
    }

    impl FakeFileInner {
        fn result(&self) -> io::Result<()> {
            if self.poison.load(Ordering::Acquire) {
                Err(io::Error::new(io::ErrorKind::Other, "poisoned fake file!"))
            } else {
                Ok(())
            }
        }
    }

    /// The files stored in a FakeFs.
    ///
    /// These are primarily vectors of bytes, but they can also be configured to return errors or
    /// block write requests.
    #[derive(Clone, Default)]
    pub struct FakeFile(Arc<FakeFileInner>);

    impl FakeFile {
        pub fn set_poison(&self, p: bool) {
            self.0.poison.store(p, Ordering::Release);
        }
        pub fn read_data(&self) -> Vec<u8> {
            (*self.0.data.lock().unwrap()).clone()
        }
        pub fn reopen(&self, spec: FileSpec) {
            if !matches!(spec, FileSpec::Append) {
                self.clear();
            }
        }
        pub fn clear(&self) {
            self.0.data.lock().unwrap().clear();
        }
    }

    impl Write for FakeFile {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.result()?;
            self.0.data.lock().unwrap().extend(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            self.0.result()?;
            Ok(())
        }
        fn write_vectored(&mut self, bufs: &[io::IoSlice]) -> io::Result<usize> {
            self.0.result()?;
            let mut written = 0;
            let mut data = self.0.data.lock().unwrap();
            for b in bufs {
                let bytes: &[u8] = &*b;
                data.extend(bytes);
                written += bytes.len();
            }
            Ok(written)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    #[test]
    fn basic_writing() {
        let s1 = Str::from("hello");
        let s2 = Str::from(" there");
        let fs = FakeFs::default();
        let mut reg = Registry::from_factory(fs.clone());
        {
            let handle = reg
                .get_handle(/*stdout*/ None, FileSpec::default())
                .unwrap();
            handle.write(&s1, FileSpec::Append).unwrap();
            handle.write(&s2, FileSpec::Append).unwrap();
            handle.flush().unwrap();
            handle.write(&s1, FileSpec::Append).unwrap();
            handle.write(&s2, FileSpec::Append).unwrap();
            handle.flush().unwrap();
        }
        let data = fs.stdout.read_data();
        assert_eq!(&data[..], "hello therehello there".as_bytes());
    }

    #[test]
    fn reopen_named_file() {
        let fname_str = "/fake";
        let fname = Str::from(fname_str);
        let s1 = Str::from("hello");
        let s2 = Str::from(" there");
        let fs = FakeFs::default();
        let mut reg = Registry::from_factory(fs.clone());
        {
            let handle = reg.get_handle(Some(&fname), FileSpec::default()).unwrap();
            handle.write(&s1, FileSpec::Append).unwrap();
            handle.write(&s2, FileSpec::Append).unwrap();
            handle.flush().unwrap();
            handle.write(&s1, FileSpec::Append).unwrap();
            handle.write(&s2, FileSpec::Append).unwrap();
        }
        {
            let handle = reg.get_handle(Some(&fname), FileSpec::default()).unwrap();
            handle.close().unwrap();
            handle.write(&s1, FileSpec::Trunc).unwrap();
            handle.write(&s2, FileSpec::Trunc).unwrap();
            handle.flush().unwrap();
        }
        let data = fs.get_handle(fname_str).unwrap().read_data();
        assert_eq!(&data[..], "hello there".as_bytes());
    }

    #[test]
    fn multithreaded_write() {
        const N_THREADS: usize = 100;
        const WRITES_PER_THREAD: usize = 1000;
        let fs = FakeFs::default();
        fs.build("/fake/BAD", FileSpec::Trunc)
            .unwrap()
            .set_poison(true);
        let mut threads = Vec::with_capacity(N_THREADS);
        {
            let reg = Registry::from_factory(fs.clone());
            for t in 0..N_THREADS {
                let mut treg = reg.clone();
                threads.push(std::thread::spawn(move || {
                    let a = Str::from("A");
                    let b = Str::from("B");
                    let fa = Str::from("/fake/A");
                    let fb = Str::from("/fake/B");
                    let fbad = Str::from("/fake/BAD");
                    for i in 0..WRITES_PER_THREAD {
                        {
                            let h1 = treg.get_handle(Some(&fa), FileSpec::default()).unwrap();
                            h1.write(&a, FileSpec::Append).unwrap();
                            if (t + i) % 100 == 0 {
                                h1.close().unwrap();
                            }
                        }
                        {
                            // We do not close file b, so append=false should not matter.
                            let h2 = treg.get_handle(Some(&fb), FileSpec::default()).unwrap();
                            h2.write(&b, FileSpec::Trunc).unwrap();
                            if (t + i) % 105 == 0 {
                                h2.flush().unwrap();
                            }
                        }

                        {
                            let h3 = treg.get_handle(Some(&fbad), FileSpec::default()).unwrap();
                            // These won't all be errors.
                            let _ = h3.write(&a, FileSpec::Append);
                            if (t + i) % 103 == 0 {
                                // But all of these will be
                                assert!(h3.flush().is_err())
                            }
                        }
                    }
                }));
            }
        }
        for t in threads.into_iter() {
            t.join().unwrap();
        }
        let expected_a = vec![b'A'; N_THREADS * WRITES_PER_THREAD];
        let expected_b = vec![b'B'; N_THREADS * WRITES_PER_THREAD];
        assert_eq!(fs.get_handle("/fake/A").unwrap().read_data(), expected_a);
        assert_eq!(fs.get_handle("/fake/B").unwrap().read_data(), expected_b);
    }
}
