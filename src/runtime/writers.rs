//! Support for writing to files from multiple threads.
//!
//! The basic idea is to launch a thread per file and to send write requests down a bounded
//! channel.
//!
//! This is tricky because frawk strings are not reference-counted in a thread-safe manner. We
//! solve this by sending the raw bytes along the channel and keeping an instance of the string
//! around in the sending thread to ensure the bytes are not garbage collected. The receiving
//! thread then flips a per-request boolean to signal that a string is no longer needed.

use std::collections::VecDeque;
use std::io::{self, Write};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Condvar, Mutex,
};

// TODO:
// * document and rearrange the declarations in this file to make it easier to read.
// * replace FileWrite in the runtime module, migrating harness to new fake stdout
// * fix lint errors around u128/extern

use crossbeam_channel::{bounded, Receiver, Sender};
use hashbrown::HashMap;

use crate::common::{CompileError, Result};
use crate::runtime::Str;

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
        fn build(&self, path: &str, append: bool) -> io::Result<Self::Output> {
            let mut named = self.named.lock().unwrap();
            if let Some(file) = named.get(path) {
                file.reopen(append);
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
        pub fn reopen(&self, append: bool) {
            if !append {
                self.0.data.lock().unwrap().clear();
            }
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
            let handle = reg.get_handle(/*stdout*/ None);
            handle.write(&s1, /*append=*/ true).unwrap();
            handle.write(&s2, /*append=*/ true).unwrap();
            handle.flush().unwrap();
            handle.write(&s1, /*append=*/ true).unwrap();
            handle.write(&s2, /*append=*/ true).unwrap();
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
            let handle = reg.get_handle(Some(&fname));
            handle.write(&s1, /*append=*/ true).unwrap();
            handle.write(&s2, /*append=*/ true).unwrap();
            handle.flush().unwrap();
            handle.write(&s1, /*append=*/ true).unwrap();
            handle.write(&s2, /*append=*/ true).unwrap();
        }
        {
            let handle = reg.get_handle(Some(&fname));
            handle.close();
            handle.write(&s1, /*append=*/ false).unwrap();
            handle.write(&s2, /*append=*/ true).unwrap();
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
        fs.build("/fake/BAD", false).unwrap().set_poison(true);
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
                            let h1 = treg.get_handle(Some(&fa));
                            h1.write(&a, /*append=*/ true).unwrap();
                            if (t + i) % 100 == 0 {
                                h1.close();
                            }
                        }
                        {
                            // We do not close file b, so append=false should not matter.
                            let h2 = treg.get_handle(Some(&fb));
                            h2.write(&b, /*append=*/ false).unwrap();
                            if (t + i) % 105 == 0 {
                                h2.flush().unwrap();
                            }
                        }

                        {
                            let h3 = treg.get_handle(Some(&fbad));
                            // These won't all be errors.
                            let _ = h3.write(&a, /*append=*/ true);
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

/// Notification is a simple object used to synchronize multiple threads around a single event
/// occuring. Based on the absl object of the same name.
struct Notification {
    notified: AtomicBool,
    mu: Mutex<()>,
    cv: Condvar,
}

impl Default for Notification {
    fn default() -> Notification {
        Notification {
            notified: AtomicBool::new(false),
            mu: Mutex::new(()),
            cv: Condvar::new(),
        }
    }
}

impl Notification {
    fn has_been_notified(&self) -> bool {
        self.notified.load(Ordering::Acquire)
    }
    fn notify(&self) {
        if self.has_been_notified() {
            return;
        }
        let _guard = self.mu.lock().unwrap();
        self.notified.store(true, Ordering::Release);
        self.cv.notify_all();
    }
    fn wait(&self) {
        loop {
            // Fast path: check if the notification has already happened.
            if self.has_been_notified() {
                return;
            }
            // Slow path: grab the lock, and check notification state again before waiting on the
            // condition variable.
            let mut _guard = self.mu.lock().unwrap();
            if self.has_been_notified() {
                return;
            }
            _guard = self.cv.wait(_guard).unwrap();
        }
    }
}

/// A basic atomic error code type:
///
/// * 0 => "ONGOING"
/// * 1 => "OK"
/// * 2 => "ERROR"
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

pub trait FileFactory: Clone + 'static + Send + Sync {
    type Output: io::Write;
    type Stdout: io::Write;
    fn build(&self, path: &str, append: bool) -> io::Result<Self::Output>;
    fn stdout(&self) -> Self::Stdout;
}

impl<W: io::Write, T: Fn(&str, bool) -> io::Result<W> + Clone + 'static + Send + Sync> FileFactory
    for T
{
    type Output = W;
    type Stdout = std::io::Stdout;
    fn build(&self, path: &str, append: bool) -> io::Result<W> {
        (&self)(path, append)
    }
    fn stdout(&self) -> Self::Stdout {
        std::io::stdout()
    }
}

// We place Root behind a trait so that we can maintain static dispatch at the level of the
// receiver threads, while still avoiding an extra type parameter all the way up the stack.
trait Root: 'static + Sync + Send {
    fn get_handle(&self, fname: &str) -> RawHandle;
    fn get_stdout(&self) -> RawHandle;
}

struct RootImpl<F> {
    handles: Mutex<HashMap<String, RawHandle>>,
    stdout_raw: RawHandle,
    file_factory: F,
}

pub fn default_factory() -> impl FileFactory {
    |path: &str, append| {
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .append(append)
            .open(path)
    }
}

const IO_CHAN_SIZE: usize = 128;

fn build_handle<W: io::Write, F: Fn(bool) -> io::Result<W> + Send + 'static>(f: F) -> RawHandle {
    let (sender, receiver) = bounded(IO_CHAN_SIZE);
    let error = Arc::new(Mutex::new(None));
    let receiver_error = error.clone();
    std::thread::spawn(move || receive_thread(receiver, receiver_error, f));
    RawHandle { error, sender }
}

impl<F: FileFactory> RootImpl<F> {
    fn from_factory(file_factory: F) -> RootImpl<F> {
        let local_factory = file_factory.clone();
        let stdout_raw = build_handle(move |_append| Ok(local_factory.stdout()));
        RootImpl {
            handles: Default::default(),
            stdout_raw,
            file_factory,
        }
    }
}

impl<F: FileFactory> Root for RootImpl<F> {
    fn get_handle(&self, fname: &str) -> RawHandle {
        let mut handles = self.handles.lock().unwrap();
        if let Some(h) = handles.get(fname) {
            return h.clone();
        }
        let local_factory = self.file_factory.clone();
        let local_name = String::from(fname);
        let global_name = local_name.clone();
        let handle = build_handle(move |append| local_factory.build(local_name.as_str(), append));
        handles.insert(global_name, handle.clone());
        handle
    }
    fn get_stdout(&self) -> RawHandle {
        self.stdout_raw.clone()
    }
}

pub struct Registry {
    global: Arc<dyn Root>,
    local: HashMap<Str<'static>, FileHandle>,
    stdout: FileHandle,
}

impl Registry {
    pub fn from_factory(f: impl FileFactory) -> Registry {
        let root_impl = RootImpl::from_factory(f);
        let stdout = root_impl.get_stdout().into_handle();
        Registry {
            global: Arc::new(root_impl),
            local: Default::default(),
            stdout,
        }
    }

    pub fn get_handle(&mut self, name: Option<&Str<'static>>) -> &mut FileHandle {
        match name {
            Some(path) => {
                use hashbrown::hash_map::Entry;
                // borrowed by with_str closure.
                let global = &self.global;
                match self.local.entry(path.clone()) {
                    Entry::Occupied(o) => o.into_mut(),
                    Entry::Vacant(v) => {
                        let raw = path.with_str(|s| global.get_handle(s));
                        v.insert(raw.into_handle())
                    }
                }
            }
            None => &mut self.stdout,
        }
    }
}

impl Clone for Registry {
    fn clone(&self) -> Registry {
        Registry {
            global: self.global.clone(),
            local: HashMap::new(),
            stdout: self.stdout.raw().into_handle(),
        }
    }
}

enum Request {
    Write {
        data: *const [u8],
        status: *const ErrorCode,
        append: bool,
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

struct WriteGuard {
    s: Str<'static>,
    status: ErrorCode,
}

impl WriteGuard {
    fn new<'a>(s: &Str<'a>) -> WriteGuard {
        WriteGuard {
            s: s.clone().unmoor(),
            status: ErrorCode::default(),
        }
    }

    fn request(&self, append: bool) -> Request {
        Request::Write {
            data: self.s.get_bytes(),
            status: &self.status,
            append,
        }
    }

    fn status(&self) -> RequestStatus {
        self.status.read()
    }

    fn reset(&mut self) {
        self.s = Str::default();
    }

    fn set_payload<'a>(&mut self, s: &Str<'a>) {
        self.status = ErrorCode::default();
        self.s = s.clone().unmoor();
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
}

impl RawHandle {
    fn into_handle(self) -> FileHandle {
        FileHandle {
            raw: self,
            guards: Default::default(),
            old_guards: Default::default(),
        }
    }
}

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
        for _ in 0..done_count {
            let mut old = self.guards.pop_front().unwrap();
            if self.old_guards.len() < IO_CHAN_SIZE {
                // Reset drops the string, but keeps the errorcode in a "done" state, making it
                // safe to drop.
                old.reset();
                self.old_guards.push(old);
            }
        }
        Ok(())
    }

    fn guard<'a>(&mut self, s: &Str<'a>) -> Box<WriteGuard> {
        if let Some(mut g) = self.old_guards.pop() {
            g.set_payload(s);
            g
        } else {
            Box::new(WriteGuard::new(s))
        }
    }

    fn read_error(&self) -> CompileError {
        // The receiver shut down before we did. That means something went wrong: probably an IO
        // error of some kind. In that case, the receiver thread stashed away the error it recieved
        // in raw.error for us to read it out. We don't optimize this path too aggressively because
        // IO errors in frawk scripts are fatal.
        const BAD_SHUTDOWN_MSG: &'static str =
            "internal error: (writer?) thread did not shut down cleanly";
        if let Ok(lock) = self.raw.error.lock() {
            match &*lock {
                Some(err) => err.clone(),
                None => CompileError(BAD_SHUTDOWN_MSG.into()),
            }
        } else {
            CompileError(BAD_SHUTDOWN_MSG.into())
        }
    }

    pub fn write<'a>(&mut self, s: &Str<'a>, append: bool) -> Result<()> {
        self.clear_guards()?;
        let guard = self.guard(s);
        let req = guard.request(append);
        self.raw.sender.send(req).unwrap();
        self.guards.push_back(guard);
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
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

    pub fn close(&self) {
        self.raw.sender.send(Request::Close).unwrap();
    }
}

impl Drop for FileHandle {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[derive(Default)]
struct WriteBatch {
    io_vec: Vec<io::IoSlice<'static>>,
    requests: Vec<Request>,
    n_writes: usize,
    flush: bool,
    close: bool,
}

impl WriteBatch {
    fn n_writes(&self) -> usize {
        self.n_writes
    }
    fn issue(&mut self, w: &mut impl Write) -> io::Result</*close=*/ bool> {
        w.write_all_vectored(&mut self.io_vec[..])?;
        if self.flush || self.close {
            w.flush()?;
        }
        let close = self.close;
        self.clear();
        Ok(close)
    }
    fn is_append(&self) -> bool {
        for req in self.requests.iter() {
            if let Request::Write { append, .. } = req {
                return *append;
            }
        }
        false
    }
    fn push(&mut self, req: Request) -> bool {
        match &req {
            Request::Write { data, .. } => {
                // TODO: this does not handle payloads larger than 4GB on windows, see
                // documentation for IoSlice. Should be an easy fix if this comes up.
                self.io_vec.push(io::IoSlice::new(unsafe { &**data }));
                self.n_writes += 1;
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
    f: impl Fn(bool) -> io::Result<W>,
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
    f: impl Fn(bool) -> io::Result<W>,
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
            writer = Some(f(batch.is_append())?);
        }
        if batch.issue(writer.as_mut().unwrap())? {
            writer = None;
        }
    }
    Ok(())
}
