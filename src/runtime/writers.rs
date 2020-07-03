//! Support for writing to files from multiple threads.
//!
//! The basic idea is to launch a thread per file and to send write requests down a bounded
//! channel.
//!
//! This is tricky because frawk strings are not reference-counted in a thread-safe manner. We
//! solve this by sending the raw bytes along the channel and keeping an instance of the string
//! around in the sending thread to ensure the bytes are not garbage collected. The receiving
//! thread then flips a per-request boolean to signal that a string is no longer needed.
//!
//! TODO: use crossbeam channel

use std::collections::VecDeque;
use std::io::{self, Write};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Condvar, Mutex,
};

use crossbeam_channel::{Receiver, Sender};
use hashbrown::HashMap;

use crate::common::{CompileError, Result};
use crate::runtime::Str;

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
        while !self.has_been_notified() {
            let _guard = self.cv.wait(self.mu.lock().unwrap()).unwrap();
        }
    }
}

// TODO: what about shutdown?
//   * Destructor for a FileHandle either does a "flush" operation (which has a notification) or it
//     just does a "shutdown".
// TODO: what about errors?
//   * Trigger a shutdown, but on the part of the pool.
//   * Writers will notice their writes fail if the thread has stopped.
//   * Once that happens they can safely return an error and clear the queue.
// TODO: parameterize Root by a "factory" of some kind?

struct Registry {
    global: Arc<Mutex<HashMap<String, RawHandle>>>,
    local: HashMap<Str<'static>, FileHandle>,
    stdout: FileHandle,
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
        done: *const AtomicBool,
    },
    Flush(Arc<Notification>),
}

impl Request {
    fn flush() -> (Arc<Notification>, Request) {
        let notify = Arc::new(Notification::default());
        let req = Request::Flush(notify.clone());
        (notify, req)
    }
    fn size(&self) -> usize {
        match self {
            // NB, aside from the invariants we maintain about the validity of `data`, grabbing the
            // length here should _always_ be safe. This is tracked by the {const_}slice_ptr_len
            // feature.
            Request::Write { data, .. } => unsafe { &**data }.len(),
            Request::Flush(_) => 0,
        }
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        match self {
            Request::Write { done, .. } => unsafe { (&**done).store(true, Ordering::Release) },
            Request::Flush(n) => n.notify(),
        }
    }
}

struct WriteGuard {
    s: Str<'static>,
    done: AtomicBool,
}

impl WriteGuard {
    fn new<'a>(s: &Str<'a>) -> WriteGuard {
        WriteGuard {
            s: s.clone().unmoor(),
            done: AtomicBool::new(false),
        }
    }

    fn request(&self) -> Request {
        Request::Write {
            data: self.s.get_bytes(),
            done: &self.done,
        }
    }

    fn done(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }
}

impl Drop for WriteGuard {
    fn drop(&mut self) {
        assert!(self.done());
    }
}

#[derive(Clone)]
struct RawHandle {
    error: Arc<Mutex<Option<CompileError>>>,
    sender: Sender<Request>,
}

struct FileHandle {
    raw: RawHandle,
    guards: VecDeque<WriteGuard>,
}

impl RawHandle {
    fn into_handle(self) -> FileHandle {
        FileHandle {
            raw: self,
            guards: VecDeque::new(),
        }
    }
}

impl FileHandle {
    fn raw(&self) -> RawHandle {
        self.raw.clone()
    }

    fn clear_guards(&mut self) {
        let done_count = self
            .guards
            .iter()
            .enumerate()
            .skip_while(|(_, guard)| guard.done())
            .map(|(i, _)| i)
            .next()
            .unwrap_or(self.guards.len());
        self.guards.rotate_left(done_count);
        self.guards.truncate(self.guards.len() - done_count);
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

    fn write<'a>(&mut self, s: &Str<'a>) -> Result<()> {
        self.clear_guards();
        let guard = WriteGuard::new(s);
        let req = guard.request();
        if let Ok(()) = self.raw.sender.send(req) {
            self.guards.push_back(guard);
            return Ok(());
        }
        Err(self.read_error())
    }
    fn flush(&mut self) -> Result<()> {
        let (n, req) = Request::flush();
        if let Ok(()) = self.raw.sender.send(req) {
            n.wait();
            self.guards.clear();
            return Ok(());
        }
        Err(self.read_error())
    }
}

fn receive_thread(
    receiver: Receiver<Request>,
    writer: impl Write,
    error: Arc<Mutex<Option<CompileError>>>,
) {
    if let Err(e) = receive_loop(receiver, writer) {
        // We got an error! install it in the `error` mutex if it is the first one.
        let mut err = error.lock().unwrap();
        if err.is_none() {
            *err = Some(CompileError(format!("{}", e)));
        }
    }
}

fn receive_loop(receiver: Receiver<Request>, mut writer: impl Write) -> io::Result<()> {
    // Just write out data as it comes in.
    const MAX_BATCH_BYTES: usize = 1 << 20;
    const MAX_BATCH_SIZE: usize = 1 << 10;

    // We separate out data writes (batch) from flush requests (flushes) because the flush
    // requests in a batch must be dropped after the write requests are. Why? Vec does not
    // guarantee a drop order, and clients can use Flush to guarantee that all of the pending write
    // requests have gone out of scope.
    //
    // We could reason that any potential reordering of these drops would result in a "benign"
    // dangling pointer that is never actually read. But then we would have to relax the "assert"
    // in the WriteGuard destructor, which is a useful safeguard.
    let mut batch = Vec::new();
    let mut flushes = Vec::new();
    let mut io_vec = Vec::new();

    fn push(
        req: Request,
        batch: &mut Vec<Request>,
        flushes: &mut Vec<Request>,
        io_vec: &mut Vec<io::IoSlice>,
    ) {
        let is_write = match &req {
            Request::Write { data, .. } => {
                // TODO: this does not handle payloads larger than 4GB on windows, see
                // documentation for IoSlice. Should be an easy fix if this comes up.
                io_vec.push(io::IoSlice::new(unsafe { &**data }));
                true
            }
            Request::Flush(_) => false,
        };
        if is_write {
            batch.push(req);
        } else {
            flushes.push(req);
        }
    }

    while let Ok(req) = receiver.recv() {
        // We have at least one request. See if we can build up a batch before issuing a write.
        let mut batch_bytes = req.size();
        push(req, &mut batch, &mut flushes, &mut io_vec);
        while let Ok(req) = receiver.try_recv() {
            batch_bytes += req.size();
            push(req, &mut batch, &mut flushes, &mut io_vec);
            if batch.len() >= MAX_BATCH_SIZE || batch_bytes >= MAX_BATCH_BYTES {
                break;
            }
        }

        writer.write_all_vectored(&mut io_vec[..])?;
        if flushes.len() > 0 {
            writer.flush()?;
        }
        io_vec.clear();
        batch.clear();
        flushes.clear();
    }
    Ok(())
}
