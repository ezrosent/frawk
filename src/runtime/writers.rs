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
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Condvar, Mutex,
};

use hashbrown::HashMap;

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

#[derive(Default)]
struct Root {
    shutdown_start: Notification,
    shutdown_done: Notification,
    registry: Mutex<HashMap<String, RawHandle>>,
}

struct Registry {
    root: Arc<Root>,
    local: HashMap<Str<'static>, FileHandle>,
}

enum Request {
    Write {
        data: *const [u8],
        done: *const AtomicBool,
    },
    Flush(Notification),
}

struct WriteRequest {
    data: *const [u8],
    done: *const AtomicBool,
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

    fn request(&self) -> WriteRequest {
        WriteRequest {
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
struct RawHandle(mpsc::SyncSender<WriteRequest>);

struct FileHandle {
    sender: mpsc::SyncSender<WriteRequest>,
    guards: VecDeque<WriteGuard>,
}

impl RawHandle {
    fn into_handle(self) -> FileHandle {
        FileHandle {
            sender: self.0,
            guards: VecDeque::new(),
        }
    }
}
