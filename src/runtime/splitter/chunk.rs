use crate::common::Result;
use crate::runtime::{
    splitter::{
        batch::{get_find_indexes, get_find_indexes_bytes, InputFormat, Offsets},
        Reader,
    },
    str_impl::UniqueBuf,
};

use std::io::Read;
use std::sync::Arc;

// TODO: rephrase CSVReader + BytesReader + DefaultReader in terms of a ChnunkProducer
//      (in that order: DefaultReader will need its own ChunkProducer, I think?)
// TODO: write a ParallelChunkProducer that wraps an arbitrary ChunkProducer

#[derive(Copy, Clone)]
enum ChunkState {
    Init,
    Main,
    Done,
}

pub struct OffsetChunkProducer<R, F> {
    inner: Reader<R>,
    name: Arc<str>,
    find_indexes: F,
    record_sep: u8,
    state: ChunkState,
}

pub fn new_offset_chunk_producer_csv<R: Read>(
    r: R,
    chunk_size: usize,
    name: &str,
    ifmt: InputFormat,
) -> OffsetChunkProducer<R, impl FnMut(&[u8], &mut Offsets)> {
    let find_indexes = get_find_indexes(ifmt);
    OffsetChunkProducer {
        name: name.into(),
        inner: Reader::new(r, chunk_size, /*padding=*/ 128),
        find_indexes: move |bs: &[u8], offs: &mut Offsets| {
            unsafe { find_indexes(bs, offs, 0, 0) };
        },
        record_sep: b'\n',
        state: ChunkState::Init,
    }
}

pub fn new_offset_chunk_producer_bytes<R: Read>(
    r: R,
    chunk_size: usize,
    name: &str,
    field_sep: u8,
    record_sep: u8,
) -> OffsetChunkProducer<R, impl FnMut(&[u8], &mut Offsets)> {
    let find_indexes = get_find_indexes_bytes().expect("byte splitter not available");
    OffsetChunkProducer {
        name: name.into(),
        inner: Reader::new(r, chunk_size, /*padding=*/ 128),
        find_indexes: move |bs: &[u8], offs: &mut Offsets| unsafe {
            find_indexes(bs, offs, field_sep, record_sep)
        },
        record_sep,
        state: ChunkState::Init,
    }
}

// TODO: strategy-wise, build a new LineReader which is just bytereader on top of the new stack.
// get it passing the bytereader tests.
// TODO: need to decide on semantics of "nextfile":
//   * hard error on "per-record" works as expected on "per-file" ?
//   * should converge on how to construct LineReaders, probably want to pass in the chunk-reader?
//   * add a "try_clone" method that attempts to make another parallel reader?
//   * figure out read_state (we should be able to "never return an error" and keep an eof signal).
//   * We'll eventually need to be able to send a "shutdown" signal.

pub struct OffsetChunk {
    pub buf: Option<UniqueBuf>,
    pub len: usize,
    pub off: Offsets,
}

pub trait ChunkProducer {
    type Chunk: Send;
    fn get_chunk(&mut self, chunk: &mut Self::Chunk) -> Result<bool /*done*/>;
    fn get_name(&self) -> &str;
    fn next_file(&mut self) -> Result<bool>;
}

impl<C: Send> ChunkProducer for Box<dyn ChunkProducer<Chunk = C>> {
    type Chunk = C;
    fn next_file(&mut self) -> Result<bool> {
        (&mut **self).next_file()
    }
    fn get_chunk(&mut self, chunk: &mut C) -> Result<bool> {
        (&mut **self).get_chunk(chunk)
    }
    fn get_name(&self) -> &str {
        (&**self).get_name()
    }
}

impl<R: Read, F: FnMut(&[u8], &mut Offsets)> ChunkProducer for OffsetChunkProducer<R, F> {
    type Chunk = OffsetChunk;
    fn next_file(&mut self) -> Result<bool> {
        self.state = ChunkState::Done;
        self.inner.force_eof();
        Ok(false)
    }
    fn get_chunk(&mut self, chunk: &mut OffsetChunk) -> Result<bool> {
        loop {
            match self.state {
                ChunkState::Init => {
                    self.state = if self.inner.reset()? {
                        ChunkState::Done
                    } else {
                        ChunkState::Main
                    };
                }
                ChunkState::Main => {
                    let buf = self.inner.buf.clone();
                    let bs = buf.as_bytes();
                    (self.find_indexes)(bs, &mut chunk.off);
                    let mut target = None;
                    let mut new_len = chunk.off.fields.len();
                    let mut always_truncate = new_len;
                    for offset in chunk.off.fields.iter().rev() {
                        let offset = *offset as usize;
                        if offset >= self.inner.end {
                            always_truncate -= 1;
                            new_len -= 1;
                            continue;
                        }
                        if bs[offset] == self.record_sep {
                            target = Some(offset + 1);
                            break;
                        }
                        new_len -= 1;
                    }
                    debug_assert!(new_len <= always_truncate);
                    let is_partial = if let Some(chunk_end) = target {
                        self.inner.start = chunk_end;
                        false
                    } else {
                        debug_assert_eq!(new_len, 0);
                        true
                    };
                    chunk.len = self.inner.end;
                    let is_eof = self.inner.reset()?;
                    return match (is_partial, is_eof) {
                        (false, false) => {
                            // Yield buffer, stay in main.
                            chunk.buf = Some(buf.try_unique().unwrap());
                            chunk.off.fields.truncate(new_len);
                            Ok(false)
                        }
                        (false, true) | (true, true) => {
                            // Yield the entire buffer, this was the last piece of data.
                            self.inner.clear_buf();
                            chunk.buf = Some(buf.try_unique().unwrap());
                            chunk.off.fields.truncate(always_truncate);
                            self.state = ChunkState::Done;
                            Ok(false)
                        }
                        // We read an entire chunk, but we didn't find a full record. Try again
                        // (note that the call to reset read in a larger chunk and would have kept
                        // a prefix)
                        (true, false) => continue,
                    };
                }
                ChunkState::Done => return Ok(true),
            }
        }
    }
    fn get_name(&self) -> &str {
        &*self.name
    }
}
