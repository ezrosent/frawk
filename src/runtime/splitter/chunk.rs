use std::borrow::Borrow;
use std::io::Read;
use std::mem;
use std::sync::Arc;

use crossbeam_channel::{bounded, Receiver, Sender};

use crate::common::Result;
use crate::runtime::{
    splitter::{
        batch::{
            get_find_indexes, get_find_indexes_ascii_whitespace, get_find_indexes_bytes,
            InputFormat, Offsets, WhitespaceOffsets,
        },
        Reader,
    },
    str_impl::UniqueBuf,
};

// TODO: We probably want a better story here about ChunkProducers propagating error values.

pub trait ChunkProducer {
    type Chunk: Chunk;
    // Create up to _requested_size additional handles to the ChunkProducer, if possible. Chunk is
    // Send, so these new producers can be used to read from the same source in parallel.
    //
    // NB: what's going on with this gnarly return type? All implementations either return an empty
    // vector or a vector containing functions returning Box<Self>, but we need this trait to be
    // object-safe, so it returns a trait object instead.
    //
    // Why return a FnOnce rather than  a trait object directly? Some ChunkProducer
    // implementations are not Send, even though the data to initialize a new ChunkProducer reading
    // from the same source is Send. Passing a FnOnce allows us to handle this, which is the case
    // for (e.g.) ShardedChunkProducer.
    fn try_dyn_resize(
        &self,
        _requested_size: usize,
    ) -> Vec<Box<dyn FnOnce() -> Box<dyn ChunkProducer<Chunk = Self::Chunk>> + Send>> {
        vec![]
    }
    fn get_chunk(&mut self, chunk: &mut Self::Chunk) -> Result<bool /*done*/>;
    fn next_file(&mut self) -> Result<bool /*new file available*/>;
}

pub trait Chunk: Send + Default {
    fn get_name(&self) -> &str;
}

// TODO: rephrase CSVReader + BytesReader + DefaultReader in terms of a ChnunkProducer
//      (in that order: DefaultReader will need its own ChunkProducer, I think?)

#[derive(Copy, Clone)]
enum ChunkState {
    Init,
    Main,
    Done,
}

pub struct OffsetChunkProducer<R, F> {
    inner: Reader<R>,
    cur_file_version: u32,
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
    start_version: u32,
) -> OffsetChunkProducer<R, impl FnMut(&[u8], &mut Offsets)> {
    let find_indexes = get_find_indexes(ifmt);
    OffsetChunkProducer {
        name: name.into(),
        inner: Reader::new(r, chunk_size, /*padding=*/ 128),
        find_indexes: move |bs: &[u8], offs: &mut Offsets| {
            unsafe { find_indexes(bs, offs, 0, 0) };
        },
        record_sep: b'\n',
        cur_file_version: start_version,
        state: ChunkState::Init,
    }
}

pub fn new_offset_chunk_producer_bytes<R: Read>(
    r: R,
    chunk_size: usize,
    name: &str,
    field_sep: u8,
    record_sep: u8,
    start_version: u32,
) -> OffsetChunkProducer<R, impl FnMut(&[u8], &mut Offsets)> {
    let find_indexes = get_find_indexes_bytes();
    OffsetChunkProducer {
        name: name.into(),
        inner: Reader::new(r, chunk_size, /*padding=*/ 128),
        find_indexes: move |bs: &[u8], offs: &mut Offsets| unsafe {
            find_indexes(bs, offs, field_sep, record_sep)
        },
        cur_file_version: start_version,
        record_sep,
        state: ChunkState::Init,
    }
}

pub fn new_offset_chunk_producer_ascii_whitespace<R: Read>(
    r: R,
    chunk_size: usize,
    name: &str,
    start_version: u32,
) -> WhitespaceChunkProducer<R, impl FnMut(&[u8], &mut WhitespaceOffsets, u64) -> u64> {
    let find_indexes = get_find_indexes_ascii_whitespace();
    WhitespaceChunkProducer(
        OffsetChunkProducer {
            name: name.into(),
            inner: Reader::new(r, chunk_size, /*padding=*/ 128),
            find_indexes: move |bs: &[u8], offs: &mut WhitespaceOffsets, start: u64| unsafe {
                find_indexes(bs, offs, start)
            },
            cur_file_version: start_version,
            record_sep: 0u8, // unused
            state: ChunkState::Init,
        },
        1,
    )
}

pub fn new_chained_offset_chunk_producer_csv<
    'a,
    R: Read,
    N: Borrow<str>,
    I: Iterator<Item = (R, N)>,
>(
    r: I,
    chunk_size: usize,
    ifmt: InputFormat,
) -> ChainedChunkProducer<OffsetChunkProducer<R, impl FnMut(&[u8], &mut Offsets)>> {
    ChainedChunkProducer(
        r.enumerate()
            .map(|(i, (r, name))| {
                new_offset_chunk_producer_csv(
                    r,
                    chunk_size,
                    name.borrow(),
                    ifmt,
                    /*start_version=*/ (i as u32).wrapping_add(1),
                )
            })
            .collect(),
    )
}

pub fn new_chained_offset_chunk_producer_bytes<
    R: Read,
    N: Borrow<str>,
    I: Iterator<Item = (R, N)>,
>(
    r: I,
    chunk_size: usize,
    field_sep: u8,
    record_sep: u8,
) -> ChainedChunkProducer<OffsetChunkProducer<R, impl FnMut(&[u8], &mut Offsets)>> {
    ChainedChunkProducer(
        r.enumerate()
            .map(|(i, (r, name))| {
                new_offset_chunk_producer_bytes(
                    r,
                    chunk_size,
                    name.borrow(),
                    field_sep,
                    record_sep,
                    /*start_version=*/ (i as u32).wrapping_add(1),
                )
            })
            .collect(),
    )
}

pub fn new_chained_offset_chunk_producer_ascii_whitespace<
    R: Read,
    N: Borrow<str>,
    I: Iterator<Item = (R, N)>,
>(
    r: I,
    chunk_size: usize,
) -> ChainedChunkProducer<
    WhitespaceChunkProducer<R, impl FnMut(&[u8], &mut WhitespaceOffsets, u64) -> u64>,
> {
    ChainedChunkProducer(
        r.enumerate()
            .map(|(i, (r, name))| {
                new_offset_chunk_producer_ascii_whitespace(
                    r,
                    chunk_size,
                    name.borrow(),
                    /*start_version=*/ (i as u32).wrapping_add(1),
                )
            })
            .collect(),
    )
}

impl<C: Chunk> ChunkProducer for Box<dyn ChunkProducer<Chunk = C>> {
    type Chunk = C;
    fn try_dyn_resize(
        &self,
        requested_size: usize,
    ) -> Vec<Box<dyn FnOnce() -> Box<dyn ChunkProducer<Chunk = C>> + Send>> {
        (&**self).try_dyn_resize(requested_size)
    }
    fn next_file(&mut self) -> Result<bool> {
        (&mut **self).next_file()
    }
    fn get_chunk(&mut self, chunk: &mut C) -> Result<bool> {
        (&mut **self).get_chunk(chunk)
    }
}

// TODO: WhitepsaceOffsetChunk{,Producer}

pub struct OffsetChunk<Off = Offsets> {
    pub version: u32,
    pub name: Arc<str>,
    pub buf: Option<UniqueBuf>,
    pub len: usize,
    pub off: Off,
}

impl<Off: Default> Default for OffsetChunk<Off> {
    fn default() -> OffsetChunk<Off> {
        OffsetChunk {
            version: 0,
            name: "".into(),
            buf: None,
            len: 0,
            off: Default::default(),
        }
    }
}

impl<Off: Default + Send> Chunk for OffsetChunk<Off> {
    fn get_name(&self) -> &str {
        &*self.name
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
                    chunk.version = self.cur_file_version;
                    chunk.name = self.name.clone();
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
}

pub struct WhitespaceChunkProducer<R, F>(OffsetChunkProducer<R, F>, u64);

impl<R: Read, F: FnMut(&[u8], &mut WhitespaceOffsets, u64) -> u64> ChunkProducer
    for WhitespaceChunkProducer<R, F>
{
    type Chunk = OffsetChunk<WhitespaceOffsets>;
    fn next_file(&mut self) -> Result<bool> {
        self.0.state = ChunkState::Done;
        self.0.inner.force_eof();
        Ok(false)
    }
    fn get_chunk(&mut self, chunk: &mut Self::Chunk) -> Result<bool> {
        loop {
            match self.0.state {
                ChunkState::Init => {
                    self.0.state = if self.0.inner.reset()? {
                        ChunkState::Done
                    } else {
                        ChunkState::Main
                    };
                }
                ChunkState::Main => {
                    chunk.version = self.0.cur_file_version;
                    chunk.name = self.0.name.clone();
                    let buf = self.0.inner.buf.clone();
                    let bs = buf.as_bytes();
                    self.1 = (self.0.find_indexes)(bs, &mut chunk.off, self.1);
                    // Find the last newline in the buffer, if there is one.
                    let (is_partial, truncate_to) =
                        if let Some(nl_off) = chunk.off.nl.fields.last().cloned() {
                            self.0.inner.start = nl_off as usize + 1;
                            let mut start = chunk.off.ws.fields.len() as isize - 1;
                            while start > 0 {
                                if chunk.off.ws.fields[start as usize] > nl_off as u64 {
                                    start -= 1;
                                } else {
                                    break;
                                }
                            }
                            (false, start as usize)
                        } else {
                            (true, 0)
                        };
                    chunk.len = self.0.inner.end;
                    let is_eof = self.0.inner.reset()?;
                    return match (is_partial, is_eof) {
                        (false, false) => {
                            // Yield buffer, stay in main.
                            chunk.buf = Some(buf.try_unique().unwrap());
                            chunk.off.ws.fields.truncate(truncate_to);
                            Ok(false)
                        }
                        (false, true) | (true, true) => {
                            // Yield the entire buffer, this was the last piece of data.
                            self.0.inner.clear_buf();
                            chunk.buf = Some(buf.try_unique().unwrap());
                            self.0.state = ChunkState::Done;
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
}

pub struct ChainedChunkProducer<P>(Vec<P>);
impl<P: ChunkProducer> ChunkProducer for ChainedChunkProducer<P> {
    type Chunk = P::Chunk;

    fn next_file(&mut self) -> Result<bool> {
        if let Some(cur) = self.0.last_mut() {
            if !cur.next_file()? {
                let _last = self.0.pop();
                debug_assert!(_last.is_some());
            }
            Ok(self.0.len() != 0)
        } else {
            Ok(false)
        }
    }

    fn get_chunk(&mut self, chunk: &mut P::Chunk) -> Result<bool> {
        while let Some(cur) = self.0.last_mut() {
            if !cur.get_chunk(chunk)? {
                return Ok(false);
            }
            let _last = self.0.pop();
            debug_assert!(_last.is_some());
        }
        Ok(true)
    }
}

/// ParallelChunkProducer allows for consumption of individual chunks from a ChunkProducer in
/// parallel.
pub struct ParallelChunkProducer<P: ChunkProducer> {
    incoming: Receiver<P::Chunk>,
    spent: Sender<P::Chunk>,
}

impl<P: ChunkProducer> Clone for ParallelChunkProducer<P> {
    fn clone(&self) -> ParallelChunkProducer<P> {
        ParallelChunkProducer {
            incoming: self.incoming.clone(),
            spent: self.spent.clone(),
        }
    }
}

impl<P: ChunkProducer + 'static> ParallelChunkProducer<P> {
    pub fn new(
        p_factory: impl FnOnce() -> P + Send + 'static,
        chan_size: usize,
    ) -> ParallelChunkProducer<P> {
        let (in_sender, in_receiver) = bounded(chan_size);
        let (spent_sender, spent_receiver) = bounded(chan_size);
        std::thread::spawn(move || {
            let mut p = p_factory();
            loop {
                let mut chunk = spent_receiver
                    .try_recv()
                    .ok()
                    .unwrap_or_else(P::Chunk::default);
                let chunk_res = p.get_chunk(&mut chunk);
                if chunk_res.is_err() || matches!(chunk_res, Ok(true)) {
                    return;
                }
                if in_sender.send(chunk).is_err() {
                    return;
                }
            }
        });
        ParallelChunkProducer {
            incoming: in_receiver,
            spent: spent_sender,
        }
    }
}

impl<P: ChunkProducer + 'static> ChunkProducer for ParallelChunkProducer<P> {
    type Chunk = P::Chunk;
    fn try_dyn_resize(
        &self,
        requested_size: usize,
    ) -> Vec<Box<dyn FnOnce() -> Box<dyn ChunkProducer<Chunk = Self::Chunk>> + Send>> {
        let mut res = Vec::with_capacity(requested_size);
        for _ in 0..requested_size {
            let p = self.clone();
            res.push(Box::new(move || Box::new(p) as Box<dyn ChunkProducer<Chunk = P::Chunk>>) as _)
        }
        res
    }
    fn next_file(&mut self) -> Result<bool> {
        err!("nextfile is not supported in record-oriented parallel mode")
    }
    fn get_chunk(&mut self, chunk: &mut P::Chunk) -> Result<bool> {
        if let Ok(mut new_chunk) = self.incoming.recv() {
            mem::swap(chunk, &mut new_chunk);
            let _ = self.spent.try_send(new_chunk);
            Ok(false)
        } else {
            Ok(true)
        }
    }
}

enum ProducerState<T> {
    Init,
    Main(T),
    Done,
}

/// ShardedChunkProducer allows consuption of entire chunk producers in parallel
pub struct ShardedChunkProducer<P> {
    incoming: Receiver<Box<dyn FnOnce() -> P + Send>>,
    state: ProducerState<P>,
}

impl<P: ChunkProducer + 'static> ShardedChunkProducer<P> {
    pub fn new<Iter>(ps: Iter) -> ShardedChunkProducer<P>
    where
        Iter: Iterator + 'static + Send,
        Iter::Item: FnOnce() -> P + 'static + Send,
    {
        // These are usually individual files, which should be fairly large, so we hard-code a
        // small buffer.
        let (sender, receiver) = bounded(1);
        std::thread::spawn(move || {
            for p_factory in ps {
                let to_send: Box<dyn FnOnce() -> P + Send> = Box::new(p_factory);
                if sender.send(to_send).is_err() {
                    return;
                }
            }
        });
        ShardedChunkProducer {
            incoming: receiver,
            state: ProducerState::Init,
        }
    }

    fn refresh_producer(&mut self) -> bool {
        let next = if let Ok(p) = self.incoming.recv() {
            p
        } else {
            self.state = ProducerState::Done;
            return false;
        };
        self.state = ProducerState::Main(next());
        true
    }
}

impl<P: ChunkProducer + 'static> ChunkProducer for ShardedChunkProducer<P> {
    type Chunk = P::Chunk;
    fn try_dyn_resize(
        &self,
        requested_size: usize,
    ) -> Vec<Box<dyn FnOnce() -> Box<dyn ChunkProducer<Chunk = Self::Chunk>> + Send>> {
        let mut res = Vec::with_capacity(requested_size);
        for _ in 0..requested_size {
            let incoming = self.incoming.clone();
            res.push(Box::new(move || {
                Box::new(ShardedChunkProducer {
                    incoming,
                    state: ProducerState::Init,
                }) as Box<dyn ChunkProducer<Chunk = P::Chunk>>
            }) as _)
        }
        res
    }
    fn next_file(&mut self) -> Result<bool> {
        match &mut self.state {
            ProducerState::Init => Ok(self.refresh_producer()),
            ProducerState::Done => Ok(false),
            ProducerState::Main(p) => Ok(p.next_file()? || self.refresh_producer()),
        }
    }
    fn get_chunk(&mut self, chunk: &mut Self::Chunk) -> Result<bool> {
        loop {
            match &mut self.state {
                ProducerState::Main(p) => {
                    if !p.get_chunk(chunk)? {
                        return Ok(false);
                    }
                    self.refresh_producer()
                }
                ProducerState::Init => self.refresh_producer(),
                ProducerState::Done => return Ok(true),
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Basic machinery to turn an iterator into a ChunkProducer. This makes it easier to unit test
    // the "derived ChunkProducers" like ParallelChunkProducer.

    fn new_iter(
        low: usize,
        high: usize,
        name: &str,
    ) -> impl FnOnce() -> IterChunkProducer<std::ops::Range<usize>> {
        let name: Arc<str> = name.into();
        move || IterChunkProducer {
            iter: (low..high),
            name,
        }
    }

    struct IterChunkProducer<I> {
        iter: I,
        name: Arc<str>,
    }

    struct ItemChunk<T> {
        item: T,
        name: Arc<str>,
    }

    impl<T: Default> Default for ItemChunk<T> {
        fn default() -> ItemChunk<T> {
            ItemChunk {
                item: Default::default(),
                name: "".into(),
            }
        }
    }

    impl<T: Send + Default> Chunk for ItemChunk<T> {
        fn get_name(&self) -> &str {
            &*self.name
        }
    }

    impl<I: Iterator> ChunkProducer for IterChunkProducer<I>
    where
        I::Item: Send + Default,
    {
        type Chunk = ItemChunk<I::Item>;
        fn next_file(&mut self) -> Result<bool> {
            // clear remaining items
            while let Some(_) = self.iter.next() {}
            Ok(false)
        }
        fn get_chunk(&mut self, chunk: &mut ItemChunk<I::Item>) -> Result<bool> {
            if let Some(item) = self.iter.next() {
                chunk.item = item;
                chunk.name = self.name.clone();
                Ok(false)
            } else {
                Ok(true)
            }
        }
    }

    #[test]
    fn chained_all_elements() {
        let mut chained_producer = ChainedChunkProducer(vec![
            new_iter(20, 30, "file3")(),
            new_iter(10, 20, "file2")(),
            new_iter(0, 10, "file1")(),
        ]);
        let mut got = Vec::new();
        let mut names = Vec::new();
        let mut chunk = ItemChunk::default();
        let mut i = 0;
        while !chained_producer
            .get_chunk(&mut chunk)
            .expect("get_chunk should succeed")
        {
            if i % 10 == 0 {
                names.push(chunk.name.clone())
            }
            got.push(chunk.item);
            i += 1;
        }

        assert_eq!(got, (0..30).collect::<Vec<_>>());
        assert_eq!(names, vec!["file1".into(), "file2".into(), "file3".into()]);
    }

    #[test]
    fn chained_next_file() {
        let mut chained_producer = ChainedChunkProducer(vec![
            new_iter(20, 30, "file3")(),
            new_iter(10, 20, "file2")(),
            new_iter(0, 10, "file1")(),
        ]);
        let mut got = Vec::new();
        let mut names = Vec::new();
        let mut chunk = ItemChunk::default();
        while !chained_producer
            .get_chunk(&mut chunk)
            .expect("get_chunk should succeed")
        {
            names.push(chunk.name.clone());
            got.push(chunk.item);
            if !chained_producer
                .next_file()
                .expect("next_file should succeed")
            {
                break;
            }
        }

        assert_eq!(got, vec![0, 10, 20]);
        assert_eq!(names, vec!["file1".into(), "file2".into(), "file3".into()]);
    }

    #[test]
    fn sharded_next_file() {
        let mut sharded_producer = ShardedChunkProducer::new(
            vec![
                new_iter(0, 10, "file1"),
                new_iter(10, 20, "file2"),
                new_iter(20, 30, "file3"),
            ]
            .into_iter(),
        );
        let mut got = Vec::new();
        let mut names = Vec::new();
        let mut chunk = ItemChunk::default();
        while !sharded_producer
            .get_chunk(&mut chunk)
            .expect("get_chunk should succeed")
        {
            names.push(chunk.name.clone());
            got.push(chunk.item);
            if !sharded_producer
                .next_file()
                .expect("next_file should succeed")
            {
                break;
            }
        }

        assert_eq!(got, vec![0, 10, 20]);
        assert_eq!(names, vec!["file1".into(), "file2".into(), "file3".into()]);
    }

    #[test]
    fn parallel_all_elements() {
        use std::{sync::Mutex, thread};
        let parallel_producer =
            ParallelChunkProducer::new(new_iter(0, 100, "file1"), /*chan_size=*/ 10);
        let got = Arc::new(Mutex::new(Vec::new()));
        let threads = {
            let _guard = got.lock().unwrap();
            let mut threads = Vec::with_capacity(5);
            for prod in parallel_producer.try_dyn_resize(5) {
                let got = got.clone();
                threads.push(thread::spawn(move || {
                    let mut prod = prod();
                    let mut chunk = ItemChunk::default();
                    while !prod
                        .get_chunk(&mut chunk)
                        .expect("get_chunk should succeed")
                    {
                        assert_eq!(chunk.name, "file1".into());
                        got.lock().unwrap().push(chunk.item);
                    }
                }));
            }
            threads
        };
        for t in threads.into_iter() {
            t.join().unwrap();
        }

        let mut g = got.lock().unwrap();
        g.sort();

        assert_eq!(*g, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn sharded_all_elements() {
        use std::{sync::Mutex, thread};
        let sharded_producer = ShardedChunkProducer::new(
            vec![
                new_iter(0, 10, "file1"),
                new_iter(10, 20, "file2"),
                new_iter(20, 30, "file3"),
                new_iter(30, 40, "file4"),
                new_iter(40, 50, "file5"),
                new_iter(50, 60, "file6"),
            ]
            .into_iter(),
        );
        let got = Arc::new(Mutex::new(Vec::new()));
        let threads = {
            let _guard = got.lock().unwrap();
            let mut threads = Vec::with_capacity(5);
            for prod in sharded_producer.try_dyn_resize(5) {
                let got = got.clone();
                threads.push(thread::spawn(move || {
                    let mut prod = prod();
                    let mut chunk = ItemChunk::default();
                    while !prod
                        .get_chunk(&mut chunk)
                        .expect("get_chunk should succeed")
                    {
                        let expected_name = match chunk.item {
                            0..=9 => "file1",
                            10..=19 => "file2",
                            20..=29 => "file3",
                            30..=39 => "file4",
                            40..=49 => "file5",
                            50..=59 => "file6",
                            x => panic!("unexpected item {} (should be in range [0,59])", x),
                        };
                        assert_eq!(&*chunk.name, expected_name);
                        got.lock().unwrap().push(chunk.item);
                    }
                }));
            }
            threads
        };
        for t in threads.into_iter() {
            t.join().unwrap();
        }

        let mut g = got.lock().unwrap();
        g.sort();

        assert_eq!(*g, (0..60).collect::<Vec<_>>());
    }

    // TODO: test that we get all elements in Chained, Sharded and Parallel chunkproducers.
    // TODO: test nextfile behavior for Chained and Sharded chunk producer.
}
