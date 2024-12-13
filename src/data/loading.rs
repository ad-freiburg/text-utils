use crate::data::{Batch, Pipeline, TrainData};
use crate::utils::{find_subsequences_of_max_size_k, py_invalid_type_error};
use anyhow::{anyhow, Context};
use log::warn;
use pyo3::prelude::*;
use rand::distributions::WeightedIndex;
use rand::prelude::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::panic;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::{Arc, Mutex};
use std::thread::Builder;

fn open(p: &Path) -> anyhow::Result<File> {
    File::open(p).with_context(|| format!("could not open file at {:}", p.display()))
}

fn count_lines(p: impl AsRef<Path>) -> anyhow::Result<usize> {
    Ok(LossyUtf8Reader::new(BufReader::new(open(p.as_ref())?))
        .lines()
        .count())
}

pub struct LossyUtf8Reader<R>
where
    R: BufRead,
{
    reader: R,
}

impl<R> LossyUtf8Reader<R>
where
    R: BufRead,
{
    pub fn new(reader: R) -> Self {
        LossyUtf8Reader { reader }
    }

    pub fn lines(self) -> LossyUtf8Lines<R> {
        LossyUtf8Lines {
            reader: self.reader,
        }
    }
}

pub struct LossyUtf8Lines<R>
where
    R: BufRead,
{
    reader: R,
}

impl<R> LossyUtf8Lines<R>
where
    R: BufRead,
{
    pub fn new(reader: R) -> Self {
        LossyUtf8Lines { reader }
    }
}

impl<R> Iterator for LossyUtf8Lines<R>
where
    R: BufRead,
{
    type Item = anyhow::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buf = vec![];
        match self.reader.read_until(b'\n', &mut buf) {
            Ok(0) => None,
            Ok(_) => {
                // remove \n or \r\n from line
                buf.pop();
                if !buf.is_empty() && *buf.last().unwrap() == b'\r' {
                    buf.pop();
                }
                let s = String::from_utf8_lossy(&buf);
                Some(Ok(s.to_string()))
            }
            Err(e) => Some(Err(anyhow!("failed to read line: {e}"))),
        }
    }
}

#[derive(Debug)]
pub struct ExactSizeGenerator<I> {
    iter: I,
    len: usize,
}

impl<I> ExactSizeIterator for ExactSizeGenerator<I> where I: Iterator {}

impl<I> Iterator for ExactSizeGenerator<I>
where
    I: Iterator,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

pub type MaybeTrainData = anyhow::Result<TrainData>;
pub type TrainDataGenerator = Box<dyn ExactSizeIterator<Item = MaybeTrainData> + Send + 'static>;

pub fn train_data_generator_from_jsonl(
    path: impl AsRef<Path>,
) -> anyhow::Result<TrainDataGenerator> {
    let len = count_lines(path.as_ref())?;
    let file_iter = LossyUtf8Reader::new(BufReader::new(open(path.as_ref())?)).lines();
    let iter = file_iter.map(|line| {
        line.and_then(|s| {
            let jsonl: serde_json::Value = serde_json::from_str(&s)
                .with_context(|| format!("failed to parse json line: {s}"))?;
            let data = match &jsonl {
                Value::Object(map) => {
                    let input = match map.get("input") {
                        None => {
                            return Err(anyhow!(
                                "key 'input' not found in json line: {jsonl}"
                            ))
                        },
                        Some(Value::String(input)) => input.to_string(),
                        Some(val) => {
                            return Err(anyhow!(
                                "key 'input' must be a string, got: {}", val
                            ))
                        }
                    };
                    let target = match map.get("target") {
                        None => None,
                        Some(Value::String(target)) => Some(target.to_string()),
                        Some(val) => {
                            return Err(anyhow!(
                                "key 'target' must be a string, got: {}", val
                            ))
                        }
                    };
                    TrainData::new(input, target)
                },
                _ => {
                    return Err(anyhow!(
                        "json line must be an object with key 'input' and optional key 'target', got: {s}"
                    ))
                },
            };
            Ok(data)
        })
    });
    Ok(Box::new(ExactSizeGenerator { iter, len }))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GenerationStrategy {
    Sequential,
    Interleaved,
    Weighted,
}

impl<'a> FromPyObject<'a> for GenerationStrategy {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        let s: String = ob.extract()?;
        let strategy = match s.as_str() {
            "sequential" => GenerationStrategy::Sequential,
            "interleaved" => GenerationStrategy::Interleaved,
            "weighted" => GenerationStrategy::Weighted,
            k => return Err(py_invalid_type_error(k, "text iteration strategy")),
        };
        Ok(strategy)
    }
}

pub struct MultiTrainDataGenerator {
    generators: Vec<TrainDataGenerator>,
    lengths: Vec<usize>,
    total_len: usize,
    strategy: GenerationStrategy,
    idx: usize,
    rng: ChaCha8Rng,
    finished: Vec<bool>,
}

impl MultiTrainDataGenerator {
    pub fn new(
        generators: Vec<TrainDataGenerator>,
        strategy: GenerationStrategy,
        seed: Option<u64>,
    ) -> anyhow::Result<Self> {
        let lengths: Vec<usize> = generators.iter().map(|g| g.len()).collect();
        if strategy == GenerationStrategy::Weighted && lengths.iter().any(|l| *l == 0) {
            return Err(anyhow!(
                "for the weighted iteration strategy all text generators must specify a positive \
            minimum length, otherwise they would never be iterated"
            ));
        }
        let finished = vec![false; generators.len()];
        Ok(MultiTrainDataGenerator {
            generators,
            total_len: lengths.iter().sum(),
            lengths,
            strategy,
            idx: 0,
            rng: if let Some(seed) = seed {
                ChaCha8Rng::seed_from_u64(seed)
            } else {
                ChaCha8Rng::from_entropy()
            },
            finished,
        })
    }

    fn next_idx(&mut self) {
        assert!(!self.all_finished());
        match self.strategy {
            GenerationStrategy::Sequential => {
                if self.finished[self.idx] {
                    self.idx = (self.idx + 1) % self.finished.len();
                }
            }
            GenerationStrategy::Interleaved => {
                let mut idx = self.idx;
                while idx == self.idx || self.finished[idx] {
                    idx = (idx + 1) % self.finished.len()
                }
                self.idx = idx;
            }
            GenerationStrategy::Weighted => {
                let non_finished_indices: Vec<usize> = self
                    .finished
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, f)| if !f { Some(idx) } else { None })
                    .collect();
                let dist = WeightedIndex::new(
                    non_finished_indices
                        .iter()
                        .map(|idx| self.lengths[*idx])
                        .collect::<Vec<usize>>(),
                )
                .expect("could not create line distribution");
                self.idx = non_finished_indices[self.rng.sample(dist)];
            }
        };
    }

    fn all_finished(&self) -> bool {
        self.finished.iter().all(|f| *f)
    }

    pub fn len(&self) -> usize {
        self.total_len
    }

    pub fn is_empty(&self) -> bool {
        self.total_len == 0
    }
}

impl Iterator for MultiTrainDataGenerator {
    type Item = (MaybeTrainData, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let data = loop {
            let next = self.generators[self.idx].next();
            match next {
                Some(v) => break v,
                None => {
                    self.finished[self.idx] = true;
                    if self.all_finished() {
                        return None;
                    }
                    self.next_idx();
                }
            };
        };
        let value = (data, self.idx);
        self.next_idx();
        Some(value)
    }
}

pub trait PipelineIterator<I, O>
where
    Self: Iterator<Item = I> + Send + 'static,
    I: Send + 'static,
    O: Send + 'static,
{
    fn pipe(self, pipeline: Pipeline<I, O>, num_threads: u8) -> Pipe<O>;
}

impl<It, I, O> PipelineIterator<I, O> for It
where
    It: Iterator<Item = I> + Send + 'static,
    I: Send + 'static,
    O: Send + 'static,
{
    fn pipe(self, pipeline: Pipeline<I, O>, num_threads: u8) -> Pipe<O> {
        Pipe::new(self, pipeline, num_threads)
    }
}

pub struct Pipe<O>
where
    O: Send + 'static,
{
    rx: Receiver<O>,
}

impl<O> Pipe<O>
where
    O: Send + 'static,
{
    fn new<I: Send + 'static>(
        inner: impl Iterator<Item = I> + Send + 'static,
        pipeline: Pipeline<I, O>,
        num_threads: u8,
    ) -> Self {
        let num_threads = num_threads.clamp(1, num_cpus::get() as u8);
        let inner = Arc::new(Mutex::new(inner.enumerate()));
        let (tx, rx) = sync_channel(num_threads as usize);
        let sent_counter = Arc::new(AtomicUsize::new(0));
        panic::set_hook(Box::new(move |info| {
            warn!("Thread panicked: {info}");
            std::process::exit(1);
        }));
        for thread in 0..num_threads {
            let inner_clone = inner.clone();
            let tx_clone = tx.clone();
            let pipeline_clone = pipeline.clone();
            let send_next = sent_counter.clone();
            let _ = Builder::new()
                .name(format!("pipeline worker thread {thread}"))
                .spawn(move || {
                    loop {
                        let Some((idx, data)) =
                            inner_clone.lock().expect("failed to lock receiver").next()
                        else {
                            return;
                        };
                        let item = pipeline_clone(data);
                        // wait until we are the next to send out item
                        while send_next.load(Ordering::SeqCst) != idx {
                            // sleep(Duration::from_micros(100));
                        }
                        let send_result = tx_clone.send(item);
                        send_next.swap(idx + 1, Ordering::SeqCst);
                        if send_result.is_err() {
                            // receiver is closed, so we can return this thread
                            return;
                        }
                    }
                })
                .unwrap_or_else(|_| panic!("failed building worker thread {thread}"));
        }
        Pipe { rx }
    }
}

impl<O> Iterator for Pipe<O>
where
    O: Send + Sync + 'static,
{
    type Item = O;

    fn next(&mut self) -> Option<Self::Item> {
        self.rx.recv().ok()
    }
}

#[derive(Copy, Clone, PartialEq)]
pub enum BatchLimitType {
    BatchSize,
    PaddedItemSize,
}

impl<'a> FromPyObject<'a> for BatchLimitType {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        let s: String = ob.extract()?;
        let limit_type = match s.as_str() {
            "batch_size" => BatchLimitType::BatchSize,
            "padded_item_size" => BatchLimitType::PaddedItemSize,
            k => return Err(py_invalid_type_error(k, "batch limit")),
        };
        Ok(limit_type)
    }
}

pub trait ItemSize {
    fn size(&self) -> usize;
}

pub trait BatchedIterator<T>
where
    Self: Iterator<Item = T> + Sized,
    T: ItemSize,
{
    fn batched(
        self,
        sort: bool,
        shuffle: bool,
        prefetch_factor: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        seed: Option<u64>,
    ) -> Batched<Self, T>;
}

impl<I, T> BatchedIterator<T> for I
where
    Self: Iterator<Item = T>,
    T: ItemSize,
{
    fn batched(
        self,
        sort: bool,
        shuffle: bool,
        prefetch_factor: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        seed: Option<u64>,
    ) -> Batched<I, T> {
        Batched::new(
            self,
            sort,
            shuffle,
            prefetch_factor,
            batch_limit,
            batch_limit_type,
            seed,
        )
    }
}

pub struct Batched<I, T> {
    iter: I,
    rng: ChaCha8Rng,
    batch_limit: usize,
    batch_limit_type: BatchLimitType,
    prefetch_factor: usize,
    sort: bool,
    shuffle: bool,
    shuffle_buffer: Vec<T>,
}

impl<I, T> Batched<I, T> {
    pub fn new(
        iter: I,
        sort: bool,
        shuffle: bool,
        prefetch_factor: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        seed: Option<u64>,
    ) -> Self {
        let batch_limit = batch_limit.max(1);
        let prefetch_factor = prefetch_factor.max(1);
        Self {
            rng: if let Some(seed) = seed {
                ChaCha8Rng::seed_from_u64(seed)
            } else {
                ChaCha8Rng::from_entropy()
            },
            batch_limit,
            batch_limit_type,
            prefetch_factor,
            sort,
            shuffle,
            shuffle_buffer: Vec::new(),
            iter,
        }
    }
}

impl<I, T> Batched<I, T>
where
    I: Iterator<Item = T>,
    T: ItemSize,
{
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn build_batch(
        iter: &mut I,
        buf: &mut Vec<T>,
        rng: &mut ChaCha8Rng,
        sort: bool,
        shuffle: bool,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        prefetch_factor: usize,
    ) -> Option<Batch<T>> {
        if !sort && !shuffle {
            let (batch, remainder) = Self::batch_from(
                || {
                    // pop from buffer first if not empty, otherwise resume with iterator
                    if !buf.is_empty() {
                        // there should only be 1 element in the buffer,
                        // which is the remainder saved below
                        assert_eq!(buf.len(), 1);
                        buf.pop()
                    } else {
                        iter.next()
                    }
                },
                batch_limit,
                batch_limit_type,
            );
            // save remainder in buffer
            if let Some(remainder) = remainder {
                buf.push(remainder);
            }
            return batch;
        }
        // we create a batch in the following way:
        // 1. fill up the buffer
        // 2. maybe sort or shuffle the buffer
        // 3. pop from buffer until we have a maxed out batch
        let mut buffer_limit = BatchLimit::from_items(&buf[..], &batch_limit_type);
        // fill buffer until we overshoot batch limit * some factor (>=1)
        while buffer_limit.limit() <= batch_limit * prefetch_factor {
            let Some(item) = iter.next() else {
                break;
            };
            buffer_limit = buffer_limit.update(&item);
            buf.push(item);
        }
        // we are only done if the buffer is empty
        if buf.is_empty() {
            // return remainder as final batch if we still have one
            return None;
        }
        if sort {
            // sort the buffer by length
            buf.sort_by_key(|a| a.size());
            // if shuffle is also true, return a random subsequence from the sorted buffer
            if shuffle {
                // calculate possible subsequences for batch
                let sub_sequences =
                    find_subsequences_of_max_size_k(buf, batch_limit, |sub_items| {
                        BatchLimit::from_items(sub_items, &batch_limit_type).limit()
                    });
                // randomly choose one subsequence
                if sub_sequences.is_empty() {
                    // only happens if there is no single item that's smaller than the batch limit,
                    // since we always need to return at least one item, just return the last
                    // and set the second last to the remainder, this should always work
                    // since the shuffle buffer contains at least two items
                    let batch = vec![buf.pop().unwrap()];
                    return Some(batch);
                }
                let index = rng.gen_range(0..sub_sequences.len());
                let (start_range, end_range) = sub_sequences[index];
                let items_in_range: Vec<T> = buf.splice(start_range..end_range, vec![]).collect();
                return Some(items_in_range);
            }
        } else if shuffle {
            // shuffle the buffer
            buf.shuffle(rng);
        }
        // now pop from the back until batch is full
        let (batch, remainder) = Self::batch_from(|| buf.pop(), batch_limit, batch_limit_type);
        // push remainder back to buffer
        if let Some(remainder) = remainder {
            buf.push(remainder);
        }
        batch
    }

    #[inline]
    fn batch_from(
        mut f: impl FnMut() -> Option<T>,
        limit: usize,
        limit_type: BatchLimitType,
    ) -> (Option<Batch<T>>, Option<T>) {
        // the implementation makes sure that always at least 1 item is returned
        // in the batch, even if the item solely exceeds the specified limit
        let mut items = vec![];
        let mut batch_limit = BatchLimit::from_items(&items, &limit_type);
        let remainder = loop {
            let Some(item) = f() else {
                return if items.is_empty() {
                    (None, None)
                } else {
                    (Some(items), None)
                };
            };
            batch_limit = batch_limit.update(&item);
            if batch_limit.limit() > limit && !items.is_empty() {
                // if adding the item would overshoot
                // just return it as remainder
                break Some(item);
            } else {
                // if adding the item would not overshoot, just add it
                // and increase the batch limit counter
                items.push(item);
            }
        };
        (Some(items), remainder)
    }
}

impl<I, T> Iterator for Batched<I, T>
where
    I: Iterator<Item = T>,
    T: ItemSize,
{
    type Item = Batch<T>;

    fn next(&mut self) -> Option<Self::Item> {
        Self::build_batch(
            &mut self.iter,
            &mut self.shuffle_buffer,
            &mut self.rng,
            self.sort,
            self.shuffle,
            self.batch_limit,
            self.batch_limit_type,
            self.prefetch_factor,
        )
    }
}

#[derive(Debug, Clone)]
enum BatchLimit {
    BatchSize(usize),
    TotalItemSize(usize, usize),
}

impl BatchLimit {
    fn from_items(items: &[impl ItemSize], limit_type: &BatchLimitType) -> Self {
        match limit_type {
            BatchLimitType::BatchSize => Self::BatchSize(items.len()),
            BatchLimitType::PaddedItemSize => Self::TotalItemSize(
                items.len(),
                items.iter().map(|i| i.size()).max().unwrap_or(0),
            ),
        }
    }

    fn update(self, item: &impl ItemSize) -> Self {
        match self {
            BatchLimit::BatchSize(count) => BatchLimit::BatchSize(count + 1),
            BatchLimit::TotalItemSize(count, max_length) => {
                BatchLimit::TotalItemSize(count + 1, max_length.max(item.size()))
            }
        }
    }

    fn limit(&self) -> usize {
        match self {
            BatchLimit::BatchSize(count) => *count,
            BatchLimit::TotalItemSize(count, max_length) => *count * *max_length,
        }
    }
}

pub trait Tensorize {
    type Output;

    fn tensorize(&self) -> Self::Output;
}

pub trait TensorizedIterator
where
    Self: Iterator + Sized,
    Self::Item: Tensorize,
{
    fn tensorized(self) -> Tensorized<Self>;
}

impl<I> TensorizedIterator for I
where
    I: Iterator,
    I::Item: Tensorize,
{
    fn tensorized(self) -> Tensorized<Self> {
        Tensorized::new(self)
    }
}

pub struct Tensorized<I> {
    iter: I,
}

impl<I> Tensorized<I>
where
    I: Iterator,
    I::Item: Tensorize,
{
    pub fn new(iter: I) -> Self {
        Self { iter }
    }
}

impl<I> Iterator for Tensorized<I>
where
    I: Iterator,
    I::Item: Tensorize,
{
    type Item = (I::Item, <I::Item as Tensorize>::Output);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|b| {
            let tensorized = b.tensorize();
            (b, tensorized)
        })
    }
}

pub trait BufferedIterator<T>
where
    Self: Iterator<Item = T>,
{
    fn buffered(self, buffer_size: usize) -> Buffered<T>;
}

impl<I, T> BufferedIterator<T> for I
where
    I: Iterator<Item = T> + Send + 'static,
    T: Send + 'static,
{
    fn buffered(self, buffer_size: usize) -> Buffered<T> {
        Buffered::new(self, buffer_size)
    }
}

pub struct Buffered<T> {
    rx: Receiver<T>,
}

impl<T> Buffered<T>
where
    T: Send + 'static,
{
    pub fn new<I>(iter: I, buffer_size: usize) -> Self
    where
        I: Iterator<Item = T> + Send + 'static,
    {
        let (tx, rx) = sync_channel(buffer_size);
        let _ = Builder::new()
            .name("buffer thread".to_string())
            .spawn(move || {
                for item in iter {
                    tx.send(item).ok();
                }
            })
            .expect("failed to build buffer thread");
        Self { rx }
    }
}

impl<T> Iterator for Buffered<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.rx.recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use crate::data::loading::{
        BatchLimitType, BatchedIterator, GenerationStrategy, MultiTrainDataGenerator,
    };
    use crate::data::postprocessing::PostprocessingFnConfig;
    use crate::data::preprocessing::PreprocessingFnConfig;
    use crate::data::task::TrainTaskConfig;
    use crate::data::{
        train_pipeline, PostprocessingConfig, PreprocessingConfig, TextDataInfo, TrainData,
        TrainItem, TrainPipelineConfig,
    };
    use crate::tokenization::{SpecialConfig, TokenizeConfig, TokenizerConfig};
    use itertools::Itertools;
    use log::info;
    use std::{
        collections::HashMap,
        path::PathBuf,
        time::{Duration, Instant},
    };

    use super::{train_data_generator_from_jsonl, BufferedIterator, PipelineIterator};

    const MULTI30K_FIRST: &str = "Two young, White males are outside near many bushes.";
    const MULTI30K_SECOND: &str = "Several men in hard hats are operating a giant pulley system.";
    const MULTI30K_REV_FIRST: &str = "A man in shorts and a Hawaiian shirt leans over the rail of a pilot boat, with fog and mountains in the background.";
    const MULTI30K_REV_SECOND: &str =
        "An elderly man sits outside a storefront accompanied by a young boy with a cart.";

    #[test]
    fn test_text_iterator() -> anyhow::Result<()> {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let d = base.clone().join("resources/test/multi30k.jsonl");
        let d2 = base.clone().join("resources/test/multi30k_rev.jsonl");
        let multi30k = train_data_generator_from_jsonl(&d)?;
        let mut it =
            MultiTrainDataGenerator::new(vec![multi30k], GenerationStrategy::Sequential, None)?;

        // first check sequential lines with one file
        assert_eq!(it.len(), 29000);
        let _data = TrainData {
            target: MULTI30K_FIRST.to_string(),
            input: MULTI30K_FIRST.to_string(),
        };
        assert!(matches!(it.next().unwrap(), (Ok(_data), 0)));
        let _data = TrainData {
            target: MULTI30K_SECOND.to_string(),
            input: MULTI30K_SECOND.to_string(),
        };
        assert!(matches!(it.next().unwrap(), (Ok(_data), 0)));
        // check sequential lines with input and target
        let multi30k = train_data_generator_from_jsonl(&d)?;
        let mut it =
            MultiTrainDataGenerator::new(vec![multi30k], GenerationStrategy::Sequential, None)?;

        assert_eq!(it.len(), 29000);
        let _data = TrainData {
            target: MULTI30K_FIRST.to_string(),
            input: MULTI30K_REV_FIRST.to_string(),
        };
        assert!(matches!(it.next().unwrap(), (Ok(_data), 0)));
        let _data = TrainData {
            target: MULTI30K_SECOND.to_string(),
            input: MULTI30K_REV_SECOND.to_string(),
        };
        assert!(matches!(it.next().unwrap(), (Ok(_data), 0)));
        // check interleaved lines with two files
        let multi30k = train_data_generator_from_jsonl(&d)?;
        let multi30k_rev = train_data_generator_from_jsonl(&d2)?;
        let mut it = MultiTrainDataGenerator::new(
            vec![multi30k, multi30k_rev],
            GenerationStrategy::Interleaved,
            None,
        )?;

        assert_eq!(it.len(), 2 * 29000);
        let _data = TrainData {
            target: MULTI30K_FIRST.to_string(),
            input: MULTI30K_FIRST.to_string(),
        };
        assert!(matches!(it.next().unwrap(), (Ok(_data), 0)));
        let _data = TrainData {
            target: MULTI30K_REV_FIRST.to_string(),
            input: MULTI30K_REV_FIRST.to_string(),
        };
        assert!(matches!(it.next().unwrap(), (Ok(_data), 1)));
        let _data = TrainData {
            target: MULTI30K_SECOND.to_string(),
            input: MULTI30K_SECOND.to_string(),
        };
        assert!(matches!(it.next().unwrap(), (Ok(_data), 0)));
        let _data = TrainData {
            target: MULTI30K_REV_SECOND.to_string(),
            input: MULTI30K_REV_SECOND.to_string(),
        };
        assert!(matches!(it.next().unwrap(), (Ok(_data), 1)));
        // check weighted lines with two files
        let multi30k = train_data_generator_from_jsonl(&d)?;
        let multi30k_rev = train_data_generator_from_jsonl(&d2)?;
        let it = MultiTrainDataGenerator::new(
            vec![multi30k, multi30k_rev],
            GenerationStrategy::Weighted,
            None,
        )?;
        assert_eq!(it.len(), 2 * 29000);
        Ok(())
    }

    #[test]
    fn test_pipeline_iterator() -> anyhow::Result<()> {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let d = base.clone().join("resources/test/multi30k.jsonl");

        // create a pipeline that simulates some real processing,
        // we use the dummy tokenizer with a delay of 100 milliseconds for that
        let tokenizer_cfg = TokenizerConfig {
            tokenize: TokenizeConfig::Dummy(Duration::from_millis(200)),
            special: SpecialConfig::default(),
        };
        let (pipeline, _) = train_pipeline(
            TrainPipelineConfig {
                preprocessing: PreprocessingConfig::Global(PreprocessingFnConfig::None),
                task: TrainTaskConfig::WhitespaceCorrection(true, tokenizer_cfg.clone()),
                postprocessing: PostprocessingConfig::Global(PostprocessingFnConfig::None),
            },
            512,
        )?;
        // test if it works with one worker and record the time it took
        let multi30k = train_data_generator_from_jsonl(&d)?;
        let text_iter =
            MultiTrainDataGenerator::new(vec![multi30k], GenerationStrategy::Sequential, None)?;

        let it = text_iter
            .filter_map(|(d, _)| {
                if let Ok(d) = d {
                    Some((d, TextDataInfo::default()))
                } else {
                    None
                }
            })
            .pipe(pipeline.clone(), 0);
        let n: usize = 20;
        let now = Instant::now();
        let _: Vec<TrainItem> = it.filter_map(|d| d.ok()).take(n).collect();
        let mut time = now.elapsed().as_secs_f64();

        let n_cpus = num_cpus::get();

        // if more cpus are available, test with more workers, check that its faster
        if n_cpus >= 2 {
            let multi30k = train_data_generator_from_jsonl(&d)?;
            let text_iter =
                MultiTrainDataGenerator::new(vec![multi30k], GenerationStrategy::Sequential, None)?;
            let it = text_iter
                .filter_map(|(d, _)| {
                    if let Ok(d) = d {
                        Some((d, TextDataInfo::default()))
                    } else {
                        None
                    }
                })
                .pipe(pipeline.clone(), 2);
            let now = Instant::now();
            let _: Vec<TrainItem> = it.filter_map(|d| d.ok()).take(n).collect();
            let time2 = now.elapsed().as_secs_f64();
            assert!(time2 < time);
            time = time2;
        }

        // test with even more workers, if available
        if n_cpus >= 4 {
            let multi30k = train_data_generator_from_jsonl(&d)?;
            let text_iter =
                MultiTrainDataGenerator::new(vec![multi30k], GenerationStrategy::Sequential, None)?;
            let it = text_iter
                .filter_map(|(d, _)| {
                    if let Ok(d) = d {
                        Some((d, TextDataInfo::default()))
                    } else {
                        None
                    }
                })
                .pipe(pipeline.clone(), 4);
            let now = Instant::now();
            let _: Vec<TrainItem> = it.filter_map(|d| d.ok()).take(n).collect();
            let time3 = now.elapsed().as_secs_f64();
            info!("took {:.2}s to fetch {} items", time3, n);
            assert!(time3 < time);
        }

        // test that all lines of multi30k.jsonl are returned in order,
        // switch to non blocking tokenizer again
        let tokenizer_cfg = TokenizerConfig {
            tokenize: TokenizeConfig::Dummy(Duration::from_millis(0)),
            special: SpecialConfig::default(),
        };
        let (pipeline, _) = train_pipeline(
            TrainPipelineConfig {
                preprocessing: PreprocessingConfig::Global(PreprocessingFnConfig::None),
                task: TrainTaskConfig::WhitespaceCorrection(true, tokenizer_cfg.clone()),
                postprocessing: PostprocessingConfig::Global(PostprocessingFnConfig::None),
            },
            512,
        )?;
        let multi30k = train_data_generator_from_jsonl(&d)?;
        let text_iter =
            MultiTrainDataGenerator::new(vec![multi30k], GenerationStrategy::Sequential, None)?;

        let it = text_iter
            .filter_map(|(d, _)| {
                if let Ok(d) = d {
                    Some((d, TextDataInfo::default()))
                } else {
                    None
                }
            })
            .pipe(pipeline.clone(), 0);
        let data: Vec<_> = train_data_generator_from_jsonl(&d)?.collect::<anyhow::Result<_>>()?;
        let mut count = 0;
        for (item, data) in it.zip(&data) {
            assert_eq!(item.unwrap().data.target, data.target);
            count += 1;
        }
        assert_eq!(count, data.len());
        Ok(())
    }

    #[test]
    fn test_batched_pipeline_iterator() -> anyhow::Result<()> {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let d = base.clone().join("resources/test/multi30k.jsonl");
        let multi30k = train_data_generator_from_jsonl(&d)?;
        let text_iter =
            MultiTrainDataGenerator::new(vec![multi30k], GenerationStrategy::Sequential, None)?;
        let data: Vec<_> = train_data_generator_from_jsonl(&d)?.collect::<anyhow::Result<_>>()?;
        let tokenizer_cfg = TokenizerConfig {
            tokenize: TokenizeConfig::Dummy(Duration::from_millis(0)),
            special: SpecialConfig::default(),
        };
        let (pipeline, _) = train_pipeline(
            TrainPipelineConfig {
                preprocessing: PreprocessingConfig::Global(PreprocessingFnConfig::None),
                task: TrainTaskConfig::WhitespaceCorrection(true, tokenizer_cfg.clone()),
                postprocessing: PostprocessingConfig::Global(PostprocessingFnConfig::None),
            },
            512,
        )?;
        let pipe_it = text_iter
            .filter_map(|(d, _)| {
                if let Ok(d) = d {
                    Some((d, TextDataInfo::default()))
                } else {
                    None
                }
            })
            .pipe(pipeline.clone(), 4)
            .filter_map(|d| d.ok())
            .batched(false, false, 0, 8, BatchLimitType::BatchSize, None);
        for (batch, data_batch) in pipe_it.zip(&data.iter().chunks(8)) {
            assert!(batch.len() > 0 && batch.len() <= 8);
            let mut count = 0;
            for (item, data) in batch.iter().zip(data_batch) {
                assert_eq!(item.data.target, data.target);
                count += 1;
            }
            assert_eq!(count, batch.len());
        }
        // now check the batched iterator with any combinations of shuffling and sorting
        // check that each line was yielded by the batch iterator once or twice
        // (because some descriptions in multi30k appear twice)
        for shuffle in [true, false] {
            for sort in [true, false] {
                let multi30k = train_data_generator_from_jsonl(&d)?;
                let text_iter = MultiTrainDataGenerator::new(
                    vec![multi30k],
                    GenerationStrategy::Weighted,
                    None,
                )?;

                let pipe_it = text_iter
                    .filter_map(|(d, _)| {
                        if let Ok(d) = d {
                            Some((d, TextDataInfo::default()))
                        } else {
                            None
                        }
                    })
                    .pipe(pipeline.clone(), 4)
                    .filter_map(|d| d.ok())
                    .batched(
                        sort,
                        shuffle,
                        32,
                        256,
                        BatchLimitType::PaddedItemSize,
                        Some(22),
                    );
                let mut line_counter: HashMap<String, usize> =
                    data.iter().cloned().map(|d| (d.input, 0)).collect();
                for batch in pipe_it {
                    let item_lengths: Vec<usize> =
                        batch.iter().map(|item| item.input.len()).collect();
                    let batch_size: usize = item_lengths.iter().sum();
                    assert!(batch.len() > 0);
                    assert!(batch_size <= 256,);
                    for item in batch {
                        let count = line_counter.get_mut(&item.data.target).unwrap();
                        *count += 1;
                    }
                }
                assert!(
                    line_counter.iter().all(|(_, c)| *c == 1 || *c == 2)
                        && line_counter.iter().map(|(_, c)| *c).sum::<usize>() == data.len()
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_buffered_iterator() {
        let nums_in: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let nums_out: Vec<i32> = nums_in.clone().into_iter().buffered(3).collect();
        assert_eq!(nums_in, nums_out);
    }
}
