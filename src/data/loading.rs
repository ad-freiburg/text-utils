use crate::data::{Batch, InferenceData, Pipeline, TrainData};
use crate::utils::{find_subsequences_of_max_size_k, py_invalid_type_error};
use anyhow::{anyhow, Context};
use log::debug;
use pyo3::prelude::*;
use rand::distributions::WeightedIndex;
use rand::prelude::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::marker::PhantomData;
use std::panic;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::{Arc, Mutex};
use std::thread::{Builder, JoinHandle};
use std::time::Instant;

pub trait DataGen: Iterator + Send {
    fn min_len(&self) -> usize;
}

fn open(p: &Path) -> anyhow::Result<File> {
    File::open(p).with_context(|| format!("could not open file at {:}", p.display()))
}

fn count_lines(p: &Path) -> anyhow::Result<usize> {
    Ok(LossyUtf8Reader::new(BufReader::new(open(p)?))
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
pub struct DataGenerator<I> {
    iter: I,
    min_len: usize,
}

impl<I> Iterator for DataGenerator<I>
where
    I: Iterator + Send,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}
impl<I> DataGen for DataGenerator<I>
where
    I: Iterator + Send,
{
    fn min_len(&self) -> usize {
        self.min_len
    }
}

pub fn train_data_generator_from_files<P: AsRef<Path>>(
    input: P,
    target: Option<P>,
) -> anyhow::Result<Box<dyn DataGen<Item = anyhow::Result<TrainData>>>> {
    let input_len = count_lines(input.as_ref())?;
    let input_iter = LossyUtf8Reader::new(BufReader::new(open(input.as_ref())?)).lines();
    let mut target_iter = if let Some(target) = target {
        let target_len = count_lines(target.as_ref())?;
        assert_eq!(
            input_len,
            target_len,
            "expected same number of lines for {:?} and {:?}",
            input.as_ref(),
            target.as_ref()
        );
        Some(LossyUtf8Reader::new(BufReader::new(open(target.as_ref())?)).lines())
    } else {
        None
    };
    let iter = input_iter.map(move |input_s| {
        let target_s = if target_iter.is_some() {
            match target_iter.as_mut().unwrap().next() {
                Some(result) => Some(result?),
                None => None,
            }
        } else {
            None
        };
        Ok(TrainData::new(input_s?, target_s))
    });
    Ok(Box::new(DataGenerator {
        min_len: input_len,
        iter,
    }))
}

struct PythonIterator<T>
where
    T: for<'a> FromPyObject<'a>,
{
    iterator: PyObject,
    _phantom: PhantomData<T>,
}

impl<T> Iterator for PythonIterator<T>
where
    T: for<'a> FromPyObject<'a>,
{
    type Item = anyhow::Result<T>;

    fn next(&mut self) -> Option<Self::Item> {
        Python::with_gil(|py| {
            let mut it = match self.iterator.bind(py).iter() {
                Err(e) => {
                    return Some(Err(anyhow!("failed to get iterator: {e}")));
                }
                Ok(it) => it,
            };
            match it.next() {
                None => None,
                Some(Err(e)) => Some(Err(anyhow!(
                    "failed to extract next item from iterator: {e}",
                ))),
                Some(Ok(value)) => {
                    let data = value
                        .extract()
                        .map_err(|e| anyhow!("failed to extract data from iterator item: {e}"));
                    Some(data)
                }
            }
        })
    }
}

pub fn inference_data_iterator_from_python(
    iterator: PyObject,
) -> impl Iterator<Item = anyhow::Result<InferenceData>> {
    PythonIterator {
        iterator,
        _phantom: PhantomData,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TextIterationStrategy {
    Sequential,
    Interleaved,
    Weighted,
}

impl<'a> FromPyObject<'a> for TextIterationStrategy {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let s: String = ob.extract()?;
        let strategy = match s.as_str() {
            "sequential" => TextIterationStrategy::Sequential,
            "interleaved" => TextIterationStrategy::Interleaved,
            "weighted" => TextIterationStrategy::Weighted,
            k => return Err(py_invalid_type_error(k, "text iteration strategy")),
        };
        Ok(strategy)
    }
}

pub struct TextIterator<T> {
    text_generators: Vec<Box<dyn DataGen<Item = T>>>,
    lengths: Vec<usize>,
    strategy: TextIterationStrategy,
    idx: usize,
    rng: ChaCha8Rng,
    finished: Vec<bool>,
}

impl<T> TextIterator<T> {
    pub fn new(
        text_generators: Vec<Box<dyn DataGen<Item = T>>>,
        strategy: TextIterationStrategy,
        seed: Option<u64>,
    ) -> anyhow::Result<Self> {
        let lengths: Vec<usize> = text_generators.iter().map(|g| g.min_len()).collect();
        if strategy == TextIterationStrategy::Weighted && lengths.iter().any(|l| *l == 0) {
            return Err(anyhow!(
                "for the weighted iteration strategy all text generators must specify a positive \
            minimum length, otherwise they would never be iterated"
            ));
        }
        let finished = vec![false; text_generators.len()];
        Ok(TextIterator {
            text_generators,
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
            TextIterationStrategy::Sequential => {
                if self.finished[self.idx] {
                    self.idx = (self.idx + 1) % self.finished.len();
                }
            }
            TextIterationStrategy::Interleaved => {
                let mut idx = self.idx;
                while idx == self.idx || self.finished[idx] {
                    idx = (idx + 1) % self.finished.len()
                }
                self.idx = idx;
            }
            TextIterationStrategy::Weighted => {
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

    pub fn min_len(&self) -> usize {
        self.lengths.iter().sum()
    }
}

impl<T> Iterator for TextIterator<T> {
    type Item = (T, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let data = loop {
            let next = self.text_generators[self.idx].next();
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
    output: Receiver<O>,
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
            println!("thread panicked: {info}");
            std::process::exit(1);
        }));
        for thread in 0..num_threads {
            let inner_clone = inner.clone();
            let tx_clone = tx.clone();
            let pipeline_clone = pipeline.clone();
            let send_next = sent_counter.clone();
            let _: JoinHandle<()> = Builder::new()
                .name(format!("pipeline worker thread {thread}"))
                .spawn(move || {
                    loop {
                        let start = Instant::now();
                        let Some((idx, data)) =
                            inner_clone.lock().expect("failed to lock receiver").next()
                        else {
                            return;
                        };
                        debug!(
                            "thread {thread}: loading item {idx} took {:.2}ms",
                            start.elapsed().as_secs_f32() * 1000.0
                        );
                        let start = Instant::now();
                        let item = pipeline_clone(data);
                        debug!(
                            "thread {thread}: running pipeline on item {idx} took {:.2}ms",
                            start.elapsed().as_secs_f32() * 1000.0
                        );
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
        Pipe { output: rx }
    }
}

impl<O> Iterator for Pipe<O>
where
    O: Send + 'static,
{
    type Item = O;

    fn next(&mut self) -> Option<Self::Item> {
        self.output.recv().ok()
    }
}

#[derive(Copy, Clone, PartialEq)]
pub enum BatchLimitType {
    BatchSize,
    PaddedItemSize,
}

impl<'a> FromPyObject<'a> for BatchLimitType {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
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
    Self: Iterator<Item = T> + Send + 'static,
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
    ) -> Batched<T>;
}

impl<I, T> BatchedIterator<T> for I
where
    I: Iterator<Item = T> + Send + 'static,
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
    ) -> Batched<T> {
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

pub struct Batched<T> {
    inner: Box<dyn Iterator<Item = T> + Send + 'static>,
    rng: ChaCha8Rng,
    batch_limit: usize,
    batch_limit_type: BatchLimitType,
    prefetch_factor: usize,
    sort: bool,
    shuffle: bool,
    shuffle_buffer: Vec<T>,
}

impl<T> Batched<T>
where
    T: ItemSize,
{
    pub fn new(
        iter: impl Iterator<Item = T> + Send + 'static,
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
            inner: Box::new(iter),
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn build_batch(
        iter: &mut impl Iterator<Item = T>,
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

impl<T> Iterator for Batched<T>
where
    T: ItemSize,
{
    type Item = Batch<T>;

    fn next(&mut self) -> Option<Self::Item> {
        Self::build_batch(
            &mut self.inner,
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

pub trait TensorizedIterator<T>
where
    Self: Iterator<Item = T> + Send + 'static,
    T: Tensorize + Send + 'static,
{
    fn tensorized(self) -> Tensorized<T>;
}

impl<I, T> TensorizedIterator<T> for I
where
    I: Iterator<Item = T> + Send + 'static,
    T: Tensorize + Send + 'static,
{
    fn tensorized(self) -> Tensorized<T> {
        Tensorized::new(self)
    }
}

pub struct Tensorized<T>
where
    T: Tensorize + Send + 'static,
{
    inner: Box<dyn Iterator<Item = T> + Send + 'static>,
}

impl<T> Tensorized<T>
where
    T: Tensorize + Send + 'static,
{
    pub fn new(inner: impl Iterator<Item = T> + Send + 'static) -> Self {
        Self {
            inner: Box::new(inner),
        }
    }
}

impl<T> Iterator for Tensorized<T>
where
    T: Tensorize + Send + 'static,
{
    type Item = (T, <T as Tensorize>::Output);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|b| {
            let tensorized = b.tensorize();
            (b, tensorized)
        })
    }
}

pub trait BufferedIterator<T>
where
    Self: Iterator<Item = T> + Send + 'static,
    T: Send + 'static,
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

pub struct Buffered<T>
where
    T: Send + 'static,
{
    // inner: Box<dyn Iterator<Item = T> + Send + 'static>,
    inner: Receiver<T>,
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
        Self { inner: rx }
    }
}

impl<T> Iterator for Buffered<T>
where
    T: Send + 'static,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use crate::data::loading::{open, BatchLimitType, BatchedIterator};
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
    use std::collections::HashMap;
    use std::io::BufReader;
    use std::io::{self, BufRead};
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use super::{
        train_data_generator_from_files, BufferedIterator, PipelineIterator, TextIterator,
    };

    const MULTI30K_FIRST: &str = "Two young, White males are outside near many bushes.";
    const MULTI30K_SECOND: &str = "Several men in hard hats are operating a giant pulley system.";
    const MULTI30K_REV_FIRST: &str = "A man in shorts and a Hawaiian shirt leans over the rail of a pilot boat, with fog and mountains in the background.";
    const MULTI30K_REV_SECOND: &str =
        "An elderly man sits outside a storefront accompanied by a young boy with a cart.";

    #[test]
    fn test_text_iterator() -> anyhow::Result<()> {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let d = base.clone().join("resources/test/multi30k.txt");
        let d2 = base.clone().join("resources/test/multi30k_rev.txt");
        let multi30k = train_data_generator_from_files(&d, None)?;
        let mut it = TextIterator::new(
            vec![multi30k],
            super::TextIterationStrategy::Sequential,
            None,
        )?;

        // first check sequential lines with one file
        assert_eq!(it.min_len(), 29000);
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
        let multi30k = train_data_generator_from_files(&d, Some(&d2))?;
        let mut it = TextIterator::new(
            vec![multi30k],
            super::TextIterationStrategy::Sequential,
            None,
        )?;

        assert_eq!(it.min_len(), 29000);
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
        let multi30k = train_data_generator_from_files(&d, None)?;
        let multi30k_rev = train_data_generator_from_files(&d2, None)?;
        let mut it = TextIterator::new(
            vec![multi30k, multi30k_rev],
            super::TextIterationStrategy::Interleaved,
            None,
        )?;

        assert_eq!(it.min_len(), 2 * 29000);
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
        let multi30k = train_data_generator_from_files(&d, None)?;
        let multi30k_rev = train_data_generator_from_files(&d2, None)?;
        let it = TextIterator::new(
            vec![multi30k, multi30k_rev],
            super::TextIterationStrategy::Weighted,
            None,
        )?;
        assert_eq!(it.min_len(), 2 * 29000);
        Ok(())
    }

    #[test]
    fn test_pipeline_iterator() -> anyhow::Result<()> {
        let _ = env_logger::try_init();

        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let d = base.clone().join("resources/test/multi30k.txt");

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
        let multi30k = train_data_generator_from_files(&d, None)?;
        let text_iter = TextIterator::new(
            vec![multi30k],
            super::TextIterationStrategy::Sequential,
            None,
        )?;

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
        info!("took {:.2}s to fetch {} items", time, n);

        let n_cpus = num_cpus::get();

        // if more cpus are available, test with more workers, check that its faster
        if n_cpus >= 2 {
            let multi30k = train_data_generator_from_files(&d, None)?;
            let text_iter = TextIterator::new(
                vec![multi30k],
                super::TextIterationStrategy::Sequential,
                None,
            )?;
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
            info!("took {:.2}s to fetch {} items", time2, n);
            assert!(time2 < time);
            time = time2;
        }

        // test with even more workers, if available
        if n_cpus >= 4 {
            let multi30k = train_data_generator_from_files(&d, None)?;
            let text_iter = TextIterator::new(
                vec![multi30k],
                super::TextIterationStrategy::Sequential,
                None,
            )?;
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

        // test that all lines of multi30k.txt are returned in order,
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
        let multi30k = train_data_generator_from_files(&d, None)?;
        let text_iter = TextIterator::new(
            vec![multi30k],
            super::TextIterationStrategy::Sequential,
            None,
        )?;

        let it = text_iter
            .filter_map(|(d, _)| {
                if let Ok(d) = d {
                    Some((d, TextDataInfo::default()))
                } else {
                    None
                }
            })
            .pipe(pipeline.clone(), 0);
        let lines = BufReader::new(open(&d)?).lines();
        for (item, line) in it.zip(lines) {
            assert_eq!(item.unwrap().data.target, line.unwrap())
        }
        Ok(())
    }

    #[test]
    fn test_batched_pipeline_iterator() -> anyhow::Result<()> {
        let _ = env_logger::try_init();

        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let d = base.clone().join("resources/test/multi30k.txt");
        let multi30k = train_data_generator_from_files(&d, None)?;
        let text_iter = TextIterator::new(
            vec![multi30k],
            super::TextIterationStrategy::Sequential,
            None,
        )?;
        let lines: Vec<String> = BufReader::new(open(&d)?)
            .lines()
            .collect::<Result<Vec<String>, io::Error>>()
            .unwrap();
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
        for (batch, line_batch) in pipe_it.zip(&lines.iter().chunks(8)) {
            assert!(batch.len() > 0 && batch.len() <= 8);
            let lines: Vec<&String> = line_batch.into_iter().collect();
            assert!(batch.len() == lines.len());
            for (item, line) in batch.into_iter().zip(lines.into_iter()) {
                assert_eq!(item.data.target, *line);
            }
        }
        // now check the batched iterator with any combinations of shuffling and sorting
        // check that each line was yielded by the batch iterator once or twice
        // (because some descriptions in multi30k appear twice)
        for shuffle in [true, false] {
            for sort in [true, false] {
                let multi30k = train_data_generator_from_files(&d, None)?;
                let text_iter = TextIterator::new(
                    vec![multi30k],
                    super::TextIterationStrategy::Weighted,
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
                    lines.iter().cloned().map(|l| (l, 0)).collect();
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
                        && line_counter.iter().map(|(_, c)| *c).sum::<usize>() == lines.len()
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
