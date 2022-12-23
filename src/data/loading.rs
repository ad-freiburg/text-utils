use crate::data::{Batch, InferenceData, Pipeline, TextData};
use crate::utils::{find_subsequences_of_max_size_k, py_invalid_type_error};
use pyo3::{intern, prelude::*, PyClass};
use rand::distributions::WeightedIndex;
use rand::prelude::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::marker::PhantomData;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{sleep, Builder, JoinHandle};
use std::time::Duration;
use std::{panic, process};

use super::InferenceDataFileFormat;

pub trait DataGen: Iterator + Send {
    fn min_len(&self) -> usize;
}

fn open(p: &Path) -> File {
    File::open(p).expect(&format!("could not open file at {:?}", p))
}

fn count_lines(p: &Path) -> usize {
    BufReader::new(open(p)).lines().count()
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
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buf = vec![];
        match self.reader.read_until(b'\n', &mut buf) {
            Ok(0) => return None,
            Ok(_) => {
                // remove \n or \r\n from line
                buf.pop();
                if !buf.is_empty() && *buf.last().unwrap() == b'\r' {
                    buf.pop();
                }
                let s = String::from_utf8_lossy(&buf);
                Some(s.to_string())
            }
            Err(e) => panic!("failed to read line: {e}"),
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

pub fn text_data_generator_from_files(
    org: &Path,
    proc: Option<&Path>,
    lang: Option<String>,
) -> impl DataGen<Item = TextData> {
    let org_len = count_lines(org);
    let org_iter = LossyUtf8Reader::new(BufReader::new(open(org))).lines();
    let mut proc_iter = if proc.is_some() {
        let proc_len = count_lines(proc.unwrap());
        assert_eq!(
            org_len,
            proc_len,
            "expected same number of lines for {:?} and {:?}",
            org,
            proc.unwrap()
        );
        Some(LossyUtf8Reader::new(BufReader::new(open(proc.unwrap()))).lines())
    } else {
        None
    };
    let iter = org_iter.map(move |org_s| {
        let proc_s = if proc_iter.is_some() {
            proc_iter.as_mut().unwrap().next()
        } else {
            None
        };
        TextData::new(org_s, proc_s, lang.clone())
    });
    DataGenerator {
        min_len: org_len,
        iter,
    }
}

pub fn inference_data_generator_from_file(
    path: &Path,
    format: InferenceDataFileFormat,
    lang: Option<String>,
) -> impl DataGen<Item = InferenceData> {
    let iter = LossyUtf8Reader::new(BufReader::new(open(path)))
        .lines()
        .map(move |s| {
            let mut data = InferenceData::from_str(&s, &format);
            if data.language.is_none() && lang.is_some() {
                data.language = lang.clone();
            };
            data
        });
    DataGenerator { min_len: 0, iter }
}

pub fn text_data_generator_from_sequences(
    original: Vec<String>,
    processed: Option<Vec<String>>,
    language: Option<Vec<String>>,
) -> impl DataGen<Item = TextData> {
    let len = original.len();
    let org_iter = original.into_iter();
    let mut proc_iter = if processed.is_some() {
        assert_eq!(processed.as_ref().unwrap().len(), len);
        Some(processed.unwrap().into_iter())
    } else {
        None
    };
    let mut lang_iter = if language.is_some() {
        assert_eq!(language.as_ref().unwrap().len(), len);
        Some(language.unwrap().into_iter())
    } else {
        None
    };
    let iter = org_iter.map(move |org_s| {
        let proc_s = if proc_iter.is_some() {
            proc_iter.as_mut().unwrap().next()
        } else {
            None
        };
        let lang_s = if lang_iter.is_some() {
            lang_iter.as_mut().unwrap().next()
        } else {
            None
        };
        TextData::new(org_s, proc_s, lang_s)
    });
    DataGenerator { iter, min_len: len }
}

pub struct PyIterator<T>
where
    T: PyClass + Clone,
{
    obj: PyObject,
    _phantom: Option<PhantomData<T>>,
}

impl<T> Iterator for PyIterator<T>
where
    T: PyClass + Clone,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        Python::with_gil(|py| {
            let return_value = self.obj.call_method0(py, intern!(py, "__next__"));
            // we return None when the python object is not an iterator
            // (no __next__ method), the items the python iterator produces
            // are not of type T items, and if the iterator is fully consumed.
            match return_value.ok() {
                Some(data) => data.extract(py).ok(),
                None => None,
            }
        })
    }
}

pub fn inference_data_generator_from_python(obj: PyObject) -> impl DataGen<Item = InferenceData> {
    let iter = PyIterator {
        obj,
        _phantom: None,
    };
    DataGenerator { iter, min_len: 0 }
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
    ) -> Self {
        assert!(!text_generators.is_empty());
        let lengths: Vec<usize> = text_generators.iter().map(|g| g.min_len()).collect();
        if strategy == TextIterationStrategy::Weighted {
            assert!(
                lengths.iter().all(|l| *l > 0),
                "for the weighted iteration strategy all text generators must specify a positive \
            minimum length, otherwise they would never be iterated"
            );
        }
        let finished = vec![false; text_generators.len()];
        TextIterator {
            text_generators,
            lengths,
            strategy,
            idx: 0,
            rng: if seed.is_some() {
                ChaCha8Rng::seed_from_u64(seed.unwrap())
            } else {
                ChaCha8Rng::from_entropy()
            },
            finished,
        }
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
    type Item = T;

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
        self.next_idx();
        Some(data)
    }
}

pub trait PipelineIterator<I, O>
where
    Self: Iterator<Item = I> + Send + 'static + Sized,
    I: Send + 'static,
    O: Send + 'static,
{
    fn pipe(
        self,
        pipeline: &Pipeline<I, O>,
        num_threads: u8,
        buffer_size: usize,
        seed: Option<u64>,
    ) -> Pipe<O>;
}

impl<It, I, O> PipelineIterator<I, O> for It
where
    It: Iterator<Item = I> + Send + 'static + Sized,
    I: Send + 'static,
    O: Send + 'static,
{
    fn pipe(
        self,
        pipeline: &Pipeline<I, O>,
        num_threads: u8,
        buffer_size: usize,
        seed: Option<u64>,
    ) -> Pipe<O> {
        if num_threads == 0 {
            Pipe::new(self, pipeline.clone(), seed)
        } else {
            Pipe::new_threaded(self, pipeline.clone(), num_threads, buffer_size, seed)
        }
    }
}

pub struct Pipe<T>
where
    T: Send + 'static,
{
    inner: Box<dyn Iterator<Item = T> + Send>,
    size_hint: (usize, Option<usize>),
}

impl<T> Pipe<T>
where
    T: Send + 'static,
{
    fn new<I: Send + 'static, It: Iterator<Item = I> + Send + 'static>(
        inner: It,
        pipeline: Pipeline<I, T>,
        seed: Option<u64>,
    ) -> Self {
        Pipe {
            size_hint: inner.size_hint(),
            inner: Box::new(inner.enumerate().map(move |(idx, data)| {
                pipeline.apply(
                    data,
                    idx,
                    if seed.is_none() {
                        None
                    } else {
                        Some(seed.unwrap() + idx as u64)
                    },
                )
            })),
        }
    }

    fn new_threaded<I: Send + 'static, It: Iterator<Item = I> + Send + 'static>(
        inner: It,
        pipeline: Pipeline<I, T>,
        num_threads: u8,
        buffer_size: usize,
        seed: Option<u64>,
    ) -> Self {
        let size_hint = inner.size_hint();
        let iter = Arc::new(Mutex::new(inner.enumerate()));
        let buffer_size = buffer_size.max(1);
        let num_threads = num_threads.min(num_cpus::get() as u8).max(1) as usize;
        let (tx, rx) = mpsc::sync_channel(buffer_size);
        let sent_counter = Arc::new(AtomicU64::new(0));
        panic::set_hook(Box::new(move |info| {
            println!("pipeline worker thread panicked: {info}");
            process::exit(1);
        }));
        for idx in 0..num_threads {
            let iter_arc = iter.clone();
            let tx_clone = tx.clone();
            let pipe_clone = pipeline.clone();
            let send_next = sent_counter.clone();
            let _: JoinHandle<()> = Builder::new()
                .name(format!("pipeline iterator thread {}", idx))
                .spawn(move || {
                    loop {
                        let data;
                        let idx;
                        // open new scope for proper mutex unlocking
                        {
                            let mut iter_lock = iter_arc.lock().unwrap();
                            let next = iter_lock.next();
                            if next.is_some() {
                                let next = next.unwrap();
                                idx = next.0 as u64;
                                data = next.1;
                            } else {
                                // we received none from the underlying iterator,
                                // stop this sender
                                return;
                            }
                        }
                        let item = pipe_clone.apply(
                            data,
                            idx as usize,
                            if seed.is_none() {
                                None
                            } else {
                                Some(seed.unwrap() + idx)
                            },
                        );
                        // wait until we are the next to send out item
                        while send_next.load(Ordering::SeqCst) != idx {
                            sleep(Duration::from_micros(100));
                        }
                        let send_result = tx_clone.send(item);
                        send_next.swap(idx as u64 + 1, Ordering::SeqCst);
                        if send_result.is_err() {
                            // receiver is closed, so we can return this thread
                            return;
                        }
                    }
                })
                .expect(&format!("failed building thread {}", idx));
        }
        Pipe {
            inner: Box::new(rx.into_iter()),
            size_hint,
        }
    }
}

impl<T> Iterator for Pipe<T>
where
    T: Clone + Send,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.size_hint
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
    Self: Iterator<Item = T> + Sized,
    T: ItemSize + Clone,
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
    I: Iterator<Item = T> + Sized,
    T: ItemSize + Clone,
{
    fn batched(
        mut self,
        sort: bool,
        shuffle: bool,
        prefetch_factor: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        seed: Option<u64>,
    ) -> Batched<Self, T> {
        let batch_limit = batch_limit.max(1);
        let prefetch_factor = prefetch_factor.max(2);
        // initialize remainder
        let remainder = self.next();
        Batched {
            batch_limit,
            batch_limit_type,
            rng: if seed.is_some() {
                ChaCha8Rng::seed_from_u64(seed.unwrap())
            } else {
                ChaCha8Rng::from_entropy()
            },
            inner: self,
            shuffle,
            buffer: Vec::new(),
            prefetch_factor,
            sort,
            remainder,
        }
    }
}

pub struct Batched<I, T>
where
    I: Iterator<Item = T> + Sized,
    T: ItemSize + Clone,
{
    batch_limit: usize,
    batch_limit_type: BatchLimitType,
    rng: ChaCha8Rng,
    inner: I,
    shuffle: bool,
    buffer: Vec<T>,
    prefetch_factor: usize,
    sort: bool,
    remainder: Option<T>,
}

impl<I, T> Batched<I, T>
where
    I: Iterator<Item = T> + Sized,
    T: ItemSize + Clone,
{
    pub fn build_batch(&mut self) -> (Option<Batch<T>>, Option<T>) {
        if !self.sort && !self.shuffle {
            return Self::batch_from(
                self.remainder.as_ref().unwrap().clone(),
                || self.inner.next(),
                self.batch_limit,
                self.batch_limit_type,
            );
        }
        // we create a batch in the following way:
        // 1. fill up the buffer
        // 2. maybe sort or shuffle the buffer
        // 3. pop from buffer until we have a maxed out batch
        let mut buffer_limit = BatchLimit::from_items(&self.buffer[..], &self.batch_limit_type);
        // fill buffer until we overshoot batch limit * some factor (>=1)
        while buffer_limit.limit() <= self.batch_limit * self.prefetch_factor {
            let Some(item) = self.inner.next() else {
                break;
            };
            buffer_limit = buffer_limit.update(&item);
            self.buffer.push(item);
        }
        // we are only done if the buffer is empty
        if self.buffer.is_empty() {
            // return remainder as final batch if we still have one
            return (
                if self.remainder.is_none() {
                    None
                } else {
                    Some(Batch::new(vec![self.remainder.as_ref().unwrap().clone()]))
                },
                None,
            );
        }
        if self.sort {
            // add remainder to buffer
            // buffer now contains at least two items
            self.buffer.push(self.remainder.as_ref().unwrap().clone());
            // sort the buffer by length
            self.buffer.sort_by(|a, b| a.size().cmp(&b.size()));
            // if shuffle is also true, return a random subsequence from the sorted buffer
            if self.shuffle {
                // calculate possible subsequences for batch
                let sub_sequences =
                    find_subsequences_of_max_size_k(&self.buffer, self.batch_limit, |sub_items| {
                        BatchLimit::from_items(&sub_items, &self.batch_limit_type).limit()
                    });
                // randomly choose one subsequence
                if sub_sequences.is_empty() {
                    // only happens if there is no single item that's smaller than the batch limit,
                    // since we always need to return at least one item, just return the last
                    // and set the second last to the remainder, this should always work
                    // since the shuffle buffer contains at least two items
                    let batch = Batch::new(vec![self.buffer.pop().unwrap()]);
                    return (Some(batch), self.buffer.pop());
                }
                let index = self.rng.gen_range(0..sub_sequences.len());
                let (start_range, end_range) = sub_sequences[index];
                let items_in_range: Vec<T> =
                    self.buffer.splice(start_range..end_range, vec![]).collect();
                return (Some(Batch::new(items_in_range)), self.buffer.pop());
            } else {
                // if shuffle is false, pop the largest element from the buffer as the remainder
                self.remainder = self.buffer.pop();
            }
        } else if self.shuffle {
            // shuffle the buffer
            self.buffer.shuffle(&mut self.rng);
        }
        // now pop from the back until batch is full
        Self::batch_from(
            self.remainder.as_ref().unwrap().clone(),
            || self.buffer.pop(),
            self.batch_limit,
            self.batch_limit_type,
        )
    }

    #[inline]
    fn batch_from(
        initial: T,
        mut f: impl FnMut() -> Option<T>,
        limit: usize,
        limit_type: BatchLimitType,
    ) -> (Option<Batch<T>>, Option<T>) {
        // the implementation makes sure that always at least 1 item is returned
        // in the batch, even if the item solely exceeds the specified limit
        let mut items = vec![initial];
        let mut batch_limit = BatchLimit::from_items(&items, &limit_type);
        let remainder = loop {
            let Some(item) = f() else {
                return (Some(Batch::new(items)), None);
            };
            batch_limit = batch_limit.update(&item);
            if batch_limit.limit() > limit {
                // if adding the item would overshoot
                // just return it as remainder
                break Some(item);
            } else {
                // if adding the item would not overshoot, just add it
                // and increase the batch limit counter
                items.push(item);
            }
        };
        (Some(Batch::new(items)), remainder)
    }
}

impl<I, T> Iterator for Batched<I, T>
where
    I: Iterator<Item = T>,
    T: ItemSize + Clone,
{
    type Item = Batch<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remainder.is_none() {
            return None;
        }
        let (batch, remainder) = self.build_batch();
        self.remainder = remainder;
        batch
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

#[cfg(test)]
mod tests {
    use crate::data::loading::{open, BatchLimitType, BatchedIterator, PipelineIterator};
    use crate::data::preprocessing::LabelingConfig;
    use crate::data::{Item, Pipeline, PreprocessingPipelineConfig, TextData};
    use crate::tokenization::{TokenizeConfig, TokenizerConfig};
    use itertools::Itertools;
    use log::info;
    use std::collections::HashMap;
    use std::io;
    use std::io::{BufRead, BufReader};
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use super::{text_data_generator_from_files, TextIterator};

    const MULTI30K_FIRST: &str = "Two young, White males are outside near many bushes.";
    const MULTI30K_SECOND: &str = "Several men in hard hats are operating a giant pulley system.";
    const MULTI30K_REV_FIRST: &str = "A man in shorts and a Hawaiian shirt leans over the rail of a pilot boat, with fog and mountains in the background.";
    const MULTI30K_REV_SECOND: &str =
        "An elderly man sits outside a storefront accompanied by a young boy with a cart.";

    #[test]
    fn test_text_iterator() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let d = base.clone().join("resources/test/multi30k.txt");
        let d2 = base.clone().join("resources/test/multi30k_rev.txt");
        let multi30k = Box::new(text_data_generator_from_files(
            &d,
            None,
            Some("1".to_string()),
        ));
        let mut it = TextIterator::new(
            vec![multi30k],
            super::TextIterationStrategy::Sequential,
            None,
        );

        // first check sequential lines with one file
        assert_eq!(it.min_len(), 29000);
        assert_eq!(
            it.next().unwrap(),
            TextData {
                original: MULTI30K_FIRST.to_string(),
                processed: MULTI30K_FIRST.to_string(),
                language: Some("1".to_string()),
            }
        );
        assert_eq!(
            it.next().unwrap(),
            TextData {
                original: MULTI30K_SECOND.to_string(),
                processed: MULTI30K_SECOND.to_string(),
                language: Some("1".to_string()),
            }
        );
        // check sequential lines with original and processed
        let multi30k = Box::new(text_data_generator_from_files(
            &d,
            Some(&d2),
            Some("1".to_string()),
        ));
        let mut it = TextIterator::new(
            vec![multi30k],
            super::TextIterationStrategy::Sequential,
            None,
        );

        assert_eq!(it.min_len(), 29000);
        assert_eq!(
            it.next().unwrap(),
            TextData {
                original: MULTI30K_FIRST.to_string(),
                processed: MULTI30K_REV_FIRST.to_string(),
                language: Some("1".to_string()),
            }
        );
        assert_eq!(
            it.next().unwrap(),
            TextData {
                original: MULTI30K_SECOND.to_string(),
                processed: MULTI30K_REV_SECOND.to_string(),
                language: Some("1".to_string()),
            }
        );
        // check interleaved lines with two files
        let multi30k = Box::new(text_data_generator_from_files(
            &d,
            None,
            Some("1".to_string()),
        ));
        let multi30k_rev = Box::new(text_data_generator_from_files(
            &d2,
            None,
            Some("2".to_string()),
        ));
        let mut it = TextIterator::new(
            vec![multi30k, multi30k_rev],
            super::TextIterationStrategy::Interleaved,
            None,
        );

        assert_eq!(it.min_len(), 2 * 29000);
        assert_eq!(
            it.next().unwrap(),
            TextData {
                original: MULTI30K_FIRST.to_string(),
                processed: MULTI30K_FIRST.to_string(),
                language: Some("1".to_string()),
            }
        );
        assert_eq!(
            it.next().unwrap(),
            TextData {
                original: MULTI30K_REV_FIRST.to_string(),
                processed: MULTI30K_REV_FIRST.to_string(),
                language: Some("2".to_string()),
            }
        );
        assert_eq!(
            it.next().unwrap(),
            TextData {
                original: MULTI30K_SECOND.to_string(),
                processed: MULTI30K_SECOND.to_string(),
                language: Some("1".to_string()),
            }
        );
        assert_eq!(
            it.next().unwrap(),
            TextData {
                original: MULTI30K_REV_SECOND.to_string(),
                processed: MULTI30K_REV_SECOND.to_string(),
                language: Some("2".to_string()),
            }
        );
        // check that they are indeed interleaved
        let mut idx: usize = 4;
        while let Some(data) = it.next() {
            assert_eq!(
                &data.language.unwrap(),
                if idx % 2 == 0 { "1" } else { "2" }
            );
            idx += 1;
        }
        // check weighted lines with two files
        let multi30k = Box::new(text_data_generator_from_files(
            &d,
            None,
            Some("1".to_string()),
        ));
        let multi30k_rev = Box::new(text_data_generator_from_files(
            &d2,
            None,
            Some("2".to_string()),
        ));
        let mut it = TextIterator::new(
            vec![multi30k, multi30k_rev],
            super::TextIterationStrategy::Weighted,
            None,
        );

        assert_eq!(it.min_len(), 2 * 29000);
        let mut first_count = 0;
        let mut second_count = 0;
        while let Some(data) = it.next() {
            if data.language.unwrap().as_str() == "1" {
                first_count += 1;
            } else {
                second_count += 1;
            }
        }
        assert_eq!(first_count, 29000);
        assert_eq!(first_count, second_count);
    }

    #[test]
    fn test_pipeline_iterator() {
        let _ = env_logger::try_init();

        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let d = base.clone().join("resources/test/multi30k.txt");

        // create a pipeline that simulates some real processing,
        // we use the dummy tokenizer with a delay of 100 milliseconds for that
        let pipeline = Pipeline::with_tokenizer(
            PreprocessingPipelineConfig::new(
                vec![],
                LabelingConfig::LabelWhitespaceCorrection(true),
            ),
            TokenizerConfig::new(
                TokenizeConfig::Dummy(Duration::from_millis(200)),
                vec![],
                vec![],
                None,
            ),
        );
        // test if it works with one worker and record the time it took
        let multi30k = Box::new(text_data_generator_from_files(
            &d,
            None,
            Some("1".to_string()),
        ));
        let text_iter = TextIterator::new(
            vec![multi30k],
            super::TextIterationStrategy::Sequential,
            None,
        );

        let it = text_iter.pipe(&pipeline, 0, 4, None);
        let now = Instant::now();
        let n: usize = 20;
        let _: Vec<Item> = it.take(n).collect();
        let time = now.elapsed().as_secs_f64();
        info!("took {:.2}s to fetch {} items", time, n);
        // test with more workers, check that its faster
        let multi30k = Box::new(text_data_generator_from_files(
            &d,
            None,
            Some("1".to_string()),
        ));
        let text_iter = TextIterator::new(
            vec![multi30k],
            super::TextIterationStrategy::Sequential,
            None,
        );

        let it = text_iter.pipe(&pipeline, 2, 4, None);
        let now = Instant::now();
        let _: Vec<Item> = it.take(n).collect();
        let time2 = now.elapsed().as_secs_f64();
        info!("took {:.2}s to fetch {} items", time2, n);
        assert!(time2 < time);
        // test with even more workers
        let multi30k = Box::new(text_data_generator_from_files(
            &d,
            None,
            Some("1".to_string()),
        ));
        let text_iter = TextIterator::new(
            vec![multi30k],
            super::TextIterationStrategy::Sequential,
            None,
        );

        let it = text_iter.pipe(&pipeline, 4, 4, None);
        let now = Instant::now();
        let _: Vec<Item> = it.take(n).collect();
        let time3 = now.elapsed().as_secs_f64();
        info!("took {:.2}s to fetch {} items", time3, n);
        assert!(time3 < time2);
        // test that all lines of multi30k.txt are returned in order,
        // switch to non blocking tokenizer again
        let pipeline = Pipeline::with_tokenizer(
            PreprocessingPipelineConfig::new(
                vec![],
                LabelingConfig::LabelWhitespaceCorrection(true),
            ),
            TokenizerConfig::new(
                TokenizeConfig::Dummy(Duration::from_millis(0)),
                vec![],
                vec![],
                None,
            ),
        );

        let multi30k = Box::new(text_data_generator_from_files(
            &d,
            None,
            Some("1".to_string()),
        ));
        let text_iter = TextIterator::new(
            vec![multi30k],
            super::TextIterationStrategy::Sequential,
            None,
        );

        let it = text_iter.pipe(&pipeline, 0, 4, None);
        let lines = BufReader::new(open(&d)).lines();
        for (line, item) in it.zip(lines) {
            assert_eq!(line.data.original, item.unwrap())
        }
    }

    #[test]
    fn test_batched_pipeline_iterator() {
        let _ = env_logger::try_init();

        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let d = base.clone().join("resources/test/multi30k.txt");
        let multi30k = Box::new(text_data_generator_from_files(
            &d,
            None,
            Some("1".to_string()),
        ));
        let text_iter = TextIterator::new(
            vec![multi30k],
            super::TextIterationStrategy::Sequential,
            None,
        );
        let lines: Vec<String> = BufReader::new(open(&d))
            .lines()
            .collect::<Result<Vec<String>, io::Error>>()
            .unwrap();
        let pipeline = Pipeline::with_tokenizer(
            PreprocessingPipelineConfig::new(
                vec![],
                LabelingConfig::LabelWhitespaceCorrection(true),
            ),
            TokenizerConfig::new(
                TokenizeConfig::Dummy(Duration::from_millis(0)),
                vec![],
                vec![],
                None,
            ),
        );
        let pipe_it = text_iter.pipe(&pipeline, 4, 16, None).batched(
            false,
            false,
            0,
            8,
            BatchLimitType::BatchSize,
            None,
        );
        for (batch, line_batch) in pipe_it.zip(&lines.iter().chunks(8)) {
            assert!(batch.len() > 0 && batch.len() <= 8);
            let lines: Vec<&String> = line_batch.into_iter().collect();
            assert!(batch.len() == lines.len());
            for (item, line) in batch.into_iter().zip(lines.into_iter()) {
                assert_eq!(item.data.original, *line);
            }
        }
        // now check the batched iterator with any combinations of shuffling and sorting
        // check that each line was yielded by the batch iterator once or twice
        // (because some descriptions in multi30k appear twice)
        for shuffle in [true, false] {
            for sort in [true, false] {
                let multi30k = Box::new(text_data_generator_from_files(
                    &d,
                    None,
                    Some("1".to_string()),
                ));
                let text_iter =
                    TextIterator::new(vec![multi30k], super::TextIterationStrategy::Weighted, None);

                let pipe_it = text_iter.pipe(&pipeline, 4, 4, None).batched(
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
                    let item_lengths: Vec<usize> = batch
                        .items
                        .iter()
                        .map(|item| item.tokenization.token_ids.len())
                        .collect();
                    let batch_size: usize = item_lengths.iter().sum();
                    assert!(batch.len() > 0);
                    assert!(batch_size <= 256,);
                    for item in batch {
                        let count = line_counter.get_mut(&item.data.original).unwrap();
                        *count += 1;
                    }
                }
                assert!(
                    line_counter.iter().all(|(_, c)| *c == 1 || *c == 2)
                        && line_counter.iter().map(|(_, c)| *c).sum::<usize>() == lines.len()
                );
            }
        }
    }
}
