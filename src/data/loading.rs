use crate::data::{Batch, Pipeline, TextData};
use crate::tokenization::LANG_UNK;
use crate::utils::find_subsequences_of_max_size_k;
use pyo3::pyclass;
use rand::distributions::WeightedIndex;
use rand::prelude::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{sleep, Builder, JoinHandle};
use std::time::Duration;
use std::{panic, process};

pub trait TextGen: Send {
    fn org_iter(&self) -> Box<dyn Iterator<Item = String> + Send>;
    fn has_proc(&self) -> bool;
    fn proc_iter(&self) -> Option<Box<dyn Iterator<Item = String> + Send>>;
    fn lang_iter(&self) -> Box<dyn Iterator<Item = String> + Send>;
    fn min_len(&self) -> usize;
}

fn open(p: &Path) -> File {
    File::open(p).expect(&format!("could not open file at {:?}", p))
}

fn count_lines(p: &Path) -> usize {
    BufReader::new(open(p)).lines().count()
}

#[derive(Clone, Debug)]
pub struct TextFile {
    original: PathBuf,
    processed: Option<PathBuf>,
    language: String,
    len: usize,
}

impl TextFile {
    pub fn new(org: &PathBuf, proc: Option<&PathBuf>, lang: Option<String>) -> Self {
        let len = count_lines(&org);
        if proc.is_some() {
            let proc_len = count_lines(proc.unwrap());
            assert_eq!(
                len,
                proc_len,
                "expected same number of lines for {:?} and {:?}",
                org,
                proc.unwrap()
            );
        }
        TextFile {
            original: org.clone(),
            processed: proc.map(|p| p.clone()),
            language: lang.unwrap_or(LANG_UNK.to_string()),
            len,
        }
    }

    pub fn new_boxed(org: &PathBuf, proc: Option<&PathBuf>, lang: Option<String>) -> Box<Self> {
        Box::new(Self::new(org, proc, lang))
    }
}

impl TextGen for TextFile {
    fn org_iter(&self) -> Box<dyn Iterator<Item = String> + Send> {
        Box::new(
            BufReader::new(open(&self.original))
                .lines()
                .map(|l| l.expect("failed to read line")),
        )
    }

    fn has_proc(&self) -> bool {
        self.processed.is_some()
    }

    fn proc_iter(&self) -> Option<Box<dyn Iterator<Item = String> + Send>> {
        if let Some(path) = &self.processed {
            Some(Box::new(
                BufReader::new(open(path))
                    .lines()
                    .map(|l| l.expect("failed to read line")),
            ))
        } else {
            None
        }
    }

    fn lang_iter(&self) -> Box<dyn Iterator<Item = String> + Send> {
        let lang = self.language.clone();
        Box::new((0..).map(move |_| lang.clone()))
    }

    fn min_len(&self) -> usize {
        self.len
    }
}

#[derive(Clone, Debug)]
pub struct TextContainer {
    original: Vec<String>,
    processed: Option<Vec<String>>,
    lang: Option<Vec<String>>,
}

impl TextContainer {
    pub fn new(
        original: Vec<String>,
        processed: Option<Vec<String>>,
        lang: Option<Vec<String>>,
    ) -> Self {
        TextContainer {
            original,
            processed,
            lang,
        }
    }

    pub fn new_boxed(
        original: Vec<String>,
        processed: Option<Vec<String>>,
        lang: Option<Vec<String>>,
    ) -> Box<Self> {
        Box::new(Self::new(original, processed, lang))
    }
}

impl TextGen for TextContainer {
    fn org_iter(&self) -> Box<dyn Iterator<Item = String> + Send> {
        Box::new(self.original.clone().into_iter())
    }

    fn has_proc(&self) -> bool {
        self.processed.is_some()
    }

    fn proc_iter(&self) -> Option<Box<dyn Iterator<Item = String> + Send>> {
        if let Some(data) = &self.processed {
            Some(Box::new(data.clone().into_iter()))
        } else {
            None
        }
    }

    fn lang_iter(&self) -> Box<dyn Iterator<Item = String> + Send> {
        if self.lang.is_some() {
            Box::new(self.lang.clone().unwrap().into_iter())
        } else {
            Box::new((0..).map(|_| LANG_UNK.to_string()))
        }
    }

    fn min_len(&self) -> usize {
        self.original.len()
    }
}

pub struct TextGenerator {
    generators: Vec<Box<dyn TextGen>>,
}

impl TextGenerator {
    pub fn new(generators: Vec<Box<dyn TextGen>>) -> Self {
        TextGenerator { generators }
    }

    pub fn with_strategy(
        &self,
        strategy: TextIterationStrategy,
        seed: Option<u64>,
    ) -> TextIterator {
        match strategy {
            TextIterationStrategy::Sequential => self.sequential(),
            TextIterationStrategy::Interleaved => self.interleaved(),
            TextIterationStrategy::Weighted => self.weighted(seed),
        }
    }

    pub fn sequential(&self) -> TextIterator {
        TextIterator::new(
            &self.generators[..],
            TextIterationStrategy::Sequential,
            None,
        )
    }

    pub fn interleaved(&self) -> TextIterator {
        TextIterator::new(
            &self.generators[..],
            TextIterationStrategy::Interleaved,
            None,
        )
    }

    pub fn weighted(&self, seed: Option<u64>) -> TextIterator {
        TextIterator::new(&self.generators[..], TextIterationStrategy::Weighted, seed)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[pyclass]
pub enum TextIterationStrategy {
    Sequential,
    Interleaved,
    Weighted,
}

pub struct TextIterator {
    org_iters: Vec<Box<dyn Iterator<Item = String> + Send>>,
    proc_iters: Vec<Box<dyn Iterator<Item = String> + Send>>,
    lang_iters: Vec<Box<dyn Iterator<Item = String> + Send>>,
    lengths: Vec<usize>,
    strategy: TextIterationStrategy,
    idx: usize,
    rng: ChaCha8Rng,
    finished: Vec<bool>,
}

impl TextIterator {
    pub fn new(
        text_generators: &[Box<dyn TextGen>],
        strategy: TextIterationStrategy,
        seed: Option<u64>,
    ) -> Self {
        assert!(!text_generators.is_empty());
        let mut org_iters = vec![];
        let mut proc_iters = vec![];
        let mut lang_iters = vec![];
        let mut lengths = vec![];
        for text_reader in text_generators {
            org_iters.push(text_reader.org_iter());
            proc_iters.push(if text_reader.has_proc() {
                text_reader.proc_iter().unwrap()
            } else {
                text_reader.org_iter()
            });
            lengths.push(text_reader.min_len());
            lang_iters.push(text_reader.lang_iter())
        }
        if strategy == TextIterationStrategy::Weighted {
            assert!(
                lengths.iter().all(|l| *l > 0),
                "for the weighted iteration strategy all text generators must specify a positive \
            minimum length, otherwise they would never be iterated"
            );
        }
        let finished = vec![false; org_iters.len()];
        TextIterator {
            org_iters,
            proc_iters,
            lengths,
            lang_iters,
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
impl Iterator for TextIterator {
    type Item = TextData;

    fn next(&mut self) -> Option<Self::Item> {
        if self.all_finished() {
            return None;
        }
        let original;
        loop {
            let maybe_original = self.org_iters[self.idx].next();
            if maybe_original.is_none() {
                self.finished[self.idx] = true;
                if self.all_finished() {
                    return None;
                }
                self.next_idx();
                continue;
            }
            original = maybe_original.unwrap();
            break;
        }
        let processed = self.proc_iters[self.idx].next();
        let language = self.lang_iters[self.idx].next();
        let data = TextData::new(original, processed, language);
        self.next_idx();
        Some(data)
    }
}

pub(crate) trait PipelineIterator<T>
where
    Self: Iterator<Item = TextData> + Send + 'static + Sized,
    T: Clone + Send,
{
    fn pipe(
        self,
        pipeline: &Pipeline<T>,
        num_threads: u8,
        buffer_size: usize,
        seed: Option<u64>,
    ) -> Pipe<T>;
}
impl<I, T> PipelineIterator<T> for I
where
    I: Iterator<Item = TextData> + Send + 'static + Sized,
    T: Clone + Send,
{
    fn pipe(
        self,
        pipeline: &Pipeline<T>,
        num_threads: u8,
        buffer_size: usize,
        seed: Option<u64>,
    ) -> Pipe<T> {
        if num_threads == 0 {
            Pipe::new(self, pipeline.clone(), seed)
        } else {
            Pipe::new_threaded(self, pipeline.clone(), num_threads, buffer_size, seed)
        }
    }
}
pub struct Pipe<T>
where
    T: Clone + Send + 'static,
{
    inner: Box<dyn Iterator<Item = T> + Send>,
    size_hint: (usize, Option<usize>),
}

impl<T> Pipe<T>
where
    T: Clone + Send + 'static,
{
    fn new<I: Iterator<Item = TextData> + Send + 'static>(
        inner: I,
        pipeline: Pipeline<T>,
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

    fn new_threaded<I: Iterator<Item = TextData> + Send + 'static>(
        mut inner: I,
        pipeline: Pipeline<T>,
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
        self.size_hint()
    }
}

#[derive(Copy, Clone, PartialEq)]
#[pyclass]
pub enum BatchLimitType {
    BatchSize,
    TotalItemSize,
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
        shuffle_prefetch_factor: usize,
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
        shuffle_prefetch_factor: usize,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        seed: Option<u64>,
    ) -> Batched<Self, T> {
        let batch_limit = batch_limit.max(1);
        let shuffle_prefetch_factor = shuffle_prefetch_factor.max(2);
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
            shuffle_buffer: Vec::new(),
            shuffle_prefetch_factor,
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
    shuffle_buffer: Vec<T>,
    shuffle_prefetch_factor: usize,
    sort: bool,
    remainder: Option<T>,
}

impl<I, T> Batched<I, T>
where
    I: Iterator<Item = T> + Sized,
    T: ItemSize + Clone,
{
    pub fn next_shuffled(&mut self) -> (Option<Batch<T>>, Option<T>) {
        // we create a batch in the following way:
        // 1. fill up the shuffle buffer
        // 2. shuffle the shuffle buffer
        // 3. pop from shuffle buffer until we have a maxed out batch
        let mut buffer_limit =
            BatchLimit::from_iter(self.shuffle_buffer.iter(), &self.batch_limit_type);
        // fill buffer until we overshoot batch limit * some factor (>=1)
        while buffer_limit.limit() <= self.batch_limit * self.shuffle_prefetch_factor {
            let Some(item) = self.inner.next() else {
                break;
            };
            buffer_limit = buffer_limit.update(&item);
            self.shuffle_buffer.push(item);
        }
        // we are only done if the shuffle buffer is empty
        if self.shuffle_buffer.is_empty() {
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
        // sort is only considered inside next shuffled (if shuffle is true) because
        // sorting effectively also shuffles the order of the underlying item iterator
        if self.sort {
            // add remainder to shuffle buffer
            // shuffle buffer now contains at least two items
            self.shuffle_buffer
                .push(self.remainder.as_ref().unwrap().clone());
            self.remainder = None;
            // sort the buffer by length
            self.shuffle_buffer.sort_by(|a, b| a.size().cmp(&b.size()));
            // calculate possible subsequences for batch
            let sub_sequences = find_subsequences_of_max_size_k(
                &self.shuffle_buffer,
                self.batch_limit,
                |sub_items| BatchLimit::from_iter(sub_items.iter(), &self.batch_limit_type).limit(),
            );
            // randomly choose one subsequence
            if sub_sequences.is_empty() {
                // only happens if there is no single item that's smaller than the batch limit,
                // since we always need to return at least one item, just return the last
                // and set the second last to the remainder, this should always work
                // since the shuffle buffer contains at least two items
                let batch = Batch::new(vec![self.shuffle_buffer.pop().unwrap()]);
                return (Some(batch), self.shuffle_buffer.pop());
            }
            let index = self.rng.gen_range(0..sub_sequences.len());
            let (start_range, end_range) = sub_sequences[index];
            let shuffle_buff_lengths = self
                .shuffle_buffer
                .iter()
                .map(|i| i.size())
                .collect::<Vec<usize>>();
            let items_in_range: Vec<T> = self
                .shuffle_buffer
                .splice(start_range..end_range, vec![])
                .collect();
            let batch_limit = BatchLimit::from_iter(items_in_range.iter(), &self.batch_limit_type);
            (Some(Batch::new(items_in_range)), self.shuffle_buffer.pop())
        } else {
            // shuffle the buffer
            self.shuffle_buffer.shuffle(&mut self.rng);
            // now pop from the back until batch is full
            Self::batch_from(
                self.remainder.as_ref().unwrap().clone(),
                || self.shuffle_buffer.pop(),
                self.batch_limit,
                self.batch_limit_type,
            )
        }
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
        let mut batch_limit = BatchLimit::from_iter(items.iter(), &limit_type);
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
        let (batch, remainder) = if self.shuffle {
            self.next_shuffled()
        } else {
            Self::batch_from(
                self.remainder.as_ref().unwrap().clone(),
                || self.inner.next(),
                self.batch_limit,
                self.batch_limit_type,
            )
        };
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
    fn from_iter<'a>(
        items: impl Iterator<Item = &'a (impl ItemSize + 'a)>,
        limit_type: &BatchLimitType,
    ) -> Self {
        match limit_type {
            BatchLimitType::BatchSize => Self::BatchSize(items.count()),
            BatchLimitType::TotalItemSize => {
                let mut count = 0;
                let max_length = items
                    .map(|item| {
                        count += 1;
                        item.size()
                    })
                    .max()
                    .unwrap_or(0);
                Self::TotalItemSize(count, max_length)
            }
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
    use crate::data::loading::{open, BatchLimitType, PipelineIterator, TextFile, TextGenerator};
    use crate::data::{Item, Pipeline, PipelineWithTokenizerConfig, TextData};
    use crate::tokenization::{TokenizeConfig, TokenizerConfig};
    use itertools::Itertools;
    use log::info;
    use std::collections::HashMap;
    use std::io;
    use std::io::{BufRead, BufReader};
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    const MULTI30K_FIRST: &str = "Two young, White males are outside near many bushes.";
    const MULTI30K_SECOND: &str = "Several men in hard hats are operating a giant pulley system.";
    const MULTI30K_REV_FIRST: &str = "A man in shorts and a Hawaiian shirt leans over the rail of a pilot boat, with fog and mountains in the background.";
    const MULTI30K_REV_SECOND: &str =
        "An elderly man sits outside a storefront accompanied by a young boy with a cart.";
    const UNK: &str = "[unk]";

    #[test]
    fn test_text_iterator() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let d = base.clone().join("resources/test/multi30k.txt");
        let d2 = base.clone().join("resources/test/multi30k_rev.txt");
        let multi30k = TextFile::new_boxed(&d, None, Some("1".to_string()));
        let multi30k_rev = TextFile::new_boxed(&d2, None, Some("2".to_string()));
        let text_files = TextGenerator::new(vec![multi30k.clone()]);
        // first check sequential lines with one file
        let mut it = text_files.sequential();
        assert_eq!(it.min_len(), 29000);
        assert_eq!(
            it.next().unwrap(),
            TextData {
                original: MULTI30K_FIRST.to_string(),
                processed: MULTI30K_FIRST.to_string(),
                language: "1".to_string(),
            }
        );
        assert_eq!(
            it.next().unwrap(),
            TextData {
                original: MULTI30K_SECOND.to_string(),
                processed: MULTI30K_SECOND.to_string(),
                language: "1".to_string(),
            }
        );
        // check sequential lines with original and processed
        let multi30k_and_rev = TextFile::new_boxed(&d, Some(&d2), Some("1".to_string()));
        let text_files = TextGenerator::new(vec![multi30k_and_rev]);
        let mut it = text_files.sequential();
        assert_eq!(it.min_len(), 29000);
        assert_eq!(
            it.next().unwrap(),
            TextData {
                original: MULTI30K_FIRST.to_string(),
                processed: MULTI30K_REV_FIRST.to_string(),
                language: "1".to_string(),
            }
        );
        assert_eq!(
            it.next().unwrap(),
            TextData {
                original: MULTI30K_SECOND.to_string(),
                processed: MULTI30K_REV_SECOND.to_string(),
                language: "1".to_string(),
            }
        );
        // check interleaved lines with two files
        let text_files = TextGenerator::new(vec![multi30k.clone(), multi30k_rev.clone()]);
        let mut it = text_files.interleaved();
        assert_eq!(it.min_len(), 2 * 29000);
        assert_eq!(
            it.next().unwrap(),
            TextData {
                original: MULTI30K_FIRST.to_string(),
                processed: MULTI30K_FIRST.to_string(),
                language: "1".to_string(),
            }
        );
        assert_eq!(
            it.next().unwrap(),
            TextData {
                original: MULTI30K_REV_FIRST.to_string(),
                processed: MULTI30K_REV_FIRST.to_string(),
                language: "2".to_string(),
            }
        );
        assert_eq!(
            it.next().unwrap(),
            TextData {
                original: MULTI30K_SECOND.to_string(),
                processed: MULTI30K_SECOND.to_string(),
                language: "1".to_string(),
            }
        );
        assert_eq!(
            it.next().unwrap(),
            TextData {
                original: MULTI30K_REV_SECOND.to_string(),
                processed: MULTI30K_REV_SECOND.to_string(),
                language: "2".to_string(),
            }
        );
        // check that they are indeed interleaved
        let mut idx: usize = 4;
        while let Some(data) = it.next() {
            assert_eq!(&data.language, if idx % 2 == 0 { "1" } else { "2" });
            idx += 1;
        }
        // check weighted lines with two files
        let text_files = TextGenerator::new(vec![multi30k_rev, multi30k]);
        let mut it = text_files.weighted(Some(22));
        assert_eq!(it.min_len(), 2 * 29000);
        let mut first_count = 0;
        let mut second_count = 0;
        while let Some(data) = it.next() {
            if data.language.as_str() == "1" {
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
        let multi30k = TextFile::new_boxed(&d, None, Some("1".to_string()));
        let text_files = TextGenerator::new(vec![multi30k]);
        // create a pipeline that simulates some real processing,
        // we use the dummy tokenizer with a delay of 100 milliseconds for that
        let pipeline = Pipeline::from_config(PipelineWithTokenizerConfig {
            preprocessing: vec![],
            labeling: None,
            tokenizer: TokenizerConfig::new(
                TokenizeConfig::Dummy(Duration::from_millis(200)),
                vec![],
                vec![],
                None,
            ),
        });
        // test if it works with one worker and record the time it took
        let mut it = text_files.sequential().pipe(&pipeline, 0, 4, None);
        let lines = BufReader::new(open(&d)).lines();
        let now = Instant::now();
        let n: usize = 20;
        let _: Vec<Item> = it.take(n).collect();
        let time = now.elapsed().as_secs_f64();
        info!("took {:.2}s to fetch {} items", time, n);
        // test with more workers, check that its faster
        let mut it = text_files.sequential().pipe(&pipeline, 2, 4, None);
        let lines = BufReader::new(open(&d)).lines();
        let now = Instant::now();
        let _: Vec<Item> = it.take(n).collect();
        let time2 = now.elapsed().as_secs_f64();
        info!("took {:.2}s to fetch {} items", time2, n);
        assert!(time2 < time);
        // test with even more workers
        let mut it = text_files.sequential().pipe(&pipeline, 4, 4, None);
        let lines = BufReader::new(open(&d)).lines();
        let now = Instant::now();
        let _: Vec<Item> = it.take(n).collect();
        let time3 = now.elapsed().as_secs_f64();
        info!("took {:.2}s to fetch {} items", time3, n);
        assert!(time3 < time2);
        // test that all lines of multi30k.txt are returned in order,
        // switch to non blocking tokenizer again
        let pipeline = Pipeline::from_config(PipelineWithTokenizerConfig {
            preprocessing: vec![],
            labeling: None,
            tokenizer: TokenizerConfig::new(
                TokenizeConfig::Dummy(Duration::from_millis(0)),
                vec![],
                vec![],
                None,
            ),
        });
        let mut it = text_files.sequential().pipe(&pipeline, 0, 4, None);
        let lines = BufReader::new(open(&d)).lines();
        for (idx, (line, item)) in it.zip(lines).enumerate() {
            assert_eq!(line.data.original, item.unwrap())
        }
    }

    #[test]
    fn test_batched_pipeline_iterator() {
        let _ = env_logger::try_init();

        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let d = base.clone().join("resources/test/multi30k.txt");
        let multi30k = TextFile::new_boxed(&d, None, Some("1".to_string()));
        let text_files = TextGenerator::new(vec![multi30k]);
        let lines: Vec<String> = BufReader::new(open(&d))
            .lines()
            .collect::<Result<Vec<String>, io::Error>>()
            .unwrap();
        let mut pipeline = Pipeline::from_config(PipelineWithTokenizerConfig {
            preprocessing: vec![],
            labeling: None,
            tokenizer: TokenizerConfig::new(TokenizeConfig::Byte(true), vec![], vec![], None),
        });
        // first check the batched iterator with shuffling disabled
        let mut pipe_it = text_files
            .sequential()
            .pipe(&pipeline, 4, 16, None)
            .batched(16, BatchLimitType::BatchSize, false, 0, false, None);
        for (batch, line_batch) in pipe_it.zip(&lines.iter().chunks(16)) {
            assert!(batch.len() > 0 && batch.len() <= 16);
            for (item, line) in batch.into_iter().zip(line_batch.into_iter()) {
                assert_eq!(item.data.original, *line);
            }
        }
        // now check the batched iterator with shuffling and sorting
        let mut pipe_it = text_files
            .weighted(Some(22))
            .pipe(&pipeline, 4, 4, None)
            .batched(256, BatchLimitType::TotalItemSize, true, 32, true, Some(22));
        // check that each line was yielded by the batch iterator once or twice
        // (because some descriptions in multi30k appear twice)
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
                let mut count = line_counter.get_mut(&item.data.original).unwrap();
                *count += 1;
            }
        }
        assert!(
            line_counter.iter().all(|(_, c)| *c == 1 || *c == 2)
                && line_counter.iter().map(|(_, c)| *c).sum::<usize>() == lines.len()
        );
    }
}
