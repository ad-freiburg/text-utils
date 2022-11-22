use std::fs::File;
use std::io::{BufRead, BufReader};
use std::{panic, process};
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::{Builder, JoinHandle, sleep};
use std::time::{Duration};
use rand::distributions::WeightedIndex;
use rand::prelude::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use crate::data::{TextData, Item, Pipeline, Batch};

pub type BoxedStringIterator = Box<dyn Iterator<Item=String> + Send + 'static>;

pub trait TextGen {
    fn org_iter(&self) -> BoxedStringIterator;
    fn has_proc(&self) -> bool;
    fn proc_iter(&self) -> Option<BoxedStringIterator>;
    fn has_lang(&self) -> bool;
    fn lang(&self) -> Option<String>;
    fn len(&self) -> usize;
}


fn open(p: &Path) -> File {
    File::open(p)
        .expect(&format!("could not open file at {:?}", p))
}

fn count_lines(p: &Path) -> usize {
    BufReader::new(open(p)).lines().count()
}

#[derive(Clone, Debug)]
pub struct TextFile {
    original: PathBuf,
    processed: Option<PathBuf>,
    language: Option<String>,
    len: usize,
}

impl TextFile {
    pub fn new(org: &PathBuf, proc: Option<&PathBuf>, lang: Option<&str>) -> Self {
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
            language: lang.map(|l| l.to_string()),
            len,
        }
    }

    pub fn new_boxed(org: &PathBuf, proc: Option<&PathBuf>, lang: Option<&str>) -> Box<Self> {
        Box::new(Self::new(org, proc, lang))
    }
}

impl TextGen for TextFile {
    fn org_iter(&self) -> BoxedStringIterator {
        Box::new(BufReader::new(open(&self.original))
            .lines()
            .map(|l| {
                l.expect("failed to read line")
            }))
    }

    fn has_proc(&self) -> bool {
        self.processed.is_some()
    }

    fn proc_iter(&self) -> Option<BoxedStringIterator> {
        if let Some(path) = &self.processed {
            Some(Box::new(BufReader::new(open(path))
                .lines()
                .map(|l| {
                    l.expect("failed to read line")
                })))
        } else {
            None
        }
    }

    fn has_lang(&self) -> bool {
        self.language.is_some()
    }

    fn lang(&self) -> Option<String> {
        self.language.clone()
    }

    fn len(&self) -> usize {
        self.len
    }
}

#[derive(Clone, Debug)]
pub struct TextContainer<'a> {
    original: &'a [String],
    processed: Option<&'a [String]>,
    lang: Option<String>,
}

impl<'a> TextContainer<'a> {
    pub fn new(original: &'a [String], processed: Option<&'a [String]>, lang: Option<String>) -> Self {
        TextContainer {
            original,
            processed,
            lang,
        }
    }
}

impl TextGen for TextContainer<'_> {
    fn org_iter(&self) -> BoxedStringIterator {
        Box::new(Vec::from(self.original).into_iter())
    }

    fn has_proc(&self) -> bool {
        self.processed.is_some()
    }

    fn proc_iter(&self) -> Option<BoxedStringIterator> {
        if let Some(data) = self.processed {
            Some(Box::new(Vec::from(data).into_iter()))
        } else {
            None
        }
    }

    fn has_lang(&self) -> bool {
        self.lang.is_some()
    }

    fn lang(&self) -> Option<String> {
        self.lang.clone()
    }

    fn len(&self) -> usize {
        self.original.len()
    }
}

pub struct TextGenerator {
    generators: Vec<Box<dyn TextGen>>,
}

impl TextGenerator {
    pub fn new(generators: Vec<Box<dyn TextGen>>) -> Self {
        TextGenerator {
            generators
        }
    }

    pub fn sequential(&self) -> TextIterator {
        TextIterator::new(
            &self.generators,
            TextIterationStrategy::Sequential,
            None,
        )
    }

    pub fn interleaved(&self) -> TextIterator {
        TextIterator::new(
            &self.generators,
            TextIterationStrategy::Interleaved,
            None,
        )
    }

    pub fn weighted(&self, seed: Option<u64>) -> TextIterator {
        TextIterator::new(
            &self.generators,
            TextIterationStrategy::Weighted,
            seed,
        )
    }
}

pub enum TextIterationStrategy {
    Sequential,
    Interleaved,
    Weighted,
}

pub struct TextIterator {
    org_iters: Vec<BoxedStringIterator>,
    proc_iters: Vec<BoxedStringIterator>,
    languages: Vec<String>,
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
        let mut languages = vec![];
        let mut lengths = vec![];
        for text_reader in text_generators {
            org_iters.push(text_reader.org_iter());
            proc_iters.push(if text_reader.has_proc() {
                text_reader.proc_iter().unwrap()
            } else {
                text_reader.org_iter()
            });
            lengths.push(text_reader.len());
            languages.push(text_reader.lang().unwrap_or("[unk]".to_string()));
        }
        let finished = vec![false; org_iters.len()];
        TextIterator {
            org_iters,
            proc_iters,
            lengths,
            languages,
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
                let non_finished_indices: Vec<usize> = self.finished
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &f)| if f { Some(idx) } else { None })
                    .collect();
                let dist = WeightedIndex::new(
                    non_finished_indices
                        .iter()
                        .map(|&idx| self.lengths[idx])
                        .collect::<Vec<usize>>()
                )
                    .expect("could not create line distribution");
                self.idx = non_finished_indices[self.rng.sample(dist)];
            }
        };
    }

    fn all_finished(&self) -> bool {
        self.finished.iter().all(|&f| f)
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
                if self.all_finished() { return None; }
                self.next_idx();
                continue;
            }
            original = maybe_original.unwrap();
            break;
        }
        let processed = self.proc_iters[self.idx]
            .next()
            .unwrap();
        let data = TextData {
            original,
            processed,
            language: self.languages[self.idx].clone(),
        };
        self.next_idx();
        Some(data)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.lengths.iter().sum()))
    }
}

pub struct PipelineIterator {
    iter: Box<dyn Iterator<Item=Item>>,
    size_hint: (usize, Option<usize>),
}

impl PipelineIterator {
    pub fn new(
        iter: impl Iterator<Item=TextData> + 'static,
        pipeline: Pipeline,
    ) -> Self {
        let size_hint = iter.size_hint();
        PipelineIterator {
            iter: Box::new(iter
                .enumerate()
                .map(move |(idx, item)| {
                    pipeline.apply(item, Some(idx as u64))
                })),
            size_hint,
        }
    }

    pub fn new_threaded(
        iter: impl Iterator<Item=TextData> + Send + 'static,
        pipeline: Pipeline,
        worker_threads: u8,
        buffer_size: usize,
    ) -> Self {
        let size_hint = iter.size_hint();
        let iter = Arc::new(Mutex::new(iter.enumerate()));
        let buffer_size = buffer_size.max(1);
        let num_threads = worker_threads
            .min(num_cpus::get() as u8)
            .max(1) as usize;
        let orig_hook = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            orig_hook(info);
            process::exit(1);
        }));
        let (tx, rx) = mpsc::sync_channel(buffer_size);
        let sent_counter = Arc::new(AtomicU64::new(0));
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
                        let item = pipe_clone.apply(data, Some(idx));
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

        PipelineIterator {
            iter: Box::new(rx.into_iter()),
            size_hint,
        }
    }

    pub fn batched(
        self,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        shuffle: bool,
        shuffle_prefetch_factor: usize,
        seed: u64,
    ) -> BatchedIterator<PipelineIterator> {
        BatchedIterator::new(
            self,
            batch_limit,
            batch_limit_type,
            shuffle,
            shuffle_prefetch_factor,
            seed,
        )
    }
}

impl Iterator for PipelineIterator {
    type Item = Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.size_hint
    }
}

#[derive(Copy, Clone)]
pub enum BatchLimitType {
    BatchSize,
    NumTokens,
}

pub struct BatchedIterator<T: Iterator<Item=Item>> {
    batch_limit: usize,
    batch_limit_type: BatchLimitType,
    rng: ChaCha8Rng,
    iter: T,
    shuffle: bool,
    shuffle_buffer: Vec<Item>,
    shuffle_prefetch_factor: usize,
    remainder: Option<Item>,
}

impl<T: Iterator<Item=Item>> BatchedIterator<T> {
    pub fn new(
        mut iter: T,
        batch_limit: usize,
        batch_limit_type: BatchLimitType,
        shuffle: bool,
        shuffle_prefetch_factor: usize,
        seed: u64,
    ) -> Self {
        let shuffle_prefetch_factor = shuffle_prefetch_factor.max(2);
        // initialize remainder
        let remainder = iter.next();
        BatchedIterator {
            batch_limit,
            batch_limit_type,
            rng: ChaCha8Rng::seed_from_u64(seed),
            iter,
            shuffle,
            shuffle_buffer: Vec::new(),
            shuffle_prefetch_factor,
            remainder,
        }
    }

    pub fn next_shuffled(&mut self) -> (Option<Batch>, Option<Item>) {
        // we create a batch in the following way:
        // 1. fill up the shuffle buffer
        // 2. shuffle the shuffle buffer
        // 3. pop from shuffle buffer until we have a maxed out batch
        let mut buffer_limit = match self.batch_limit_type {
            BatchLimitType::BatchSize => self.shuffle_buffer.len(),
            BatchLimitType::NumTokens => self.shuffle_buffer
                .iter()
                .map(|item| item.tokenization.token_ids.len())
                .sum()
        };
        // fill buffer until batch limit * some factor is hit
        while buffer_limit < self.batch_limit * self.shuffle_prefetch_factor {
            let Some(item) = self.iter.next() else {
                // we are only done if the shuffle buffer is empty
                if self.shuffle_buffer.is_empty() {
                    return (None, None);
                }
                break;
            };
            match self.batch_limit_type {
                BatchLimitType::BatchSize => buffer_limit += 1,
                BatchLimitType::NumTokens => buffer_limit += item.tokenization.token_ids.len()
            }
            self.shuffle_buffer.push(item);
        }
        // shuffle the buffer
        self.shuffle_buffer.shuffle(&mut self.rng);
        // now pop from the back until batch is full
        Self::batch_from(
            self.remainder.as_ref().unwrap().clone(),
            &mut || self.shuffle_buffer.pop(),
            self.batch_limit,
            self.batch_limit_type,
        )
    }

    fn batch_from(
        initial: Item,
        f: &mut dyn FnMut() -> Option<Item>,
        limit: usize,
        limit_type: BatchLimitType,
    ) -> (Option<Batch>, Option<Item>) {
        let mut batch_limit: usize = 0;
        let mut items = vec![initial];
        while batch_limit < limit {
            let Some(item) = f() else {
                return if items.is_empty() {
                    (None, None)
                } else {
                    (Some(Batch { items }), None)
                };
            };
            match limit_type {
                BatchLimitType::BatchSize => batch_limit += 1,
                BatchLimitType::NumTokens => batch_limit += item.tokenization.token_ids.len()
            }
            items.push(item);
        }
        let remainder = items.pop();
        (Some(Batch { items }), remainder)
    }
}

impl<T: Iterator<Item=Item>> Iterator for BatchedIterator<T> {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remainder.is_none() {
            return None;
        }
        let (batch, remainder) = if self.shuffle {
            self.next_shuffled()
        } else {
            Self::batch_from(
                self.remainder.as_ref().unwrap().clone(),
                &mut || self.iter.next(),
                self.batch_limit,
                self.batch_limit_type,
            )
        };
        self.remainder = remainder;
        batch
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io;
    use std::io::{BufRead, BufReader};
    use std::path::PathBuf;
    use std::thread::sleep;
    use std::time::{Duration, Instant};
    use itertools::Itertools;
    use log::info;
    use crate::data::loading::{BatchedIterator, BatchLimitType, open, PipelineIterator, TextFile, TextGenerator, TextIterator};
    use crate::data::{Item, Pipeline, PipelineConfig, TextData};
    use crate::tokenization::{ByteTokenizer, Tokenization, TokenizationInfo, Tokenize, TokenizerConfig};

    const MULTI30K_FIRST: &str = "Two young, White males are outside near many bushes.";
    const MULTI30K_SECOND: &str = "Several men in hard hats are operating a giant pulley system.";
    const MULTI30K_REV_FIRST: &str = "A man in shorts and a Hawaiian shirt leans over the rail of a pilot boat, with fog and mountains in the background.";
    const MULTI30K_REV_SECOND: &str = "An elderly man sits outside a storefront accompanied by a young boy with a cart.";
    const UNK: &str = "[unk]";

    #[test]
    fn test_text_iterator() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let d = base.clone().join("resources/test/multi30k.txt");
        let d2 = base.clone().join("resources/test/multi30k_rev.txt");
        let multi30k = TextFile::new_boxed(&d, None, Some("1"));
        let multi30k_rev = TextFile::new_boxed(&d2, None, Some("2"));
        let text_files = TextGenerator::new(vec![multi30k.clone()]);
        // first check sequential lines with one file
        let mut it = text_files.sequential();
        assert_eq!(it.size_hint(), (0, Some(29000)));
        assert_eq!(it.next().unwrap(), TextData {
            original: MULTI30K_FIRST.to_string(),
            processed: MULTI30K_FIRST.to_string(),
            language: "1".to_string(),
        });
        assert_eq!(it.next().unwrap(), TextData {
            original: MULTI30K_SECOND.to_string(),
            processed: MULTI30K_SECOND.to_string(),
            language: "1".to_string(),
        });
        // check sequential lines with original and processed
        let multi30k_and_rev = TextFile::new_boxed(&d, Some(&d2), Some("1"));
        let text_files = TextGenerator::new(vec![multi30k_and_rev]);
        let mut it = text_files.sequential();
        assert_eq!(it.size_hint(), (0, Some(29000)));
        assert_eq!(it.next().unwrap(), TextData {
            original: MULTI30K_FIRST.to_string(),
            processed: MULTI30K_REV_FIRST.to_string(),
            language: "1".to_string(),
        });
        assert_eq!(it.next().unwrap(), TextData {
            original: MULTI30K_SECOND.to_string(),
            processed: MULTI30K_REV_SECOND.to_string(),
            language: "1".to_string(),
        });
        // check interleaved lines with two files
        let text_files = TextGenerator::new(vec![multi30k, multi30k_rev]);
        let mut it = text_files.interleaved();
        assert_eq!(it.size_hint(), (0, Some(2 * 29000)));
        assert_eq!(it.next().unwrap(), TextData {
            original: MULTI30K_FIRST.to_string(),
            processed: MULTI30K_FIRST.to_string(),
            language: "1".to_string(),
        });
        assert_eq!(it.next().unwrap(), TextData {
            original: MULTI30K_REV_FIRST.to_string(),
            processed: MULTI30K_REV_FIRST.to_string(),
            language: "2".to_string(),
        });
        assert_eq!(it.next().unwrap(), TextData {
            original: MULTI30K_SECOND.to_string(),
            processed: MULTI30K_SECOND.to_string(),
            language: "1".to_string(),
        });
        assert_eq!(it.next().unwrap(), TextData {
            original: MULTI30K_REV_SECOND.to_string(),
            processed: MULTI30K_REV_SECOND.to_string(),
            language: "2".to_string(),
        });
        // check that they are indeed interleaved
        let mut idx: usize = 4;
        while let Some(data) = it.next() {
            assert_eq!(&data.language, if idx % 2 == 0 { "1" } else { "2" });
            idx += 1;
        }
    }

    #[test]
    fn test_pipeline_iterator() {
        let _ = env_logger::try_init();

        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let d = base.clone().join("resources/test/multi30k.txt");
        let multi30k = TextFile::new_boxed(&d, None, Some("1"));
        let text_files = TextGenerator::new(vec![multi30k]);
        // create a pipeline that simulates some real processing,
        // we use the dummy tokenizer with a delay of 100 milliseconds for that
        let pipeline = Pipeline::from_config(PipelineConfig {
            preprocessing: vec![],
            labeling: None,
            tokenizer: TokenizerConfig::Dummy(Duration::from_millis(100)),
        });
        // test if it works with one worker and record the time it took
        let mut it = pipeline
            .clone()
            .apply_iter_threaded(
                text_files.sequential(),
                1,
                1,
            );
        let now = Instant::now();
        let n: usize = 20;
        let _: Vec<Item> = it.take(n).collect();
        let time = now.elapsed().as_secs_f64();
        info!("took {:.2}s to fetch {} items", time, n);
        // test with more workers, check that its faster
        let mut it = pipeline
            .clone()
            .apply_iter_threaded(
                text_files.sequential(),
                2,
                2,
            );
        let now = Instant::now();
        let _: Vec<Item> = it.take(n).collect();
        let time2 = now.elapsed().as_secs_f64();
        info!("took {:.2}s to fetch {} items", time2, n);
        assert!(time2 < time);
        // test with even more workers
        let mut it = pipeline.apply_iter_threaded(
            text_files.sequential(),
            4,
            4,
        );
        let now = Instant::now();
        let _: Vec<Item> = it.take(n).collect();
        let time3 = now.elapsed().as_secs_f64();
        info!("took {:.2}s to fetch {} items", time3, n);
        assert!(time3 < time2);
        // test that all lines of multi30k.txt are returned in order,
        // switch to non blocking tokenizer again
        let pipeline = Pipeline::from_config(PipelineConfig {
            preprocessing: vec![],
            labeling: None,
            tokenizer: TokenizerConfig::Dummy(Duration::from_micros(0)),
        });
        let mut it = pipeline
            .clone()
            .apply_iter_threaded(
                text_files.sequential(),
                4,
                4,
            );
        let lines = BufReader::new(open(&d))
            .lines();
        for (idx, (line, item)) in it.zip(lines).enumerate() {
            assert_eq!(line.data.original, item.unwrap())
        }
    }

    #[test]
    fn test_batched_pipeline_iterator() {
        let _ = env_logger::try_init();

        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let d = base.clone().join("resources/test/multi30k.txt");
        let multi30k = TextFile::new_boxed(&d, None, Some("1"));
        let text_files = TextGenerator::new(vec![multi30k]);
        let lines: Vec<String> = BufReader::new(open(&d))
            .lines()
            .collect::<Result<Vec<String>, io::Error>>()
            .unwrap();
        let mut pipeline = Pipeline::from_config(PipelineConfig {
            preprocessing: vec![],
            labeling: None,
            tokenizer: TokenizerConfig::Dummy(Duration::from_millis(0)),
        });
        // first check the batched iterator with shuffling disabled
        let mut pipe_it = pipeline
            .clone()
            .apply_iter_threaded(
                text_files.sequential(),
                4,
                4,
            )
            .batched(
                16,
                BatchLimitType::BatchSize,
                false,
                0,
                0,
            );
        for (batch, line_batch) in pipe_it
            .zip(&lines.iter().chunks(16)) {
            assert!(batch.len() > 0 && batch.len() <= 16);
            for (item, line) in batch
                .into_iter()
                .zip(line_batch.into_iter()) {
                assert_eq!(item.data.original, *line);
            }
        }
        // now check the batched iterator with shuffling
        let mut pipe_it = pipeline
            .clone()
            .apply_iter_threaded(
                text_files.sequential(),
                4,
                4,
            )
            .batched(
                16,
                BatchLimitType::BatchSize,
                true,
                4,
                0,
            );
        // check that each line was yielded by the batch iterator once or twice
        // (because some descriptions in multi30k appear twice)
        let mut line_counter: HashMap<String, usize> = HashMap::from_iter(
            lines
                .iter()
                .cloned()
                .map(|l| (l, 0))
        );
        for (batch, line_batch) in pipe_it
            .zip(&lines.iter().chunks(16)) {
            assert!(batch.len() > 0 && batch.len() <= 16);
            for item in batch {
                let mut count = line_counter.get_mut(&item.data.original).unwrap();
                *count += 1;
            }
        }
        assert!(line_counter.iter().all(|(_, c)| *c == 1 || *c == 2)
            && line_counter.iter().map(|(_, c)| *c).sum::<usize>() == lines.len());
    }
}
