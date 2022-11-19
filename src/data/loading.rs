use std::fs::File;
use std::io::{BufRead, BufReader, Lines, Read};
use std::{panic, process};
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver};
use std::thread::{Builder, JoinHandle, sleep};
use std::time::{Duration};
use rand::distributions::WeightedIndex;
use rand::prelude::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use crate::data::{TextData, Item, Pipeline, Batch};

#[derive(Clone, Debug)]
pub struct TextFileDataset {
    files: Vec<(PathBuf, Option<PathBuf>, Option<String>)>,
}

impl TextFileDataset {
    pub fn new(files: &[(PathBuf, Option<PathBuf>, Option<String>)]) -> Self {
        TextFileDataset {
            files: files.iter().cloned().collect()
        }
    }
}

impl TextFileDataset {
    pub fn sequential_lines(&self) -> TextIterator<File> {
        TextIterator::from_files(
            &self.files,
            TextIterationStrategy::Sequential,
            None,
        )
    }

    pub fn interleaved_lines(&self) -> TextIterator<File> {
        TextIterator::from_files(
            &self.files,
            TextIterationStrategy::Interleaved,
            None,
        )
    }

    pub fn weighted_lines(&self, seed: Option<u64>) -> TextIterator<File> {
        TextIterator::from_files(
            &self.files,
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

pub struct TextIterator<R: Read> {
    org_lines: Vec<Lines<BufReader<R>>>,
    proc_lines: Vec<Lines<BufReader<R>>>,
    languages: Vec<String>,
    num_lines: Vec<usize>,
    strategy: TextIterationStrategy,
    idx: usize,
    rng: ChaCha8Rng,
    finished: Vec<bool>,
}

fn open(p: &Path) -> File {
    File::open(p)
        .expect(&format!("could not open file at {:?}", p))
}

impl<R: Read> TextIterator<R> {
    fn new<T, OpenFn: Fn(&T) -> R>(
        readers: &[(T, Option<T>, Option<String>)],
        open_fn: OpenFn,
        strategy: TextIterationStrategy,
        seed: Option<u64>,
    ) -> Self {
        assert!(!readers.is_empty());
        let mut org_lines = vec![];
        let mut proc_lines = vec![];
        let mut languages = vec![];
        let mut num_lines = vec![];
        for (org, proc, language) in readers {
            let org_num_lines = BufReader::new(open_fn(org))
                .lines()
                .count();
            let org_l = BufReader::new(open_fn(org))
                .lines();
            let proc_l = if proc.is_some() {
                let proc_num_lines = BufReader::new(
                    open_fn(proc.as_ref().unwrap())
                ).lines().count();
                assert_eq!(
                    org_num_lines,
                    proc_num_lines,
                    "original and processed text data do not have the same number of lines: {} != {}",
                    org_num_lines,
                    proc_num_lines
                );
                BufReader::new(open_fn(proc.as_ref().unwrap())).lines()
            } else {
                BufReader::new(open_fn(org)).lines()
            };
            org_lines.push(org_l);
            proc_lines.push(proc_l);
            num_lines.push(org_num_lines);
            languages.push(language.as_ref().cloned().unwrap_or("[unk]".to_string()));
        }
        let finished = vec![false; org_lines.len()];
        TextIterator {
            org_lines,
            proc_lines,
            num_lines,
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
                        .map(|&idx| self.num_lines[idx])
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

impl TextIterator<File> {
    pub fn from_files(
        files: &[(PathBuf, Option<PathBuf>, Option<String>)],
        strategy: TextIterationStrategy,
        seed: Option<u64>,
    ) -> Self {
        Self::new(
            files,
            |p| open(p),
            strategy,
            seed
        )
    }
}

impl<R: Read> Iterator for TextIterator<R> {
    type Item = TextData;

    fn next(&mut self) -> Option<Self::Item> {
        if self.all_finished() {
            return None;
        }
        let original;
        loop {
            let maybe_original = self.org_lines[self.idx].next();
            if maybe_original.is_none() {
                self.finished[self.idx] = true;
                if self.all_finished() { return None; }
                self.next_idx();
                continue;
            }
            original = maybe_original
                .unwrap()
                .expect("could not read line");
            break;
        }
        let processed = self.proc_lines[self.idx]
            .next()
            .expect("expected original and processed text to have the same number of lines")
            .expect("could not read line");
        Some(TextData {
            original,
            processed,
            language: self.languages[self.idx].clone(),
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.num_lines.iter().sum()))
    }
}

pub struct PipelineIterator {
    rx: Receiver<Item>,
    size_hint: (usize, Option<usize>),
}

impl PipelineIterator {
    pub fn new<T: Iterator<Item=TextData> + Send + Sync + 'static>(
        iter: T,
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
            rx,
            size_hint,
        }
    }
}

impl Iterator for PipelineIterator {
    type Item = Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.rx.recv().ok()
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

pub struct BatchedPipelineIterator<T: Iterator<Item=Item>> {
    batch_limit: usize,
    batch_limit_type: BatchLimitType,
    rng: ChaCha8Rng,
    iter: T,
    shuffle: bool,
    shuffle_buffer: Vec<Item>,
    shuffle_prefetch_factor: usize,
    remainder: Option<Item>,
}

impl<T: Iterator<Item=Item>> BatchedPipelineIterator<T> {
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
        BatchedPipelineIterator {
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
                .map(|item| item.tokenization.0.len())
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
                BatchLimitType::NumTokens => buffer_limit += item.tokenization.0.len()
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
                BatchLimitType::NumTokens => batch_limit += item.tokenization.0.len()
            }
            items.push(item);
        }
        let remainder = items.pop();
        (Some(Batch { items }), remainder)
    }
}

impl<T: Iterator<Item=Item>> Iterator for BatchedPipelineIterator<T> {
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
    use std::io::{BufRead, BufReader};
    use std::path::PathBuf;
    use std::thread::sleep;
    use std::time::{Duration, Instant};
    use itertools::Itertools;
    use log::info;
    use crate::data::loading::{BatchedPipelineIterator, BatchLimitType, open, PipelineIterator, TextIterator};
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
        let mut it = TextIterator::new(&d, None, None);
        assert_eq!(it.size_hint(), (0, Some(29000)));
        assert_eq!(it.next().unwrap(), TextData {
            original: MULTI30K_FIRST.to_string(),
            processed: MULTI30K_FIRST.to_string(),
            language: UNK.to_string(),
        });
        assert_eq!(it.next().unwrap(), TextData {
            original: MULTI30K_SECOND.to_string(),
            processed: MULTI30K_SECOND.to_string(),
            language: UNK.to_string(),
        });
        let d2 = base.clone().join("resources/test/multi30k_rev.txt");
        let mut it = TextIterator::new(&d, Some(&d2), None);
        assert_eq!(it.size_hint(), (0, Some(29000)));
        assert_eq!(it.next().unwrap(), TextData {
            original: MULTI30K_FIRST.to_string(),
            processed: MULTI30K_REV_FIRST.to_string(),
            language: UNK.to_string(),
        });
        assert_eq!(it.next().unwrap(), TextData {
            original: MULTI30K_SECOND.to_string(),
            processed: MULTI30K_REV_SECOND.to_string(),
            language: UNK.to_string(),
        });
    }

    #[test]
    fn test_pipeline_iterator() {
        let _ = env_logger::try_init();

        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let d = base.clone().join("resources/test/multi30k.txt");
        // create a pipeline that simulates some real processing,
        // we use the dummy tokenizer with a delay of 100 milliseconds for that
        let mut pipeline = Pipeline::from_config(PipelineConfig {
            preprocessing: vec![],
            labeling: None,
            tokenizer: TokenizerConfig::Dummy(Duration::from_millis(100)),
        });
        // test if it works with one worker and record the time it took
        let mut it = PipelineIterator::new(
            TextIterator::new(&d, None, None),
            pipeline.clone(),
            1,
            2,
        );
        let now = Instant::now();
        let n: usize = 20;
        let _: Vec<Item> = it.take(n).collect();
        let time = now.elapsed().as_secs_f64();
        info!("took {:.2}s to fetch {} items", time, n);
        // test with more workers, check that its faster
        let mut it = PipelineIterator::new(
            TextIterator::new(&d, None, None),
            pipeline.clone(),
            2,
            2,
        );
        let now = Instant::now();
        let _: Vec<Item> = it.take(n).collect();
        let time2 = now.elapsed().as_secs_f64();
        info!("took {:.2}s to fetch {} items", time2, n);
        assert!(time2 < time);
        // test with even more workers
        let mut it = PipelineIterator::new(
            TextIterator::new(&d, None, None),
            pipeline.clone(),
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
        let mut pipeline = Pipeline::from_config(PipelineConfig {
            preprocessing: vec![],
            labeling: None,
            tokenizer: TokenizerConfig::Dummy(Duration::from_micros(0)),
        });
        let mut it = PipelineIterator::new(
            TextIterator::new(&d, None, None),
            pipeline,
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
        env_logger::init();

        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let d = base.clone().join("resources/test/multi30k.txt");
        let mut pipeline = Pipeline::from_config(PipelineConfig {
            preprocessing: vec![],
            labeling: None,
            tokenizer: TokenizerConfig::Dummy(Duration::from_millis(0)),
        });
        let pipe_it = PipelineIterator::new(
            TextIterator::new(&d, None, None),
            pipeline.clone(),
            1,
            2,
        );
        // first check the batched iterator with shuffling disabled
        let it = BatchedPipelineIterator::new(
            pipe_it,
            16,
            BatchLimitType::BatchSize,
            false,
            0,
            0,
        );
        let d = base.clone().join("resources/test/multi30k.txt");
        let lines = BufReader::new(open(&d)).lines();
        for (batch, line_batch) in it
            .zip(&lines.chunks(16)) {
            assert!(batch.len() <= 16);
            for (item, line) in batch
                .into_iter()
                .zip(line_batch.into_iter()) {
                assert_eq!(item.data.original, line.unwrap());
            }
        }
        // first check the batched iterator with shuffling disabled
        let it = BatchedPipelineIterator::new(
            pipe_it,
            16,
            BatchLimitType::BatchSize,
            false,
            0,
            0,
        );
        let d = base.clone().join("resources/test/multi30k.txt");
        let lines = BufReader::new(open(&d)).lines();
        for (batch, line_batch) in it
            .zip(&lines.chunks(16)) {
            assert!(batch.len() <= 16);
            for (item, line) in batch
                .into_iter()
                .zip(line_batch.into_iter()) {
                assert_eq!(item.data.original, line.unwrap());
            }
        }
    }
}
