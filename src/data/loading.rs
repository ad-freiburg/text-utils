use std::fs::File;
use std::io::{BufRead, BufReader, Lines, Read};
use std::{panic, process};
use std::path::Path;
use std::sync::{Arc, mpsc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver};
use std::thread::{Builder, JoinHandle, sleep};
use std::time::{Duration};
use rand::prelude::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use crate::data::{TextData, Item, Pipeline, Batch};

struct TextIterator<R: Read> {
    org_lines: Lines<BufReader<R>>,
    proc_lines: Option<Lines<BufReader<R>>>,
    counter: usize,
    num_lines: usize,
    language: String,
}

fn open(p: &Path) -> File {
    File::open(p)
        .expect(&format!("could not open file at {:?}", p))
}

impl TextIterator<File> {
    pub fn new(
        original_path: &Path,
        processed_path: Option<&Path>,
        language: Option<&str>,
    ) -> Self {
        let org_num_lines = BufReader::new(open(original_path)).lines().count();
        let org_lines = BufReader::new(open(original_path)).lines();
        let proc_lines = if processed_path.is_some() {
            let proc_num_lines = BufReader::new(
                open(processed_path.unwrap())
            ).lines().count();
            assert_eq!(
                org_num_lines,
                proc_num_lines,
                "original and processed text data do not have the same number of lines: {} != {}",
                org_num_lines,
                proc_num_lines
            );
            Some(BufReader::new(open(processed_path.unwrap())).lines())
        } else {
            None
        };
        TextIterator {
            org_lines,
            proc_lines,
            counter: 0,
            num_lines: org_num_lines,
            language: language.unwrap_or("[unk]").to_string(),
        }
    }
}

impl Iterator for TextIterator<File> {
    type Item = TextData;

    fn next(&mut self) -> Option<Self::Item> {
        // no more lines to process
        let Some(original) = self.org_lines.next() else {
            return None;
        };
        let original = original
            .expect(&format!("could not read original line {}", self.counter));
        let processed = if self.proc_lines.is_some() {
            self.proc_lines.as_mut().unwrap().next()
                .expect("should not happen")
                .expect(&format!("could not read processed line {}", self.counter))
        } else {
            original.clone()
        };
        self.counter += 1;
        Some(TextData { original, processed, language: self.language.clone() })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.num_lines))
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

pub struct BatchedPipelineIterator {
    batch_limit: usize,
    batch_limit_type: BatchLimitType,
    rng: ChaCha8Rng,
    iter: PipelineIterator,
    shuffle: bool,
    shuffle_buffer: Vec<Item>,
    shuffle_prefetch_factor: usize,
    remainder: Option<Item>,
}

impl BatchedPipelineIterator {
    pub fn new(
        mut iter: PipelineIterator,
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

impl Iterator for BatchedPipelineIterator {
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
    use std::path::PathBuf;
    use log::info;
    use crate::data::loading::{PipelineIterator, TextIterator};
    use crate::data::{Pipeline, PipelineConfig, TextData};
    use crate::tokenization::TokenizerConfig;

    #[test]
    fn test_text_iterator() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let d = base.clone().join("resources/test/multi30k.txt");
        let mut it = TextIterator::new(&d, None, None);
        assert_eq!(it.size_hint(), (0, Some(29000)));
        let first = "Two young, White males are outside near many bushes.".to_string();
        let second = "Several men in hard hats are operating a giant pulley system.".to_string();
        let unk = "[unk]".to_string();
        assert_eq!(it.next().unwrap(), TextData {
            original: first.clone(),
            processed: first.clone(),
            language: unk.clone(),
        });
        assert_eq!(it.next().unwrap(), TextData {
            original: second.clone(),
            processed: second.clone(),
            language: unk.clone(),
        });
        let d2 = base.clone().join("resources/test/multi30k_rev.txt");
        let mut it = TextIterator::new(&d, Some(&d2), None);
        assert_eq!(it.size_hint(), (0, Some(29000)));
        let last = "A man in shorts and a Hawaiian shirt leans over the rail of a pilot boat, with fog and mountains in the background.".to_string();
        let second_last = "An elderly man sits outside a storefront accompanied by a young boy with a cart.".to_string();
        assert_eq!(it.next().unwrap(), TextData {
            original: first,
            processed: last,
            language: unk.clone(),
        });
        assert_eq!(it.next().unwrap(), TextData {
            original: second,
            processed: second_last,
            language: unk.clone(),
        });
    }

    #[test]
    fn test_pipeline_iterator() {
        env_logger::init();

        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let d = base.clone().join("resources/test/multi30k.txt");
        let it = TextIterator::new(&d, None, None);
        let pipeline = Pipeline::new(PipelineConfig {
            preprocessing: vec![],
            labeling: None,
            tokenizer: TokenizerConfig::Byte(true, vec![], vec![]),
        });
        let mut it = PipelineIterator::new(it, pipeline, 2, 2);
        for (idx, item) in it.take(2).enumerate() {

        }
    }
}
