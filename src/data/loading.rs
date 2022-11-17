use std::fs::File;
use std::io::{BufRead, BufReader, Lines, Read};
use std::{panic, process};
use std::path::Path;
use std::sync::{Arc, mpsc, Mutex};
use std::sync::mpsc::{Receiver};
use std::thread::{spawn};
use crate::data::{TextData, Item, Pipeline, Batch};

pub struct TextIterator<R: Read> {
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
    buffer: Receiver<Item>,
    size_hint: (usize, Option<usize>)
}

impl PipelineIterator {
    pub fn new<I: Iterator<Item=TextData> + Send + Sync + 'static>(
        iter: I,
        pipeline: Pipeline,
        n_threads: u8,
        buffer_size: usize,
    ) -> Self {
        let size_hint = iter.size_hint();
        let iter = Arc::new(Mutex::new(iter));
        let pipe = Arc::new(Mutex::new(pipeline));
        let num_threads = n_threads.min(num_cpus::get() as u8).max(1) as usize;
        let buffer_size = buffer_size.max(1);
        let (tx, rx) = mpsc::sync_channel(buffer_size);
        let orig_hook = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            orig_hook(info);
            process::exit(1);
        }));
        for _ in 0..num_threads {
            let iter_arc = iter.clone();
            let tx_clone = tx.clone();
            let pipe_clone = pipe.clone();
            let _ = spawn(move || {
                loop {
                    let v = iter_arc.lock().unwrap().next();
                    let Some(data) = v else {
                        // we received none from the underlying iterator,
                        // stop this sender
                        return;
                    };
                    let item = pipe_clone.lock().unwrap().apply(data);
                    if let Err(e) = tx_clone.send(item) {
                        panic!("send error in thread: {}", e)
                    };
                }
            });
        }
        PipelineIterator {
            buffer: rx,
            size_hint
        }
    }
}

impl Iterator for PipelineIterator {
    type Item = Item;

    fn next(&mut self) -> Option<Self::Item> {
        if let Ok(item) = self.buffer.recv() {
            Some(item)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.size_hint
    }
}

pub struct BatchedPipelineIterator<I: Iterator<Item=Item>> {
    shuffle: bool,
    seed: u64,
    iter: I,
}

impl<I: Iterator<Item=Item>> Iterator for BatchedPipelineIterator<I> {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}
