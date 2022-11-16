use std::fs::File;
use std::io::{BufRead, BufReader, Lines, Read};
use std::path::Path;
use crate::data::{TextData, Item};
use crate::data::preprocessing::{LabelingFn, PreprocessingFn};
use crate::tokenization::{TokenizationFn};

pub struct JSONLineIterator<R: Read> {
    lines: Lines<BufReader<R>>,
    counter: usize,
}

impl<R: Read> JSONLineIterator<R> {
    pub fn new(lines: Lines<BufReader<R>>) -> Self {
        JSONLineIterator { lines, counter: 0 }
    }
}

impl Iterator for JSONLineIterator<File> {
    type Item = TextData;

    fn next(&mut self) -> Option<Self::Item> {
        // no more lines to process
        let Some(line) = self.lines.next() else {
            return None;
        };
        let s = line
            .expect(&format!("could not read line {}", self.counter));
        let data: TextData = serde_json::from_str(&s)
            .expect(&format!("could not deserialize from json string \n{}", s));
        Some(data)
    }
}

pub fn jsonl_iterator_from_file(path: &Path) -> JSONLineIterator<File> {
    let file = File::open(path)
        .expect(&format!("could not open file at {:?}", path));
    let bufreader = BufReader::new(file);
    JSONLineIterator::new(bufreader.lines())
}

pub struct TextLoader<I: Iterator<Item=TextData>> {
    iter: I,
    size: usize,
    preprocessing_fn: PreprocessingFn,
    label_fn: LabelingFn,
    tokenization_fn: TokenizationFn,
    n_threads: u8,
    seed: u64,
    idx: usize,
}

impl<I: Iterator<Item=TextData>> Iterator for TextLoader<I> {
    type Item = Item;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(item) = self.iter.next() else {
            return None;
        };
        let data = (self.preprocessing_fn)(item);
        let label = (self.label_fn)(&data);
        let tokenization = (self.tokenization_fn)(&data.processed);
        Some(Item {
            tokenization,
            data,
            label,
        })
    }
}
