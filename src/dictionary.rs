use anyhow::anyhow;
use indicatif::MultiProgress;
use itertools::Itertools;
use pyo3::prelude::*;
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    fs::File,
    io::{BufRead, BufReader, Write},
    path::Path,
    sync::{mpsc::sync_channel, Arc, Mutex},
    thread,
};

use crate::{
    edit::distances,
    text::{clean, file_size},
    unicode::{normalize, Normalization, CS},
    utils::{progress_bar, py_invalid_type_error},
};

#[pyclass]
pub struct Dictionary {
    inner: HashMap<String, usize>,
    freq_sum: usize,
}

impl Dictionary {
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    fn new(inner: HashMap<String, usize>) -> Self {
        let freq_sum = inner.values().sum();
        Self { inner, freq_sum }
    }

    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let mut inner = HashMap::new();
        for (idx, line) in BufReader::new(file).lines().enumerate() {
            let line = line?;
            let splits: Vec<&str> = line.trim().split("\t").collect();
            if splits.len() != 2 {
                return Err(anyhow!("expected two tab separated values for every line in dictionary file, but got '{line}' on line {idx}"));
            }
            inner.insert(splits[0].to_string(), splits[1].parse()?);
        }
        Ok(Self::new(inner))
    }

    pub fn create(
        files: &[impl AsRef<Path>],
        max_size: usize,
        num_threads: u8,
        show_progress: bool,
    ) -> anyhow::Result<Self> {
        let inner = Arc::new(Mutex::new(HashMap::new()));
        let (tx, rx) = sync_channel::<Vec<String>>(num_threads as usize);
        let rx = Arc::new(Mutex::new(rx));
        let mut threads = vec![];
        for _ in 0..num_threads.max(1) {
            let rx_clone = rx.clone();
            let inner_clone = inner.clone();
            let t_handle = thread::spawn(move || {
                while let Ok(lines) = rx_clone.lock().unwrap().recv() {
                    let mut counts = HashMap::new();
                    for line in lines {
                        let line = clean(&line, true);
                        let line = normalize(&line, Normalization::NFKC, true);
                        let words: Vec<String> =
                            line.split_whitespace().map(|s| s.to_string()).collect();
                        for word in words {
                            *counts.entry(word).or_insert(0) += 1;
                        }
                    }
                    let mut inner = inner_clone.lock().unwrap();
                    for (word, freq) in counts {
                        *inner.entry(word).or_insert(0) += freq;
                    }
                }
            });
            threads.push(t_handle);
        }
        let file_p_bar = progress_bar("processing files", files.len() as u64, !show_progress);
        let multi_p_bar = MultiProgress::new();
        multi_p_bar.add(file_p_bar.clone());
        for file in files {
            let (num_lines, _) = file_size(file)?;
            let chunk_size = (num_lines / num_threads as usize).max(1).min(4096);
            let mut file_name = file.as_ref().to_string_lossy().to_string();
            if file_name.len() > 16 {
                file_name = "...".to_string() + &file_name[file_name.len() - 13..];
            }
            let line_p_bar = progress_bar(
                &format!("processing lines of {}", file_name),
                num_lines as u64,
                !show_progress,
            );
            multi_p_bar.insert_after(&file_p_bar, line_p_bar.clone());
            let lines = BufReader::new(File::open(file)?).lines();
            for line_chunk in &lines.chunks(chunk_size) {
                let line_chunk: Vec<String> = line_chunk.filter_map(|l| l.ok()).collect();
                let line_chunk_len = line_chunk.len();
                tx.send(line_chunk)?;
                line_p_bar.inc(line_chunk_len as u64);
            }
            line_p_bar.finish_and_clear();
            file_p_bar.inc(1);
        }
        file_p_bar.finish();
        // we are done sending, drop the sender to signal
        // to the thread receivers they should stop
        drop(tx);
        for t in threads {
            t.join().expect("failed to join thread");
        }
        // filter for the top max size words
        // build a binary heap for this and always pop the top element
        let mut heap = BinaryHeap::with_capacity(max_size + 1);
        let inner = Arc::try_unwrap(inner)
            .expect("should not fail")
            .into_inner()
            .expect("should not fail");
        for (word, freq) in inner {
            heap.push(Reverse((freq, word)));
            if heap.len() > max_size {
                heap.pop();
            }
        }
        let mut inner = HashMap::with_capacity(max_size);
        while let Some(Reverse((freq, word))) = heap.pop() {
            inner.insert(word, freq);
        }
        Ok(Self::new(inner))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let mut file = File::create(path)?;
        // when saving dictionary, do it in sorted order
        // because the file looks nicer this way
        let mut items: Vec<_> = self.inner.iter().collect();
        items.sort_by_key(|item| Reverse(item.1));
        for &(key, value) in items.iter() {
            writeln!(file, "{}\t{}", key, value)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DictionaryDistanceMeasure {
    EditDistance,
    NormalizedEditDistance,
}

impl<'a> FromPyObject<'a> for DictionaryDistanceMeasure {
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        let s: String = obj.extract()?;
        let dist = match s.as_str() {
            "ed" | "edit_distance" => Self::EditDistance,
            "ned" | "normalized_edit_distance" => Self::NormalizedEditDistance,
            k => return Err(py_invalid_type_error(k, "dictionary distance measure")),
        };
        Ok(dist)
    }
}

#[pymethods]
impl Dictionary {
    #[staticmethod]
    #[pyo3(name = "load")]
    pub fn load_py(path: &str) -> anyhow::Result<Self> {
        Self::load(path)
    }

    #[pyo3(name = "save")]
    fn save_py(&self, path: &str) -> anyhow::Result<()> {
        self.save(path)
    }

    #[staticmethod]
    #[pyo3(
        name = "create",
        signature = (files, max_size, num_threads=(num_cpus::get() as u8).min(4), show_progress=false),
    )]
    fn create_py(
        files: Vec<&str>,
        max_size: usize,
        num_threads: u8,
        show_progress: bool,
    ) -> anyhow::Result<Self> {
        Self::create(&files, max_size, num_threads, show_progress)
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    #[pyo3(signature = (weighted_by_freq = true))]
    pub fn avg_length(&self, weighted_by_freq: bool) -> f64 {
        let sum = self
            .inner
            .iter()
            .map(|(s, freq)| CS::new(s, true).len() * if weighted_by_freq { *freq } else { 1 })
            .sum::<usize>();
        if weighted_by_freq {
            sum as f64 / self.freq_sum as f64
        } else {
            sum as f64 / self.len() as f64
        }
    }

    pub fn contains(&self, s: &str) -> bool {
        self.inner.contains_key(s)
    }

    pub fn get(&self, s: &str) -> Option<(usize, f64)> {
        self.inner
            .get(s)
            .copied()
            .map(|freq| (freq as usize, freq as f64 / self.freq_sum as f64))
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[pyo3(signature = (s, measure = DictionaryDistanceMeasure::EditDistance))]
    pub fn get_closest(
        &self,
        s: &str,
        measure: DictionaryDistanceMeasure,
    ) -> Option<(String, usize, f64)> {
        if self.is_empty() {
            return None;
        }
        let a: Vec<&str> = (0..self.len()).map(|_| s).collect();
        let mut b: Vec<&str> = Vec::with_capacity(self.len());
        let mut v: Vec<usize> = Vec::with_capacity(self.len());
        for (key, value) in &self.inner {
            b.push(key);
            v.push(*value);
        }
        let dists = distances(
            &a,
            &b,
            true,
            false,
            false,
            measure == DictionaryDistanceMeasure::NormalizedEditDistance,
        )
        .expect("should not fail");
        let mut min_dist = f64::INFINITY;
        let mut terms = vec![];
        let mut freqs = vec![];
        for i in 0..dists.len() {
            if dists[i] < min_dist {
                min_dist = dists[i];
                terms = vec![b[i]];
                freqs = vec![v[i]];
            } else if dists[i] == min_dist {
                terms.push(b[i]);
                freqs.push(v[i]);
            }
        }
        if let Some((term, freq)) = terms
            .into_iter()
            .zip(freqs.into_iter())
            .max_by(|(_, a), (_, b)| a.cmp(b))
        {
            Some((term.to_string(), freq, freq as f64 / self.freq_sum as f64))
        } else {
            None
        }
    }
}

/// A submodule for creating and querying dictionaries.
pub(super) fn add_submodule(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m_name = "dictionary";
    let m = PyModule::new(py, m_name)?;
    m.add_class::<Dictionary>()?;
    parent_module.add_submodule(m)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::dictionary::DictionaryDistanceMeasure;

    use super::Dictionary;

    #[test]
    fn test_dictionary_loading() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path = base.clone().join("resources/test/bad_dict.txt");
        let d = Dictionary::load(path);
        assert!(d.is_err());
        let path = base.clone().join("resources/test/good_dict.txt");
        let d = Dictionary::load(path).unwrap();
        assert_eq!(d.get("this").unwrap(), 7);
        assert_eq!(d.get("is").unwrap(), 4);
        assert_eq!(d.get("good").unwrap(), 8);
    }

    #[test]
    fn test_dictionary_creation() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path1 = base.clone().join("resources/test/multi30k.txt");
        let path2 = base.clone().join("resources/test/multi30k_rev.txt");
        let _d = Dictionary::create(&[path1, path2], (num_cpus::get() as u8).min(4), true).unwrap();
    }

    #[test]
    fn test_dictionary_functionality() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path = base.clone().join("resources/test/good_dict.txt");
        let d = Dictionary::load(path).unwrap();
        assert_eq!(
            d.get_closest("god", DictionaryDistanceMeasure::EditDistance)
                .unwrap(),
            ("good".to_string(), 8)
        );
        assert_eq!(
            d.get_closest("his", DictionaryDistanceMeasure::EditDistance)
                .unwrap(),
            ("this".to_string(), 7)
        );
    }
}
