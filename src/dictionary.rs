use anyhow::anyhow;
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
    text::{clean, file_size, split_words},
    unicode::{is_alphabetic, is_punctuation, normalize, Normalization, CS},
    utils::{progress_bar, py_invalid_type_error},
};

#[pyclass]
pub struct Dictionary {
    inner: HashMap<String, usize>,
    pub freq_sum: usize,
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
            let splits: Vec<&str> = line.trim().split('\t').collect();
            if splits.len() != 2 {
                return Err(anyhow!("expected two tab separated values for every line in dictionary file, but got '{line}' on line {idx}"));
            }
            inner.insert(splits[0].to_string(), splits[1].parse()?);
        }
        Ok(Self::new(inner))
    }

    pub fn items(&self) -> impl Iterator<Item = (&String, &usize)> {
        self.inner.iter()
    }

    pub fn create(
        files: &[impl AsRef<Path>],
        max_size: Option<usize>,
        max_sequences: Option<usize>,
        num_threads: u8,
        use_characters: bool,
        char_grams: u8,
        show_progress: bool,
    ) -> anyhow::Result<Self> {
        if use_characters && char_grams != 1 && char_grams != 3 {
            return Err(anyhow!("char_grams must be 1 or 3"));
        }
        let max_size = max_size.unwrap_or(usize::MAX);
        let max_sequences = max_sequences.unwrap_or(usize::MAX);
        let num_total_lines: usize = files
            .iter()
            .map(|path| Ok(file_size(path)?.0))
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .sum();
        let path_bufs: Vec<_> = files
            .iter()
            .map(|path| path.as_ref().to_path_buf())
            .collect();
        let line_iter = path_bufs
            .into_iter()
            .flat_map(move |path| {
                let file = File::open(path).expect("failed to open file");
                let reader = BufReader::new(file);
                reader.lines().map_while(Result::ok)
            })
            .take(max_sequences);
        let line_iter = Arc::new(Mutex::new(line_iter));
        let (count_tx, count_rx) = sync_channel(num_threads as usize);
        let mut threads = vec![];
        for _ in 0..num_threads.max(1) {
            let line_iter_clone = line_iter.clone();
            let count_tx_clone = count_tx.clone();
            let t_handle = thread::spawn(move || loop {
                let Some(mut line) = line_iter_clone.lock().expect("failed to lock line iter").next() else {
                    return;
                };
                line = normalize(&clean(&line, true), Normalization::NFKC, true);
                let counts = if use_characters {
                    split_words(&line)
                        .into_iter()
                        .flat_map(|(word, _)| {
                            let mut chars = vec![];
                            if char_grams > 1 {
                                chars.push("<bow>");
                            }
                            chars.extend(CS::split(word, true));
                            if char_grams > 1 {
                                chars.push("<eow>");
                            }
                            chars
                                .windows(char_grams as usize)
                                .filter(|&window| {
                                    let s = window[window.len() / 2];
                                    is_alphabetic(s) || is_punctuation(s)
                                })
                                .map(|window| window.join(" "))
                                .collect::<Vec<_>>()
                        })
                        .fold(HashMap::new(), |mut counts, token| {
                            if let Some(count) = counts.get_mut(&token) {
                                *count += 1;
                            } else {
                                counts.insert(token, 1);
                            }
                            counts
                        })
                } else {
                    split_words(&line)
                        .into_iter()
                        .filter_map(|(_, parts)| parts)
                        .flat_map(|parts| parts.into_iter().map(|(s, _)| s))
                        .fold(HashMap::new(), |mut counts, token| {
                            if let Some(count) = counts.get_mut(token) {
                                *count += 1;
                            } else {
                                counts.insert(token.to_string(), 1);
                            }
                            counts
                        })
                };
                if count_tx_clone.send(counts).is_err() {
                    return;
                };
            });
            threads.push(t_handle);
        }
        let pbar = progress_bar(
            &format!(
                "counting {} in sequences",
                if use_characters { "chars" } else { "words" }
            ),
            num_total_lines as u64,
            !show_progress,
        );
        drop(count_tx);
        let counts = count_rx
            .into_iter()
            .fold(HashMap::new(), |mut acc, counts| {
                for (word, count) in counts {
                    *acc.entry(word).or_insert(0) += count;
                }
                pbar.inc(1);
                acc
            });
        pbar.finish_and_clear();
        // filter for the top max size words
        // build a binary heap for this and always pop the top element
        let mut heap = BinaryHeap::with_capacity(max_size + 1);
        for (word, freq) in counts {
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
            writeln!(file, "{key}\t{value}")?;
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
        signature = (files, max_size = None, max_sequences = None, num_threads=num_cpus::get() as u8, use_characters = false, char_grams=1, show_progress=false),
    )]
    fn create_py(
        files: Vec<&str>,
        max_size: Option<usize>,
        max_sequences: Option<usize>,
        num_threads: u8,
        use_characters: bool,
        char_grams: u8,
        show_progress: bool,
    ) -> anyhow::Result<Self> {
        Self::create(
            &files,
            max_size,
            max_sequences,
            num_threads,
            use_characters,
            char_grams,
            show_progress,
        )
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
        let ns = normalize(s, Normalization::NFKC, true);
        self.inner.contains_key(&ns)
    }

    pub fn get(&self, s: &str) -> Option<(usize, f64)> {
        let ns = normalize(s, Normalization::NFKC, true);
        self.inner
            .get(&ns)
            .copied()
            .map(|freq| (freq, freq as f64 / self.freq_sum as f64))
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
        let ns = normalize(s, Normalization::NFKC, true);
        let a: Vec<_> = (0..self.len()).map(|_| ns.as_str()).collect();
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
            .zip(freqs)
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
        assert_eq!(d.get("this").unwrap(), (7, 7.0 / 19.0));
        assert_eq!(d.get("is").unwrap(), (4, 4.0 / 19.0));
        assert_eq!(d.get("good").unwrap(), (8, 8.0 / 19.0));
    }

    #[test]
    fn test_dictionary_creation() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path = base.clone().join("resources/test/multi30k.txt");
        let _d = Dictionary::create(
            &[path],
            Some(100),
            Some(1000),
            num_cpus::get() as u8,
            false,
            1,
            true,
        )
        .unwrap();
    }

    #[test]
    fn test_dictionary_functionality() {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path = base.clone().join("resources/test/good_dict.txt");
        let d = Dictionary::load(path).unwrap();
        assert_eq!(
            d.get_closest("god", DictionaryDistanceMeasure::EditDistance)
                .unwrap(),
            ("good".to_string(), 8, 8.0 / 19.0)
        );
        assert_eq!(
            d.get_closest("his", DictionaryDistanceMeasure::EditDistance)
                .unwrap(),
            ("this".to_string(), 7, 7.0 / 19.0)
        );
    }
}
