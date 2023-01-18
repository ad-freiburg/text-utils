use anyhow::anyhow;
use pyo3::prelude::*;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Write},
    path::Path,
};

use crate::{edit::distances, utils::py_invalid_type_error};

#[pyclass]
pub struct Dictionary {
    inner: HashMap<String, usize>,
}

impl Dictionary {
    pub fn len(&self) -> usize {
        self.inner.len()
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
        Ok(Self { inner })
    }

    pub fn create(files: &[impl AsRef<Path>]) -> anyhow::Result<Self> {
        Ok(Self {
            inner: HashMap::new(),
        })
    }

    pub fn save(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let mut file = File::create(path)?;
        for (key, value) in self.inner.iter() {
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
    #[pyo3(name = "create")]
    fn create_py(files: Vec<&str>) -> anyhow::Result<Self> {
        Self::create(&files)
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    pub fn rel_frequency(&self, freq: usize) -> f64 {
        freq as f64 / self.inner.values().sum::<usize>() as f64
    }

    pub fn contains(&self, s: &str) -> bool {
        self.inner.contains_key(s)
    }

    pub fn get(&self, s: &str) -> Option<usize> {
        self.inner.get(s).copied()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[args(measure = "DictionaryDistanceMeasure::EditDistance")]
    pub fn get_closest(
        &self,
        s: &str,
        measure: DictionaryDistanceMeasure,
    ) -> Option<(String, usize)> {
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
        .unwrap();
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
            Some((term.to_string(), freq))
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
    fn test_dictionary_creation() {}

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
