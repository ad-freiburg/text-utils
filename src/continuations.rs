use std::{
    fs::File,
    io::{BufRead, BufReader},
    sync::Arc,
};

use crate::utils::SerializeMsgPack;
use anyhow::{anyhow, Result};
use numpy::{ndarray::Array1, IntoPyArray};
use pyo3::{prelude::*, pybacked::PyBackedStr};
use text_utils_prefix::{optimized_continuation_permutation, AdaptiveRadixContinuationTrie};

#[pyclass]
pub struct ContinuationIndex {
    trie: Arc<AdaptiveRadixContinuationTrie<String>>,
    continuations: Vec<Vec<u8>>,
    permutation: Vec<usize>,
    skips: Vec<usize>,
}

#[pymethods]
impl ContinuationIndex {
    #[staticmethod]
    fn load(file: &str) -> anyhow::Result<Self> {
        let trie = Arc::new(AdaptiveRadixContinuationTrie::load(file)?);
        Ok(Self {
            trie,
            continuations: vec![],
            permutation: vec![],
            skips: vec![],
        })
    }

    #[staticmethod]
    fn load_with_continuations(file: &str, continuations: Vec<Vec<u8>>) -> anyhow::Result<Self> {
        let trie = Arc::new(AdaptiveRadixContinuationTrie::load(file)?);
        let (permutation, skips) = optimized_continuation_permutation(&continuations);
        Ok(Self {
            trie,
            continuations,
            permutation,
            skips,
        })
    }

    #[staticmethod]
    fn build_from_file(path: &str, out: &str, common_suffix: Option<&str>) -> anyhow::Result<()> {
        // build patricia trie similar to prefix vec above
        let file = File::open(path)?;
        let sfx = common_suffix.unwrap_or_default().as_bytes();
        // now loop over lines and insert them into the trie
        let trie: AdaptiveRadixContinuationTrie<_> = BufReader::new(file)
            .lines()
            .flat_map(|line| {
                let Ok(line) = line else {
                    return vec![Err(anyhow!("failed to read line"))];
                };
                let splits: Vec<_> = line.trim_end_matches(&['\r', '\n']).split('\t').collect();
                if splits.len() < 2 {
                    return vec![Err(anyhow!("invalid line: {}", line))];
                }
                let value = splits[0];
                splits[1..]
                    .iter()
                    .map(move |s| {
                        let mut s = s.as_bytes().to_vec();
                        s.extend_from_slice(sfx);
                        Ok((s, value.to_string()))
                    })
                    .collect()
            })
            .collect::<Result<_>>()?;
        // save art to out file
        trie.save(out)?;
        Ok(())
    }

    fn clone_with_continuations(&self, continuations: Vec<Vec<u8>>) -> Self {
        let (permutation, skips) = optimized_continuation_permutation(&continuations);
        Self {
            trie: self.trie.clone(),
            continuations,
            permutation,
            skips,
        }
    }

    fn set_continuations(&mut self, continuations: Vec<Vec<u8>>) {
        let (permutation, skips) = optimized_continuation_permutation(&continuations);
        self.continuations = continuations;
        self.permutation = permutation;
        self.skips = skips;
    }

    fn get(&self, py: Python<'_>, prefix: &[u8]) -> (PyObject, Option<String>) {
        (
            Array1::from_vec(self.trie.continuation_indices(
                prefix,
                &self.continuations,
                &self.permutation,
                &self.skips,
            ))
            .into_pyarray_bound(py)
            .into_py(py),
            self.trie.get(prefix).cloned(),
        )
    }

    fn sub_index_by_values(&self, values: Vec<PyBackedStr>) -> Self {
        let sub_trie = self
            .trie
            .sub_index_by_values(values.iter().map(|v| -> &str { v.as_ref() }));
        Self {
            trie: Arc::new(sub_trie),
            continuations: self.continuations.clone(),
            permutation: self.permutation.clone(),
            skips: self.skips.clone(),
        }
    }

    fn continuation_at(&self, index: usize) -> anyhow::Result<Vec<u8>> {
        self.continuations.get(index).cloned().ok_or_else(|| {
            anyhow!(
                "index out of bounds: {} (continuations length: {})",
                index,
                self.continuations.len()
            )
        })
    }
}

/// A submodule containing python implementations of a continuation trie
pub(super) fn add_submodule(py: Python, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(py, "continuations")?;
    m.add_class::<ContinuationIndex>()?;
    parent_module.add_submodule(&m)?;

    Ok(())
}
