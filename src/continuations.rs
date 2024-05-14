use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use crate::utils::SerializeMsgPack;
use anyhow::anyhow;
use pyo3::prelude::*;
use text_utils_prefix::{AdaptiveRadixTrie, ContinuationSearch, ContinuationTrie, PrefixSearch};

#[pyclass]
pub struct ContinuationIndex {
    cont_trie: ContinuationTrie<AdaptiveRadixTrie<String>>,
}

pub type ContinuationIndices = (Vec<usize>, Vec<usize>);
#[pymethods]
impl ContinuationIndex {
    #[staticmethod]
    fn load_with_continuations(file: &str, continuations: Vec<Vec<u8>>) -> anyhow::Result<Self> {
        let trie = AdaptiveRadixTrie::load(file)?;
        Ok(Self {
            cont_trie: ContinuationTrie::new(trie, continuations),
        })
    }

    #[staticmethod]
    fn build_from_file(path: &str, out: &str, common_suffix: Option<&str>) -> anyhow::Result<()> {
        // build patricia trie similar to prefix vec above
        let file = File::open(path)?;
        let sfx = common_suffix.unwrap_or_default().as_bytes();
        // now loop over lines and insert them into the trie
        let mut trie = AdaptiveRadixTrie::default();
        for line in BufReader::new(file).lines() {
            let line = line?;
            let splits: Vec<_> = line.trim_end_matches(&['\r', '\n']).split('\t').collect();
            if splits.len() < 2 {
                return Err(anyhow!("invalid line: {}", line));
            }
            let value = splits[0];
            for s in &splits[1..] {
                let mut s = s.as_bytes().to_vec();
                s.extend_from_slice(sfx);
                trie.insert(&s, value.to_string());
            }
        }
        // save art to out file
        trie.save(out)?;
        Ok(())
    }

    fn get_value(&self, key: &[u8]) -> Option<String> {
        self.cont_trie.get(key).cloned()
    }

    fn get_continuation(&self, index: usize) -> Option<&[u8]> {
        self.cont_trie
            .continuations
            .get(index)
            .map(|c| c.as_slice())
    }

    fn get(&self, prefix: &[u8]) -> (Vec<usize>, Option<String>) {
        (
            self.cont_trie.contains_continuations(prefix),
            self.cont_trie.get(prefix).cloned(),
        )
    }

    fn batch_continuation_indices(
        &self,
        prefixes: Vec<Vec<u8>>,
    ) -> (ContinuationIndices, Vec<Option<String>>) {
        (
            self.cont_trie
                .batch_contains_continuations(&prefixes)
                .into_iter()
                .enumerate()
                .fold(
                    (vec![], vec![]),
                    |(mut batch_indices, mut cont_indices), (i, cont)| {
                        for c in cont {
                            batch_indices.push(i);
                            cont_indices.push(c);
                        }
                        (batch_indices, cont_indices)
                    },
                ),
            prefixes
                .iter()
                .map(|prefix| self.cont_trie.get(prefix).cloned())
                .collect(),
        )
    }
}

/// A submodule containing python implementations of a continuation trie
pub(super) fn add_submodule(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "continuations")?;
    m.add_class::<ContinuationIndex>()?;
    parent_module.add_submodule(m)?;

    Ok(())
}
