use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use crate::utils::SerializeMsgPack;
use anyhow::anyhow;
use pyo3::prelude::*;
use text_utils_prefix::{AdaptiveRadixTrie, ContinuationSearch, ContinuationTrie, PrefixSearch};

#[pyclass]
pub struct Continuations {
    continuations: ContinuationTrie<AdaptiveRadixTrie<u32>>,
}

#[pymethods]
impl Continuations {
    #[staticmethod]
    fn load_with_continuations(file: &str, continuations: Vec<Vec<u8>>) -> anyhow::Result<Self> {
        let trie = AdaptiveRadixTrie::load(file)?;
        Ok(Self {
            continuations: ContinuationTrie::new(trie, continuations),
        })
    }

    #[staticmethod]
    fn build_from_file(path: &str, out: &str) -> anyhow::Result<()> {
        // build patricia trie similar to prefix vec above
        let file = File::open(path)?;
        // now loop over lines and insert them into the trie
        let mut trie = AdaptiveRadixTrie::default();
        for line in BufReader::new(file).lines() {
            let line = line?;
            let splits: Vec<_> = line.split('\t').collect();
            if splits.len() < 2 {
                return Err(anyhow!("invalid line: {}", line));
            }
            let value: u32 = splits[0].trim().parse()?;
            for s in &splits[1..] {
                trie.insert(s.trim().as_bytes(), value);
            }
        }
        // save art to out file
        trie.save(out)?;
        Ok(())
    }

    fn continuation_indices(&self, prefix: &[u8]) -> Vec<usize> {
        self.continuations.contains_continuations(prefix)
    }

    fn batch_continuation_indices(&self, prefixes: Vec<Vec<u8>>) -> (Vec<usize>, Vec<usize>) {
        self.continuations
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
            )
    }
}

/// A submodule containing python implementations of a continuation trie
pub(super) fn add_submodule(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "continuations")?;
    m.add_class::<Continuations>()?;
    parent_module.add_submodule(m)?;

    Ok(())
}
