use std::sync::Arc;

use anyhow::anyhow;
use numpy::{ndarray::Array1, IntoPyArray};
use pyo3::{prelude::*, pybacked::PyBackedStr};
use text_utils_prefix::{optimized_continuation_permutation, ArtMmapContinuationTrie};

struct Continuations {
    continuations: Vec<Vec<u8>>,
    permutation: Vec<usize>,
    skips: Vec<usize>,
}

#[pyclass]
pub struct MmapContinuationIndex {
    trie: Arc<ArtMmapContinuationTrie>,
    continuations: Arc<Continuations>,
}

#[pymethods]
impl MmapContinuationIndex {
    #[staticmethod]
    #[pyo3(signature = (data, dir, common_suffix = None))]
    fn load(data: &str, dir: &str, common_suffix: Option<&str>) -> anyhow::Result<Self> {
        let trie = Arc::new(ArtMmapContinuationTrie::load(data, dir, common_suffix)?);
        Ok(Self {
            trie,
            continuations: Arc::new(Continuations {
                continuations: vec![],
                permutation: vec![],
                skips: vec![],
            }),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (data, dir, continuations, common_suffix = None))]
    fn load_with_continuations(
        data: &str,
        dir: &str,
        continuations: Vec<Vec<u8>>,
        common_suffix: Option<&str>,
    ) -> anyhow::Result<Self> {
        let trie = Arc::new(ArtMmapContinuationTrie::load(data, dir, common_suffix)?);
        let (permutation, skips) = optimized_continuation_permutation(&continuations);
        Ok(Self {
            trie,
            continuations: Arc::new(Continuations {
                continuations,
                permutation,
                skips,
            }),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (data, output_dir, common_suffix = None))]
    fn build_from_file(
        data: &str,
        output_dir: &str,
        common_suffix: Option<&str>,
    ) -> anyhow::Result<()> {
        ArtMmapContinuationTrie::build(data, output_dir, common_suffix)
    }

    fn clone(&self) -> Self {
        Self {
            trie: self.trie.clone(),
            continuations: self.continuations.clone(),
        }
    }

    fn clone_with_continuations(&self, continuations: Vec<Vec<u8>>) -> Self {
        let (permutation, skips) = optimized_continuation_permutation(&continuations);
        Self {
            trie: self.trie.clone(),
            continuations: Arc::new(Continuations {
                continuations,
                permutation,
                skips,
            }),
        }
    }

    fn set_continuations(&mut self, continuations: Vec<Vec<u8>>) {
        let (permutation, skips) = optimized_continuation_permutation(&continuations);
        self.continuations = Arc::new(Continuations {
            continuations,
            permutation,
            skips,
        });
    }

    fn get(&self, py: Python<'_>, prefix: &[u8]) -> (PyObject, Option<String>) {
        (
            Array1::from_vec(self.trie.continuation_indices(
                prefix,
                &self.continuations.continuations,
                &self.continuations.permutation,
                &self.continuations.skips,
            ))
            .into_pyarray_bound(py)
            .into_py(py),
            self.trie
                .get(prefix)
                .map(|s| String::from_utf8_lossy(s).to_string()),
        )
    }

    fn sub_index_by_values(&self, values: Vec<PyBackedStr>) -> Self {
        let sub_trie = self
            .trie
            .sub_index_by_values(values.iter().map(|v| -> &str { v.as_ref() }));
        Self {
            trie: Arc::new(sub_trie),
            continuations: self.continuations.clone(),
        }
    }

    fn continuation_at(&self, index: usize) -> anyhow::Result<Vec<u8>> {
        self.continuations
            .continuations
            .get(index)
            .cloned()
            .ok_or_else(|| {
                anyhow!(
                    "index out of bounds: {} (continuations length: {})",
                    index,
                    self.continuations.continuations.len()
                )
            })
    }
}

/// A submodule containing python implementations of a continuation trie
pub(super) fn add_submodule(py: Python, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(py, "continuations")?;
    m.add_class::<MmapContinuationIndex>()?;
    parent_module.add_submodule(&m)?;

    Ok(())
}
