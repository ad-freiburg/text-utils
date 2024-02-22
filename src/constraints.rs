use std::sync::Arc;

use anyhow::anyhow;
use pyo3::prelude::*;
use text_utils_constraints::{Constraint, RegularExpressionConstraint, RegularExpressionState};

#[pyclass]
struct Regex {
    inner: Arc<RegularExpressionConstraint>,
    state: RegularExpressionState,
    indices: Vec<usize>,
    next_states: Vec<RegularExpressionState>,
}

#[pymethods]
impl Regex {
    #[new]
    fn new(pattern: &str, continuations: Vec<Vec<u8>>) -> anyhow::Result<Self> {
        let inner = RegularExpressionConstraint::new(pattern, continuations).map_err(|e| {
            anyhow!(
                "failed to create regular expression constraint from pattern '{}': {}",
                pattern,
                e
            )
        })?;
        let state = inner.get_start_state();
        let (indices, next_states) = inner.get_valid_continuations_with_state(state);
        Ok(Self {
            inner: Arc::new(inner),
            state,
            indices,
            next_states,
        })
    }

    #[staticmethod]
    fn from_file(path: &str, continuations: Vec<Vec<u8>>) -> anyhow::Result<Self> {
        let inner = RegularExpressionConstraint::from_file(path, continuations).map_err(|e| {
            anyhow!(
                "failed to create regular expression constraint from file '{}': {}",
                path,
                e
            )
        })?;
        let state = inner.get_start_state();
        let (indices, next_states) = inner.get_valid_continuations_with_state(state);
        Ok(Self {
            inner: Arc::new(inner),
            state,
            indices,
            next_states,
        })
    }

    fn reset(&mut self, prefix: Option<Vec<u8>>) {
        self.state = self.inner.get_state(&prefix.unwrap_or_default());
        let (indices, next_states) = self.inner.get_valid_continuations_with_state(self.state);
        self.indices = indices;
        self.next_states = next_states;
    }

    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            state: self.state,
            indices: self.indices.clone(),
            next_states: self.next_states.clone(),
        }
    }

    fn get_constraint_indices(&self) -> Vec<usize> {
        self.indices.clone()
    }

    fn is_final_state(&self) -> bool {
        self.inner.is_match_state(self.state)
    }

    fn next(&mut self, index: usize) -> anyhow::Result<()> {
        let idx = self.indices.binary_search(&index).map_err(|_| {
            anyhow!(
                "index {} not found in valid constraint indices: {:?}",
                index,
                self.indices
            )
        })?;
        self.state = self.next_states[idx];
        let (indices, states) = self.inner.get_valid_continuations_with_state(self.state);
        self.indices = indices;
        self.next_states = states;
        Ok(())
    }
}

/// A submodule containing python implementations of regex and CFG constraints
pub(super) fn add_submodule(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "constraints")?;
    m.add_class::<Regex>()?;
    parent_module.add_submodule(m)?;

    Ok(())
}
