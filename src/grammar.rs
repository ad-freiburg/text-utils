use std::sync::Arc;

use anyhow::anyhow;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use text_utils_grammar::{
    Constraint, LR1GrammarConstraint, LR1GrammarParser, LR1NextState, LR1Parse, LR1State,
    RegularExpressionConstraint, RegularExpressionState,
};

#[pyclass]
struct RegexConstraint {
    inner: Arc<RegularExpressionConstraint>,
    state: RegularExpressionState,
    indices: Vec<usize>,
    is_match: bool,
    next_states: Vec<RegularExpressionState>,
}

#[pymethods]
impl RegexConstraint {
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
        let (indices, next_states) = inner.get_valid_continuations_with_state(&state);
        let is_match = inner.is_match_state(&state);
        Ok(Self {
            inner: Arc::new(inner),
            state,
            indices,
            is_match,
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
        let (indices, next_states) = inner.get_valid_continuations_with_state(&state);
        let is_match = inner.is_match_state(&state);
        Ok(Self {
            inner: Arc::new(inner),
            state,
            indices,
            is_match,
            next_states,
        })
    }

    fn reset(&mut self, prefix: Option<Vec<u8>>) {
        self.state = self
            .inner
            .get_state(&prefix.unwrap_or_default())
            .expect("failed to reset to given prefix");
        let (indices, next_states) = self.inner.get_valid_continuations_with_state(&self.state);
        self.indices = indices;
        self.is_match = self.inner.is_match_state(&self.state);
        self.next_states = next_states;
    }

    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            state: self.state,
            indices: self.indices.clone(),
            is_match: self.is_match,
            next_states: self.next_states.clone(),
        }
    }

    fn get_constraint(&self) -> (Vec<usize>, bool) {
        (self.indices.clone(), self.is_match)
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
        let (indices, states) = self.inner.get_valid_continuations_with_state(&self.state);
        self.indices = indices;
        self.is_match = self.inner.is_match_state(&self.state);
        self.next_states = states;
        Ok(())
    }
}

#[pyclass]
struct LR1Constraint {
    inner: Arc<LR1GrammarConstraint>,
    state: LR1State,
    indices: Vec<usize>,
    is_match: bool,
    next_states: Vec<LR1NextState>,
}

#[pymethods]
impl LR1Constraint {
    #[new]
    fn new(grammar: &str, lexer: &str, continuations: Vec<Vec<u8>>) -> anyhow::Result<Self> {
        let inner = LR1GrammarConstraint::new(grammar, lexer, continuations)
            .map_err(|e| anyhow!("failed to create LR(1) grammar constraint: {}", e))?;
        let state = inner.get_start_state();
        let (indices, next_states) = inner.get_valid_continuations_with_state(&state);
        let is_match = inner.is_match_state(&state);
        Ok(Self {
            inner: Arc::new(inner),
            state,
            indices,
            is_match,
            next_states,
        })
    }

    #[staticmethod]
    fn from_files(
        grammar_path: &str,
        lexer_path: &str,
        continuations: Vec<Vec<u8>>,
    ) -> anyhow::Result<Self> {
        let inner = LR1GrammarConstraint::from_files(grammar_path, lexer_path, continuations)
            .map_err(|e| {
                anyhow!(
                    "failed to create LR(1) grammar constraint from files {} and {}: {}",
                    grammar_path,
                    lexer_path,
                    e
                )
            })?;
        let state = inner.get_start_state();
        let (indices, next_states) = inner.get_valid_continuations_with_state(&state);
        let is_match = inner.is_match_state(&state);
        Ok(Self {
            inner: Arc::new(inner),
            state,
            indices,
            is_match,
            next_states,
        })
    }

    fn reset(&mut self, prefix: Option<Vec<u8>>) {
        self.state = self
            .inner
            .get_state(&prefix.unwrap_or_default())
            .expect("failed to reset to given prefix");
        let (indices, next_states) = self.inner.get_valid_continuations_with_state(&self.state);
        self.indices = indices;
        self.is_match = self.inner.is_match_state(&self.state);
        self.next_states = next_states;
    }

    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            state: self.state.clone(),
            indices: self.indices.clone(),
            is_match: self.is_match,
            next_states: self.next_states.clone(),
        }
    }

    fn get_constraint(&self) -> (Vec<usize>, bool) {
        (self.indices.clone(), self.is_match)
    }

    fn next(&mut self, index: usize) -> anyhow::Result<()> {
        let idx = self.indices.binary_search(&index).map_err(|_| {
            anyhow!(
                "index {} not found in valid constraint indices: {:?}",
                index,
                self.indices
            )
        })?;
        self.state.next(std::mem::take(&mut self.next_states[idx]));
        let (indices, states) = self.inner.get_valid_continuations_with_state(&self.state);
        self.indices = indices;
        self.is_match = self.inner.is_match_state(&self.state);
        self.next_states = states;
        Ok(())
    }
}

#[pyclass]
pub struct LR1Parser {
    inner: LR1GrammarParser,
}

#[pymethods]
impl LR1Parser {
    #[new]
    fn new(grammar: &str, lexer: &str) -> anyhow::Result<Self> {
        let inner = LR1GrammarParser::new(grammar, lexer).map_err(|e| {
            anyhow!(
                "failed to create LR(1) grammar parser from grammar {} and lexer {}: {}",
                grammar,
                lexer,
                e
            )
        })?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_files(grammar_path: &str, lexer_path: &str) -> anyhow::Result<Self> {
        let inner = LR1GrammarParser::from_files(grammar_path, lexer_path).map_err(|e| {
            anyhow!(
                "failed to create LR(1) grammar parser from files {} and {}: {}",
                grammar_path,
                lexer_path,
                e
            )
        })?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (input, collapse=false))]
    fn parse_pretty(&self, input: &str, collapse: bool) -> anyhow::Result<String> {
        let parse = self
            .inner
            .parse(input, collapse)
            .map_err(|e| anyhow!("failed to parse input: {e}"))?;
        Ok(parse.pretty(input, collapse))
    }

    #[pyo3(signature = (input, collapse=false))]
    fn parse(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        input: &str,
        collapse: bool,
    ) -> anyhow::Result<PyObject> {
        let parse = slf
            .inner
            .parse(input, collapse)
            .map_err(|e| anyhow!("failed to parse input: {e}"))?;
        let chars: Vec<_> = input.chars().collect();
        Ok(parse_into_py(input, &chars, &parse, py)?)
    }
}

fn byte_range_to_char_range(
    s: &str,
    chars: &[char],
    byte_start: usize,
    byte_end: usize,
) -> Option<(usize, usize)> {
    if byte_start >= byte_end || byte_end > s.len() {
        return None;
    }
    let mut char_start = None;
    let mut char_end = None;

    let mut curr_byte_index = 0;

    for (i, char) in chars.iter().enumerate() {
        let char_bytes = char.len_utf8();

        if curr_byte_index == byte_start {
            char_start = Some(i);
        }

        curr_byte_index += char_bytes;

        if curr_byte_index == byte_end {
            char_end = Some(i + 1);
            break;
        }
    }

    match (char_start, char_end) {
        (Some(start), Some(end)) => Some((start, end)),
        _ => None, // Invalid byte range
    }
}

fn parse_into_py(
    text: &str,
    chars: &[char],
    parse: &LR1Parse<'_>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    match parse {
        LR1Parse::Empty(_) => unreachable!("empty parse should not be returned"),
        LR1Parse::Terminal(name, span) => {
            dict.set_item("name", name)?;
            let &(start, len) = span;
            dict.set_item("value", &text[start..start + len])?;
            let (start, end) = byte_range_to_char_range(text, chars, start, start + len)
                .ok_or_else(|| PyErr::new::<PyValueError, _>("invalid byte range"))?;
            dict.set_item("span", (start, end))?;
        }
        LR1Parse::NonTerminal(name, children, span) => {
            dict.set_item("name", name)?;
            let children = PyList::new(
                py,
                children
                    .iter()
                    .map(|c| parse_into_py(text, chars, c, py))
                    .collect::<PyResult<Vec<_>>>()?,
            );
            dict.set_item("children", children)?;
            let &(start, len) = span;
            let (start, end) = byte_range_to_char_range(text, chars, start, start + len)
                .ok_or_else(|| PyErr::new::<PyValueError, _>("invalid byte range"))?;
            dict.set_item("span", (start, end))?;
        }
    };
    Ok(dict.into())
}

/// A submodule containing python implementations of regex and CFG (LR1) constraints
pub(super) fn add_submodule(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "grammar")?;
    m.add_class::<RegexConstraint>()?;
    m.add_class::<LR1Constraint>()?;
    m.add_class::<LR1Parser>()?;
    parent_module.add_submodule(m)?;

    Ok(())
}
