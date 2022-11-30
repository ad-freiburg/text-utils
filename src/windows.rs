use crate::unicode::CS;
use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Debug, Clone)]
pub enum InferenceResult {
    SequenceClassification(Vec<usize>),
    SequenceGeneration(Vec<usize>),
}

impl IntoPy<Py<PyDict>> for InferenceResult {
    fn into_py(self, py: Python<'_>) -> Py<PyDict> {
        let d = PyDict::new(py);
        d.into()
    }
}

impl<'a> FromPyObject<'a> for InferenceResult {
    fn extract(_: &'a PyAny) -> PyResult<Self> {
        Ok(InferenceResult::SequenceGeneration(vec![]))
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct InferenceWindow {
    #[pyo3(get)]
    ctx_start: usize,
    #[pyo3(get)]
    ctx_end: usize,
    #[pyo3(get)]
    window_start: usize,
    #[pyo3(get)]
    window_end: usize,
    #[pyo3(get)]
    str: String,
}

pub fn char_windows(
    s: &str,
    max_length: usize,
    context_length: usize,
    use_graphemes: bool,
) -> Vec<InferenceWindow> {
    assert!(
        max_length > 2 * context_length,
        "max length must be larger than 2 times the context \
        length, otherwise there are no tokens left for the window itself"
    );
    let window_length = max_length - 2 * context_length;
    let cs = CS::new(s, use_graphemes);
    (0..cs.len())
        .step_by(window_length)
        .map(|window_start| {
            let window_length = max_length - (1 + (window_start > 0) as usize) * context_length;
            let ctx_start = window_start.saturating_sub(context_length);
            let ctx_end = cs.len().min(window_start + window_length + context_length);
            let window_end = cs.len().min(window_start + window_length);
            InferenceWindow {
                ctx_start,
                ctx_end,
                window_start,
                window_end,
                str: cs.sub(ctx_start, ctx_end).to_string(),
            }
        })
        .collect()
}

#[pyfunction(use_graphemes = "true")]
#[pyo3(name = "char_windows")]
fn char_windows_py(
    s: &str,
    max_length: usize,
    context_length: usize,
    use_graphemes: bool,
) -> PyResult<Vec<InferenceWindow>> {
    Ok(char_windows(s, max_length, context_length, use_graphemes))
}

#[inline]
fn count_until(mut iter: impl Iterator<Item = usize>, max_length: usize, cs: &CS) -> usize {
    iter.fold_while(0usize, |acc, idx| {
        let next_acc = acc + cs.char_byte_len(idx);
        if next_acc > max_length {
            Done(acc)
        } else {
            Continue(next_acc)
        }
    })
    .into_inner()
}

pub fn byte_windows(
    s: &str,
    max_length: usize,
    context_length: usize,
    use_graphemes: bool,
) -> Vec<InferenceWindow> {
    assert!(
        max_length > 2 * context_length,
        "max length must be larger than 2 times the context \
        length, otherwise there are no tokens left for the window itself"
    );
    let cs = CS::new(s, use_graphemes);

    let mut windows = vec![];
    let mut window_start = 0;
    while window_start < cs.len() {
        let window_length = max_length - (1 + (window_start > 0) as usize) * context_length;
        let window_end = window_start + count_until(window_start..cs.len(), window_length, &cs);
        assert!(
            window_end > window_start,
            "{}",
            format!(
                "single character at position {window_start} has more bytes \
                ({}) than the window length {window_length}, \
                this suggests that something with your input string is wrong or the window length \
                is too small",
                cs.char_byte_len(window_start)
            )
        );
        let ctx_start =
            window_start.saturating_sub(count_until((0..window_start).rev(), context_length, &cs));
        let ctx_end = window_end + count_until(window_end..cs.len(), context_length, &cs);

        windows.push(InferenceWindow {
            ctx_start,
            ctx_end,
            window_start,
            window_end,
            str: cs.sub(ctx_start, ctx_end).to_string(),
        });

        window_start = window_end;
    }
    windows
}

#[pyfunction(use_graphemes = "true")]
#[pyo3(name = "byte_windows")]
fn byte_windows_py(
    s: &str,
    max_bytes: usize,
    context_length: usize,
    use_graphemes: bool,
) -> PyResult<Vec<InferenceWindow>> {
    Ok(byte_windows(s, max_bytes, context_length, use_graphemes))
}

/// A submodule containing helper functions needed for splitting long strings
/// into multiple windows (useful for text correction inference).
pub(super) fn add_submodule(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "windows")?;
    m.add_class::<InferenceWindow>()?;
    m.add_function(wrap_pyfunction!(char_windows_py, m)?)?;
    m.add_function(wrap_pyfunction!(byte_windows_py, m)?)?;
    parent_module.add_submodule(m)?;

    Ok(())
}
