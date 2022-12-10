use crate::unicode::CS;
use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[derive(Debug, Clone)]
pub enum Result {
    SequenceClassification(Vec<usize>),
    SequenceGeneration(Vec<usize>),
    MultiSequenceGeneration(Vec<Vec<usize>>),
}

impl IntoPy<Py<PyDict>> for Result {
    fn into_py(self, py: Python<'_>) -> Py<PyDict> {
        let d = PyDict::new(py);
        d.into()
    }
}

impl<'a> FromPyObject<'a> for Result {
    fn extract(_: &'a PyAny) -> PyResult<Self> {
        Ok(Result::SequenceGeneration(vec![]))
    }
}

#[derive(Debug, Clone)]
pub struct Window<'a> {
    ctx_start: usize,
    ctx_end: usize,
    window_start: usize,
    window_end: usize,
    str: &'a str,
}

impl<'a> Window<'a> {
    pub fn new(
        ctx_start: usize,
        window_start: usize,
        window_end: usize,
        ctx_end: usize,
        str: &'a str,
    ) -> Self {
        Self {
            ctx_start,
            window_start,
            ctx_end,
            window_end,
            str,
        }
    }
}

impl IntoPy<PyObject> for Window<'_> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let window = PyList::new(
            py,
            &[
                self.ctx_start,
                self.window_start,
                self.window_end,
                self.ctx_end,
            ],
        );
        window.into()
    }
}

#[pyfunction(use_graphemes = "true")]
pub fn char_windows<'a>(
    s: &'a str,
    max_length: usize,
    context_length: usize,
    use_graphemes: bool,
) -> Vec<Window<'a>> {
    assert!(
        max_length > 2 * context_length,
        "max length must be larger than 2 times the context \
        length, otherwise there are no tokens left for the window itself"
    );
    let window_length = max_length - 2 * context_length;
    let cs = CS::new(s, use_graphemes);
    (0..cs.len())
        .step_by(window_length)
        .map(move |window_start| {
            let window_length = max_length - (1 + (window_start > 0) as usize) * context_length;
            let ctx_start = window_start.saturating_sub(context_length);
            let ctx_end = cs.len().min(window_start + window_length + context_length);
            let window_end = cs.len().min(window_start + window_length);
            Window::new(
                ctx_start,
                window_start,
                window_end,
                ctx_end,
                cs.sub(ctx_start, ctx_end),
            )
        })
        .collect()
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

#[pyfunction(use_graphemes = "true")]
pub fn byte_windows(
    s: &str,
    max_length: usize,
    context_length: usize,
    use_graphemes: bool,
) -> Vec<Window> {
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

        windows.push(Window::new(
            ctx_start,
            window_start,
            window_end,
            ctx_end,
            cs.sub(ctx_start, ctx_end),
        ));

        window_start = window_end;
    }
    windows
}

/// A submodule containing helper functions needed for splitting long strings
/// into multiple windows (useful for text correction inference).
pub(super) fn add_submodule(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "windows")?;
    m.add_function(wrap_pyfunction!(char_windows, m)?)?;
    m.add_function(wrap_pyfunction!(byte_windows, m)?)?;
    parent_module.add_submodule(m)?;

    Ok(())
}
