use crate::unicode::CS;
use crate::utils::{py_invalid_type_error, py_required_key_error};
use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
    pub ctx_start: usize,
    ctx_end: usize,
    window_start: usize,
    window_end: usize,
    pub str: &'a str,
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

    pub fn boundaries(&self) -> (usize, usize, usize, usize) {
        (
            self.ctx_start,
            self.window_start,
            self.window_end,
            self.ctx_end,
        )
    }
}

#[derive(Debug, Clone)]
pub enum WindowConfig {
    Character(usize, usize, bool),
    Bytes(usize, usize, bool),
}

impl<'a> FromPyObject<'a> for WindowConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let Some(window_type) = d.get_item("type") else {
            return Err(py_required_key_error("type", "window config"));
        };
        let window_type: String = window_type.extract()?;
        let window_config = match window_type.as_str() {
            "character" => {
                let Some(max_chars) = d.get_item("max_chars") else {
                    return Err(py_required_key_error("max_chars", "window config"));
                };
                let Some(context_chars) = d.get_item("context_chars") else {
                    return Err(py_required_key_error("context_chars", "window config"));
                };
                let use_graphemes: bool = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };
                WindowConfig::Character(
                    max_chars.extract()?,
                    context_chars.extract()?,
                    use_graphemes,
                )
            }
            "byte" => {
                let Some(max_bytes) = d.get_item("max_bytes") else {
                    return Err(py_required_key_error("max_bytes", "window config"));
                };
                let Some(context_bytes) = d.get_item("context_bytes") else {
                    return Err(py_required_key_error("context_bytes", "window config"));
                };
                let use_graphemes: bool = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };
                WindowConfig::Bytes(
                    max_bytes.extract()?,
                    context_bytes.extract()?,
                    use_graphemes,
                )
            }
            k => return Err(py_invalid_type_error(k, "window")),
        };
        Ok(window_config)
    }
}

pub fn windows<'a>(s: &'a str, config: &WindowConfig) -> Vec<Window<'a>> {
    match *config {
        WindowConfig::Character(max_chars, context_chars, use_graphemes) => {
            char_windows(s, max_chars, context_chars, use_graphemes)
        }
        WindowConfig::Bytes(max_bytes, context_bytes, use_graphemes) => {
            byte_windows(s, max_bytes, context_bytes, use_graphemes)
        }
    }
}

#[pyclass]
#[pyo3(name = "Window")]
pub struct PyWindow {
    #[pyo3(get)]
    ctx_start: usize,
    #[pyo3(get)]
    window_start: usize,
    #[pyo3(get)]
    window_end: usize,
    #[pyo3(get)]
    ctx_end: usize,
    #[pyo3(get)]
    str: String,
}

impl<'a> From<Window<'a>> for PyWindow {
    fn from(w: Window<'a>) -> Self {
        PyWindow {
            ctx_start: w.ctx_start,
            window_start: w.window_start,
            window_end: w.window_end,
            ctx_end: w.ctx_end,
            str: w.str.to_string(),
        }
    }
}

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
        .map(|window_start| {
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

#[pyfunction(use_graphemes = "true")]
#[pyo3(name = "char_windows")]
pub fn char_windows_py(
    s: String,
    max_length: usize,
    context_length: usize,
    use_graphemes: bool,
) -> Vec<PyWindow> {
    char_windows(&s, max_length, context_length, use_graphemes)
        .into_iter()
        .map(PyWindow::from)
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

pub fn byte_windows<'a>(
    s: &'a str,
    max_bytes: usize,
    context_bytes: usize,
    use_graphemes: bool,
) -> Vec<Window<'a>> {
    assert!(
        max_bytes > 2 * context_bytes,
        "max length must be larger than 2 times the context \
        length, otherwise there are no tokens left for the window itself"
    );
    let cs = CS::new(s, use_graphemes);

    let mut windows = vec![];
    let mut window_start = 0;
    while window_start < cs.len() {
        let window_length = max_bytes - (1 + (window_start > 0) as usize) * context_bytes;
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
            window_start.saturating_sub(count_until((0..window_start).rev(), context_bytes, &cs));
        let ctx_end = window_end + count_until(window_end..cs.len(), context_bytes, &cs);

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

#[pyfunction(use_graphemes = "true")]
#[pyo3(name = "byte_windows")]
pub fn byte_windows_py(
    s: String,
    max_bytes: usize,
    context_bytes: usize,
    use_graphemes: bool,
) -> Vec<PyWindow> {
    byte_windows(&s, max_bytes, context_bytes, use_graphemes)
        .into_iter()
        .map(PyWindow::from)
        .collect()
}

/// A submodule containing helper functions needed for splitting long strings
/// into multiple windows (useful for text correction inference).
pub(super) fn add_submodule(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "windows")?;
    m.add_function(wrap_pyfunction!(byte_windows_py, m)?)?;
    m.add_function(wrap_pyfunction!(char_windows_py, m)?)?;
    parent_module.add_submodule(m)?;

    Ok(())
}
