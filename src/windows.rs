use crate::unicode::CS;
use crate::utils::{py_invalid_type_error, py_required_key_error};
use anyhow::anyhow;
use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyDict;

#[derive(Debug, Clone)]
pub struct Window<'s> {
    ctx_start: usize,
    ctx_end: usize,
    window_start: usize,
    window_end: usize,
    byte_ctx_start: usize,
    byte_ctx_end: usize,
    byte_window_start: usize,
    byte_window_end: usize,
    pub str: &'s str,
}

impl Window<'_> {
    pub fn boundaries(&self) -> (usize, usize, usize, usize) {
        (
            self.ctx_start,
            self.window_start,
            self.window_end,
            self.ctx_end,
        )
    }

    pub fn byte_boundaries(&self) -> (usize, usize, usize, usize) {
        (
            self.byte_ctx_start,
            self.byte_window_start,
            self.byte_window_end,
            self.byte_ctx_end,
        )
    }
}

#[derive(Debug, Clone)]
pub enum WindowConfig {
    Character(usize, usize, bool),
    Bytes(usize, usize, bool),
    Full(bool),
}

impl<'a> FromPyObject<'a> for WindowConfig {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        let d: &Bound<'_, PyDict> = ob.downcast()?;
        let Some(window_type) = d.get_item("type")? else {
            return Err(py_required_key_error("type", "window config"));
        };
        let window_type: String = window_type.extract()?;
        let window_config = match window_type.as_str() {
            "character" => {
                let Some(max_chars) = d.get_item("max_chars")? else {
                    return Err(py_required_key_error("max_chars", "window config"));
                };
                let Some(context_chars) = d.get_item("context_chars")? else {
                    return Err(py_required_key_error("context_chars", "window config"));
                };
                let use_graphemes: bool = if let Some(value) = d.get_item("use_graphemes")? {
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
                let Some(max_bytes) = d.get_item("max_bytes")? else {
                    return Err(py_required_key_error("max_bytes", "window config"));
                };
                let Some(context_bytes) = d.get_item("context_bytes")? else {
                    return Err(py_required_key_error("context_bytes", "window config"));
                };
                let use_graphemes: bool = if let Some(value) = d.get_item("use_graphemes")? {
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
            "full" => {
                let use_graphemes: bool = if let Some(value) = d.get_item("use_graphemes")? {
                    value.extract()?
                } else {
                    true
                };
                WindowConfig::Full(use_graphemes)
            }
            k => return Err(py_invalid_type_error(k, "window")),
        };
        Ok(window_config)
    }
}

pub fn windows<'a>(s: &'a str, config: &WindowConfig) -> anyhow::Result<Vec<Window<'a>>> {
    if s.is_empty() {
        return Ok(vec![Window {
            ctx_start: 0,
            window_start: 0,
            window_end: 0,
            ctx_end: 0,
            byte_ctx_start: 0,
            byte_window_start: 0,
            byte_window_end: 0,
            byte_ctx_end: 0,
            str: s,
        }]);
    }
    match *config {
        WindowConfig::Character(max_chars, context_chars, use_graphemes) => {
            char(s, max_chars, context_chars, use_graphemes)
        }
        WindowConfig::Bytes(max_bytes, context_bytes, use_graphemes) => {
            byte(s, max_bytes, context_bytes, use_graphemes)
        }
        WindowConfig::Full(use_graphemes) => {
            let cs = CS::new(s, use_graphemes);
            Ok(vec![Window {
                ctx_start: 0,
                window_start: 0,
                window_end: cs.len(),
                ctx_end: cs.len(),
                byte_ctx_start: 0,
                byte_window_start: 0,
                byte_window_end: s.len(),
                byte_ctx_end: s.len(),
                str: s,
            }])
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

pub fn char(
    s: &str,
    max_length: usize,
    context_length: usize,
    use_graphemes: bool,
) -> anyhow::Result<Vec<Window>> {
    if max_length <= 2 * context_length {
        return Err(anyhow!(
            "max length must be larger than 2 times the context \
        length, otherwise there are no characters left for the window itself"
        ));
    }
    let cs = CS::new(s, use_graphemes);
    let mut window_start = 0;
    let mut windows = vec![];
    while window_start < cs.len() {
        let window_length = max_length - (1 + usize::from(window_start > 0)) * context_length;
        let ctx_start = window_start.saturating_sub(context_length);
        let ctx_end = cs.len().min(window_start + window_length + context_length);
        let window_end = cs.len().min(window_start + window_length);
        let byte_ctx = cs.char_range_to_byte_range(ctx_start, ctx_end);
        let byte_window = cs.char_range_to_byte_range(window_start, window_end);
        windows.push(Window {
            ctx_start,
            window_start,
            window_end,
            ctx_end,
            byte_ctx_start: byte_ctx.0,
            byte_window_start: byte_window.0,
            byte_window_end: byte_window.1,
            byte_ctx_end: byte_ctx.1,
            str: cs.sub(ctx_start, ctx_end),
        });

        window_start = window_end;
    }
    Ok(windows)
}

#[pyfunction(name = "char_windows", signature = (s, max_length, context_length, use_graphemes = true))]
pub fn char_py(
    s: String,
    max_length: usize,
    context_length: usize,
    use_graphemes: bool,
) -> anyhow::Result<Vec<PyWindow>> {
    Ok(char(&s, max_length, context_length, use_graphemes)?
        .into_iter()
        .map(PyWindow::from)
        .collect())
}

#[inline]
fn count_until(mut iter: impl Iterator<Item = usize>, max_length: usize, cs: &CS) -> usize {
    iter.fold_while((0usize, 0usize), |(count, acc), idx| {
        let next_acc = acc + cs.char_byte_len(idx);
        if next_acc > max_length {
            Done((count, acc))
        } else {
            Continue((count + 1, next_acc))
        }
    })
    .into_inner()
    .0
}

pub fn byte(
    s: &str,
    max_bytes: usize,
    context_bytes: usize,
    use_graphemes: bool,
) -> anyhow::Result<Vec<Window>> {
    if max_bytes <= 2 * context_bytes {
        return Err(anyhow!(
            "max bytes must be larger than 2 times the context \
        bytes, otherwise there are no bytes left for the window itself"
        ));
    }
    let cs = CS::new(s, use_graphemes);

    let mut windows = vec![];
    let mut window_start = 0;
    while window_start < cs.len() {
        let window_length = max_bytes - (1 + usize::from(window_start > 0)) * context_bytes;
        let window_end = window_start + count_until(window_start..cs.len(), window_length, &cs);
        if window_end <= window_start {
            return Err(anyhow!(
                "single character in '{s}' at position {window_start} has more bytes \
                ({}) than the window length ({window_length}), \
                this suggests that something with your input string is wrong or the window length \
                is too small",
                cs.char_byte_len(window_start)
            ));
        }
        let ctx_start =
            window_start.saturating_sub(count_until((0..window_start).rev(), context_bytes, &cs));
        let ctx_end = window_end + count_until(window_end..cs.len(), context_bytes, &cs);
        let byte_ctx = cs.char_range_to_byte_range(ctx_start, ctx_end);
        let byte_window = cs.char_range_to_byte_range(window_start, window_end);

        windows.push(Window {
            ctx_start,
            window_start,
            window_end,
            ctx_end,
            byte_ctx_start: byte_ctx.0,
            byte_window_start: byte_window.0,
            byte_window_end: byte_window.1,
            byte_ctx_end: byte_ctx.1,
            str: cs.sub(ctx_start, ctx_end),
        });

        window_start = window_end;
    }
    Ok(windows)
}

#[allow(clippy::needless_pass_by_value)]
#[pyfunction(name = "byte_windows", signature = (s, max_bytes, context_bytes, use_graphemes = true))]
pub fn byte_py(
    s: PyBackedStr,
    max_bytes: usize,
    context_bytes: usize,
    use_graphemes: bool,
) -> anyhow::Result<Vec<PyWindow>> {
    Ok(byte(s.as_ref(), max_bytes, context_bytes, use_graphemes)?
        .into_iter()
        .map(PyWindow::from)
        .collect())
}

/// A submodule containing helper functions needed for splitting long strings
/// into multiple windows (useful for text correction inference).
pub(super) fn add_submodule(py: Python<'_>, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "windows")?;
    m.add_function(wrap_pyfunction!(byte_py, m.clone())?)?;
    m.add_function(wrap_pyfunction!(char_py, m.clone())?)?;
    parent_module.add_submodule(&m)?;

    Ok(())
}
