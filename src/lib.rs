extern crate core;

use pyo3::prelude::*;

pub mod corrupt;
pub mod data;
pub mod dictionary;
pub mod edit;
pub mod metrics;
pub mod text;
pub mod tokenization;
pub mod unicode;
pub mod utils;
pub mod whitespace;
pub mod windows;

#[pymodule]
fn _internal(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    // add submodules
    edit::add_submodule(py, m)?;
    text::add_submodule(py, m)?;
    tokenization::add_submodule(py, m)?;
    dictionary::add_submodule(py, m)?;
    data::add_submodule(py, m)?;
    whitespace::add_submodule(py, m)?;
    windows::add_submodule(py, m)?;
    metrics::add_submodule(py, m)?;
    unicode::add_submodule(py, m)?;

    Ok(())
}
