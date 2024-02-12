extern crate core;

use pyo3::prelude::*;

pub mod constraints;
pub mod continuations;
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
fn _internal(py: Python<'_>, m: &PyModule) -> PyResult<()> {
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
    corrupt::add_submodule(py, m)?;
    continuations::add_submodule(py, m)?;
    constraints::add_submodule(py, m)?;

    Ok(())
}
