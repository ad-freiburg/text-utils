extern crate core;

use pyo3::prelude::*;
use pyo3::types::PyDict;

pub mod data;
pub mod edit_distance;
pub mod windows;
pub mod text;
pub mod tokenization;
pub mod unicode;
pub mod utils;
pub mod whitespace;

#[pymodule]
fn _internal(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let _ = pyo3_log::try_init();

    // add submodules
    edit_distance::add_submodule(py, m)?;
    text::add_submodule(py, m)?;
    tokenization::add_submodule(py, m)?;
    data::add_submodule(py, m)?;
    whitespace::add_submodule(py, m)?;
    windows::add_submodule(py, m)?;

    Ok(())
}
