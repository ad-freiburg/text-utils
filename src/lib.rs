extern crate core;

use pyo3::prelude::*;

pub mod edit_distance;
pub mod text;
pub mod data;
pub mod tokenization;
pub mod whitespace;
pub mod utils;
pub mod unicode;
pub mod inference;

#[pymodule]
fn _internal(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let _ = pyo3_log::try_init();

    // add submodules
    edit_distance::add_submodule(py, m)?;
    text::add_submodule(py, m)?;
    tokenization::add_submodule(py, m)?;
    data::add_submodule(py, m)?;
    whitespace::add_submodule(py, m)?;
    inference::add_submodule(py, m)?;

    Ok(())
}
