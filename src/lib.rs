use pyo3::prelude::*;

pub mod data;
pub mod edit;
pub mod windows;
pub mod text;
pub mod tokenization;
pub mod unicode;
pub mod utils;
pub mod whitespace;
pub mod metrics;

#[pymodule]
fn _internal(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let _ = pyo3_log::try_init();

    // add submodules
    edit::add_submodule(py, m)?;
    text::add_submodule(py, m)?;
    tokenization::add_submodule(py, m)?;
    data::add_submodule(py, m)?;
    whitespace::add_submodule(py, m)?;
    windows::add_submodule(py, m)?;
    metrics::add_submodule(py, m)?;

    Ok(())
}
