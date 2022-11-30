extern crate core;

use pyo3::prelude::*;
use pyo3::types::PyDict;

pub mod data;
pub mod edit_distance;
pub mod inference;
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
    inference::add_submodule(py, m)?;

    // Inserting to sys.modules allows importing submodules nicely from Python
    // e.g. from text_correction_utils.tokenization import DataLoader
    let sys = PyModule::import(py, "sys")?;
    let sys_modules: &PyDict = sys.getattr("modules")?.downcast()?;
    sys_modules.set_item("_internal.edit_distance", m.getattr("edit_distance")?)?;
    sys_modules.set_item("_internal.text", m.getattr("text")?)?;
    sys_modules.set_item("_internal.tokenization", m.getattr("tokenization")?)?;
    sys_modules.set_item("_internal.data", m.getattr("data")?)?;
    sys_modules.set_item("_internal.whitespace", m.getattr("whitespace")?)?;
    sys_modules.set_item("_internal.inference", m.getattr("inference")?)?;

    Ok(())
}
