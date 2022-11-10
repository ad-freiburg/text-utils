use pyo3::prelude::*;

#[pyfunction]
fn test() -> PyResult<String> {
    Ok("test".into())
}

#[pymodule]
fn _internal(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test, m)?)?;

    Ok(())
}
