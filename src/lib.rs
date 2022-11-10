use pyo3::prelude::*;
use pyo3::types::*;

mod edit_distance;
mod utils;

#[pyfunction]
fn word_boundaries(s: String) -> PyResult<Vec<(u64, u64)>> {
    let mut boundaries = vec![];
    let mut running_len = 0u64;
    for word in s.split(" ") {
        let len = word.len() as u64;
        boundaries.push((running_len, running_len + len));
        running_len += len + 1;
    }
    Ok(boundaries)
}

#[pymodule]
fn _internal(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    let _ = pyo3_log::try_init();

    // add edit distance functions
    edit_distance::add_functions(m)?;

    m.add_function(wrap_pyfunction!(word_boundaries, m)?)?;

    Ok(())
}
