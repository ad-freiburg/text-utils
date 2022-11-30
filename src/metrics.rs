use pyo3::prelude::*;
use crate::edit_distance::edit_distance;
use crate::utils::constrain;

#[inline]
fn _mean_edit_distance(
    sequences: &[impl AsRef<str>],
    target_sequences: &[impl AsRef<str>],
    use_graphemes: bool,
    normalize: bool,
) -> f64 {
    assert_eq!(
        sequences.len(),
        target_sequences.len(),
        "expected the same number of sequences and target sequences"
    );
    sequences
        .iter()
        .zip(target_sequences.iter())
        .map(|(s, ts)| {
            let norm = if normalize {
                s.as_ref().len().max(ts.as_ref().len()).max(1) as f64
            } else {
                1.0
            };
            edit_distance(
                s.as_ref(),
                ts.as_ref(),
                use_graphemes,
                false,
                false,
            ) as f64 / norm
        })
        .sum::<f64>() / sequences.len().max(1) as f64
}

pub fn mean_edit_distance(
    sequences: &[impl AsRef<str>],
    target_sequences: &[impl AsRef<str>],
    use_graphemes: bool,
) -> f64 {
    _mean_edit_distance(sequences, target_sequences, use_graphemes, false)
}

#[pyfunction(
use_graphemes = "true"
)]
#[pyo3(name = "mean_edit_distance")]
fn mean_edit_distance_py(
    sequences: Vec<&str>,
    target_sequences: Vec<&str>,
    use_graphemes: bool,
) -> f64 {
    mean_edit_distance(&sequences, &target_sequences, use_graphemes)
}

pub fn mean_normalized_edit_distance(
    sequences: &[impl AsRef<str>],
    target_sequences: &[impl AsRef<str>],
    use_graphemes: bool,
) -> f64 {
    _mean_edit_distance(sequences, target_sequences, use_graphemes, true)
}

#[pyfunction(
use_graphemes = "true"
)]
#[pyo3(name = "mean_normalized_edit_distance")]
fn mean_normalized_edit_distance_py(
    sequences: Vec<&str>,
    target_sequences: Vec<&str>,
    use_graphemes: bool,
) -> f64 {
    mean_normalized_edit_distance(&sequences, &target_sequences, use_graphemes)
}

pub fn accuracy(
    sequences: &[impl AsRef<str>],
    target_sequences: &[impl AsRef<str>],
) -> f64 {
    assert_eq!(
        sequences.len(),
        target_sequences.len(),
        "expected the same number of sequences and target sequences"
    );
    sequences
        .iter()
        .zip(target_sequences.iter())
        .map(|(s, ts)| {
            (s.as_ref() == ts.as_ref()) as usize
        })
        .sum::<usize>() as f64 / sequences.len().max(1) as f64
}

#[inline]
fn _f1_precision_recall(
    true_pos: usize,
    false_pos: usize,
    false_neg: usize,
    mut beta: f64,
) -> (f64, f64, f64) {
    let precision = true_pos as f64 / (true_pos + false_pos).max(1) as f64;
    let recall = true_pos as f64 / (true_pos + false_neg).max(1) as f64;
    let f1 = if precision + recall > 0.0 {
        beta = constrain(beta, 0.0, 1.0);
        let beta_sq = beta.powi(2);
        ((1.0 + beta_sq) * precision * recall) / (beta_sq * precision + recall)
    } else {
        0.0
    };
    (f1, precision, recall)
}

#[inline]
fn _count_tp_fp_fn(
    a: &[bool],
    b: &[bool],
) -> (usize, usize, usize) {
    a
        .iter()
        .zip(b.iter())
        .fold(
            (0, 0, 0),
            |(mut true_pos, mut false_pos, mut false_neg), (p, t)| {
                match (p, t) {
                    (true, true) => true_pos += 1,
                    (true, false) => false_pos += 1,
                    (false, true) => false_neg += 1,
                    _ => ()
                };
                (true_pos, false_pos, false_neg)
            })
}

pub fn binary_f1(
    predictions: &[bool],
    targets: &[bool],
    beta: f64,
) -> (f64, f64, f64) {
    let (true_pos, false_pos, false_neg) = _count_tp_fp_fn(predictions, targets);
    _f1_precision_recall(true_pos, false_pos, false_neg, beta)
}

#[pyfunction(
beta = "1.0"
)]
#[pyo3(name = "binary_f1")]
fn binary_f1_py(
    predictions: Vec<bool>,
    targets: Vec<bool>,
    beta: f64,
) -> (f64, f64, f64) {
    binary_f1(&predictions, &targets, beta)
}

/// A submodule containing functions to calculate various text correction metrics.
pub(super) fn add_submodule(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "metrics")?;
    m.add_function(wrap_pyfunction!(mean_edit_distance_py, m)?)?;
    m.add_function(wrap_pyfunction!(mean_normalized_edit_distance_py, m)?)?;
    m.add_function(wrap_pyfunction!(binary_f1_py, m)?)?;
    parent_module.add_submodule(m)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::metrics::binary_f1;

    #[test]
    fn test_binary_f1() {
        // binary_f1(&[12, 12], &[12, 12], 1.0);
    }
}
