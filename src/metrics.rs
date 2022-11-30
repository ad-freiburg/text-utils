use std::collections::HashSet;
use std::ops::Add;
use pyo3::prelude::*;
use rayon::prelude::*;
use crate::edit_distance::edit_distance;
use crate::utils::constrain;
use crate::whitespace::operations;

#[inline]
fn _mean_edit_distance(
    sequences: &[String],
    target_sequences: &[String],
    length: usize,
    use_graphemes: bool,
    normalize: bool,
) -> f64 {
    sequences
        .into_par_iter()
        .zip(target_sequences)
        .map(|(s, ts)| {
            let norm = if normalize {
                s.len().max(ts.len()).max(1) as f64
            } else {
                1.0
            };
            edit_distance(
                s,
                ts,
                use_graphemes,
                false,
                false,
            ) as f64 / norm
        })
        .sum::<f64>() / length.max(1) as f64
}

pub fn mean_edit_distance(
    sequences: &[String],
    target_sequences: &[String],
    use_graphemes: bool,
) -> f64 {
    assert_eq!(
        sequences.len(),
        target_sequences.len(),
        "expected the same number of sequences and target sequences"
    );
    _mean_edit_distance(
        sequences,
        target_sequences,
        sequences.len(),
        use_graphemes,
        false,
    )
}

#[pyfunction(
use_graphemes = "true"
)]
#[pyo3(name = "mean_edit_distance")]
fn mean_edit_distance_py(
    sequences: Vec<String>,
    target_sequences: Vec<String>,
    use_graphemes: bool,
) -> f64 {
    mean_edit_distance(&sequences, &target_sequences, use_graphemes)
}

pub fn mean_normalized_edit_distance(
    sequences: &[String],
    target_sequences: &[String],
    use_graphemes: bool,
) -> f64 {
    assert_eq!(
        sequences.len(),
        target_sequences.len(),
        "expected the same number of sequences and target sequences"
    );
    _mean_edit_distance(
        sequences,
        target_sequences,
        sequences.len(),
        use_graphemes,
        true,
    )
}

#[pyfunction(
use_graphemes = "true"
)]
#[pyo3(name = "mean_normalized_edit_distance")]
fn mean_normalized_edit_distance_py(
    sequences: Vec<String>,
    target_sequences: Vec<String>,
    use_graphemes: bool,
) -> f64 {
    mean_normalized_edit_distance(&sequences, &target_sequences, use_graphemes)
}

pub fn accuracy<T: Ord>(
    predictions: &[T],
    targets: &[T],
) -> f64 {
    assert_eq!(
        predictions.len(),
        targets.len(),
        "expected the same number of predictions and targets"
    );
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| {
            if p == t {
                1
            } else {
                0
            }
        })
        .sum::<usize>() as f64 / predictions.len().max(1) as f64
}

#[pyfunction]
#[pyo3(name = "accuracy")]
fn accuracy_py(
    predictions: Vec<&str>,
    targets: Vec<&str>,
) -> f64 {
    accuracy(&predictions, &targets)
}

struct TpFpFn {
    tp: usize,
    fp: usize,
    fn_: usize,
}

impl TpFpFn {
    fn new(tp: usize, fp: usize, fn_: usize) -> Self {
        TpFpFn { tp, fp, fn_ }
    }
    fn default() -> Self {
        TpFpFn { tp: 0, fp: 0, fn_: 0 }
    }

    fn f1(&self, mut beta: f64) -> F1PrecRec {
        let precision = self.tp as f64 / (self.tp + self.fp).max(1) as f64;
        let recall = self.tp as f64 / (self.tp + self.fn_).max(1) as f64;
        let f1 = if precision + recall > 0.0 {
            beta = constrain(beta, 0.0, 1.0);
            let beta_sq = beta.powi(2);
            ((1.0 + beta_sq) * precision * recall) / (beta_sq * precision + recall)
        } else {
            0.0
        };
        (f1, precision, recall)
    }
}

impl Add for TpFpFn {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        TpFpFn {
            tp: self.tp + rhs.tp,
            fp: self.fp + rhs.fp,
            fn_: self.fn_ + rhs.fn_,
        }
    }
}

pub type F1PrecRec = (f64, f64, f64);

#[inline]
fn _count_tp_fp_fn(
    a: &[bool],
    b: &[bool],
) -> TpFpFn {
    a
        .iter()
        .zip(b.iter())
        .fold(
            TpFpFn::default(),
            |tp_fp_fn, (p, t)| {
                match (p, t) {
                    (true, true) => TpFpFn { tp: tp_fp_fn.tp + 1, ..tp_fp_fn },
                    (true, false) => TpFpFn { fp: tp_fp_fn.fp + 1, ..tp_fp_fn },
                    (false, true) => TpFpFn { fn_: tp_fp_fn.fn_ + 1, ..tp_fp_fn },
                    _ => tp_fp_fn
                }
            })
}

pub fn binary_f1(
    predictions: &[bool],
    targets: &[bool],
    beta: f64,
) -> F1PrecRec {
    let tp_fp_fn = _count_tp_fp_fn(predictions, targets);
    tp_fp_fn.f1(beta)
}

#[pyfunction(
beta = "1.0"
)]
#[pyo3(name = "binary_f1")]
fn binary_f1_py(
    predictions: Vec<bool>,
    targets: Vec<bool>,
    beta: f64,
) -> F1PrecRec {
    binary_f1(&predictions, &targets, beta)
}

fn _spelling_correction_tp_fp_fn(
    input: &str,
    predicted: &str,
    target: &str,
    use_graphemes: bool,
) -> TpFpFn {
    TpFpFn::default()
}

pub fn spelling_correction_f1(
    input_sequences: &[String],
    predicted_sequences: &[String],
    target_sequences: &[String],
    beta: f64,
    sequence_averaged: bool,
    use_graphemes: bool,
) -> F1PrecRec {
    (0.0, 0.0, 0.0)
}

#[pyclass]
pub enum WhitespaceCorrectionMode {
    Insertions,
    Deletions,
    InsertionsAndDeletions,
}

#[inline]
fn _whitespace_ops_to_set(
    ops: &[u8],
    mode: &WhitespaceCorrectionMode,
) -> HashSet<(usize, u8)> {
    ops
        .iter()
        .enumerate()
        .filter_map(|(idx, op)| {
            match (*op, mode) {
                (0, _) => None,
                (1 | 2, WhitespaceCorrectionMode::InsertionsAndDeletions)
                | (1, WhitespaceCorrectionMode::Insertions)
                | (2, WhitespaceCorrectionMode::Deletions) => Some((idx, *op)),
                _ => panic!("should not happen, op should be either 0, 1, or 2, but got {op}")
            }
        })
        .collect()
}

#[inline]
fn _whitespace_correction_tp_fp_fn(
    input: &str,
    predicted: &str,
    target: &str,
    mode: &WhitespaceCorrectionMode,
    use_graphemes: bool,
) -> (TpFpFn, bool) {
    let gt_ops = operations(input, target, use_graphemes);
    let pred_ops = operations(input, predicted, use_graphemes);
    assert_eq!(gt_ops.len(), pred_ops.len());
    let gt_ops = _whitespace_ops_to_set(&gt_ops, mode);
    let pred_ops = _whitespace_ops_to_set(&pred_ops, mode);

    let tp_fp_fn = TpFpFn::new(
        gt_ops.intersection(&pred_ops).count(),
        pred_ops.difference(&gt_ops).count(),
        gt_ops.difference(&pred_ops).count()
    );
    (tp_fp_fn, gt_ops.is_empty() && pred_ops.is_empty())
}

pub fn whitespace_correction_f1(
    input_sequences: &[String],
    predicted_sequences: &[String],
    target_sequences: &[String],
    beta: f64,
    sequence_averaged: bool,
    mode: WhitespaceCorrectionMode,
    use_graphemes: bool,
) -> F1PrecRec {
    assert!(
        input_sequences.len() == target_sequences.len()
            && input_sequences.len() == predicted_sequences.len(),
        "expected the same number of input, target, and predicted sequences"
    );
    let values: Vec<(TpFpFn, bool)> = (0..input_sequences.len())
        .into_par_iter()
        .map(|idx| {
            let input = &input_sequences[idx];
            let predicted = &predicted_sequences[idx];
            let target = &target_sequences[idx];
            _whitespace_correction_tp_fp_fn(
                input,
                predicted,
                target,
                &mode,
                use_graphemes,
            )
        })
        .collect();
    if sequence_averaged {
        let (f1_sum, prec_sum, rec_sum) = values
            .into_iter()
            .map(|(tp_fp_fn, empty)| {
                if empty {
                    (1.0, 1.0, 1.0)
                } else {
                    tp_fp_fn.f1(beta)
                }
            })
            .fold(
                (0.0, 0.0, 0.0),
                |(f1_sum, prec_sum, rec_sum), (f1, prec, rec)| {
                (f1_sum + f1, prec_sum + prec, rec_sum + rec)
            });
        let num_values = input_sequences.len().max(1) as f64;
        (f1_sum / num_values, prec_sum / num_values, rec_sum / num_values)
    } else {
        let tp_fp_fn = values
            .into_iter()
            .fold(
                TpFpFn::default(),
                |total, (tp_fp_fn, _)| {
                total + tp_fp_fn
            });
        tp_fp_fn.f1(beta)
    }
}

/// A submodule containing functions to calculate various text correction metrics.
pub(super) fn add_submodule(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "metrics")?;
    m.add_function(wrap_pyfunction!(mean_edit_distance_py, m)?)?;
    m.add_function(wrap_pyfunction!(mean_normalized_edit_distance_py, m)?)?;
    m.add_function(wrap_pyfunction!(binary_f1_py, m)?)?;
    m.add_function(wrap_pyfunction!(accuracy_py, m)?)?;
    parent_module.add_submodule(m)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::metrics::{accuracy, binary_f1};

    #[test]
    fn test_binary_f1() {
        let (f1, prec, rec) = binary_f1(
            &[true, true, false, false],
            &[true, false, true, false],
            1.0,
        );
        assert!((f1 - 0.5).abs() < 1e-8);
        assert!((prec - 0.5).abs() < 1e-8);
        assert!((rec - 0.5).abs() < 1e-8);
    }

    #[test]
    fn test_accuracy() {
        let acc = accuracy(&[true, false, true], &[true, false, false]);
        assert!((acc - 0.666).abs() < 1e-3);
    }
}
