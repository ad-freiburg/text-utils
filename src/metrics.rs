use crate::edit;
use crate::edit::{distance, edited_words, EditOperation};
use crate::text::{clean, match_words, word_boundaries};
use crate::unicode::{normalize, Normalization, CS};
use crate::utils::{as_ref_slice_to_vec, py_invalid_type_error};
use crate::whitespace::{operations, WhitespaceOperation};
use anyhow::anyhow;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

#[inline]
fn _mean_edit_distance(
    sequences: &[impl AsRef<str>],
    target_sequences: &[impl AsRef<str>],
    length: usize,
    use_graphemes: bool,
    normalized: bool,
) -> anyhow::Result<f64> {
    let sequences = as_ref_slice_to_vec(sequences);
    let target_sequences = as_ref_slice_to_vec(target_sequences);
    if sequences.len() != target_sequences.len() {
        return Err(anyhow!(
            "different number of sequences and target sequences"
        ));
    }
    Ok(sequences
        .into_par_iter()
        .zip(target_sequences)
        .map(|(s, ts)| {
            distance(
                &normalize(&clean(s, true), Normalization::NFKC, true),
                &normalize(&clean(ts, true), Normalization::NFKC, true),
                use_graphemes,
                false,
                false,
                normalized,
            )
        })
        .sum::<f64>()
        / length.max(1) as f64)
}

pub fn mean_edit_distance(
    sequences: &[impl AsRef<str>],
    target_sequences: &[impl AsRef<str>],
    use_graphemes: bool,
) -> anyhow::Result<f64> {
    _mean_edit_distance(
        sequences,
        target_sequences,
        sequences.len(),
        use_graphemes,
        false,
    )
}

#[pyfunction(name = "mean_edit_distance", signature = (sequences, target_sequences, use_graphemes = true))]
fn mean_edit_distance_py(
    sequences: Vec<String>,
    target_sequences: Vec<String>,
    use_graphemes: bool,
) -> anyhow::Result<f64> {
    mean_edit_distance(&sequences, &target_sequences, use_graphemes)
}

pub fn mean_normalized_edit_distance(
    sequences: &[impl AsRef<str>],
    target_sequences: &[impl AsRef<str>],
    use_graphemes: bool,
) -> anyhow::Result<f64> {
    _mean_edit_distance(
        sequences,
        target_sequences,
        sequences.len(),
        use_graphemes,
        true,
    )
}

#[pyfunction(name = "mean_normalized_edit_distance", signature = (sequences, target_sequences, use_graphemes = true))]
fn mean_normalized_edit_distance_py(
    sequences: Vec<String>,
    target_sequences: Vec<String>,
    use_graphemes: bool,
) -> anyhow::Result<f64> {
    mean_normalized_edit_distance(&sequences, &target_sequences, use_graphemes)
}

pub fn accuracy<T: Ord>(predictions: &[T], targets: &[T]) -> anyhow::Result<f64> {
    if predictions.len() != targets.len() {
        return Err(anyhow!(
            "expected the same number of predictions and targets"
        ));
    }
    Ok(predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p == t) as usize)
        .sum::<usize>() as f64
        / predictions.len().max(1) as f64)
}

#[pyfunction(name = "accuracy", signature = (predictions, targets))]
fn accuracy_py(predictions: Vec<&str>, targets: Vec<&str>) -> anyhow::Result<f64> {
    let predictions: Vec<String> = predictions
        .into_iter()
        .map(|s| normalize(&clean(s, true), Normalization::NFKC, true))
        .collect();
    let targets: Vec<String> = targets
        .into_iter()
        .map(|s| normalize(&clean(s, true), Normalization::NFKC, true))
        .collect();
    accuracy(&predictions, &targets)
}

type F1Sets = (bool, HashSet<usize>, HashSet<usize>, HashSet<usize>);
pub struct TpFpFn {
    sets: Vec<F1Sets>,
}

impl IntoPy<PyObject> for TpFpFn {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.sets.into_py(py)
    }
}

impl TpFpFn {
    fn new(sets: Vec<F1Sets>) -> Self {
        TpFpFn { sets }
    }

    fn micro_f1(&self, beta: f64) -> F1PrecRec {
        let mut tps = 0;
        let mut fps = 0;
        let mut fns_ = 0;
        for (_, tp_, fp_, fn_) in self.sets.iter() {
            tps += tp_.len();
            fps += fp_.len();
            fns_ += fn_.len();
        }
        _f1(tps, fps, fns_, beta)
    }

    fn sequence_averaged_f1(&self, beta: f64) -> F1PrecRec {
        let (f1, precision, recall) = self
            .sets
            .iter()
            .map(|(empty, tp, fp, fn_)| {
                if *empty {
                    (1.0, 1.0, 1.0)
                } else {
                    _f1(tp.len(), fp.len(), fn_.len(), beta)
                }
            })
            .fold(
                (0.0, 0.0, 0.0),
                |(f1, precision, recall), (f1_, precision_, recall_)| {
                    (f1 + f1_, precision + precision_, recall + recall_)
                },
            );
        (
            f1 / self.sets.len().max(1) as f64,
            precision / self.sets.len().max(1) as f64,
            recall / self.sets.len().max(1) as f64,
        )
    }
}

pub type F1PrecRec = (f64, f64, f64);
#[inline]
fn _f1(tp: usize, fp: usize, fn_: usize, beta: f64) -> F1PrecRec {
    let precision = tp as f64 / (tp + fp).max(1) as f64;
    let recall = tp as f64 / (tp + fn_).max(1) as f64;
    let f1 = if precision + recall > 0.0 {
        let beta_sq = beta.powi(2);
        ((1.0 + beta_sq) * precision * recall) / (beta_sq * precision + recall)
    } else {
        0.0
    };
    (f1, precision, recall)
}

#[inline]
fn _count_tp_fp_fn(a: &[bool], b: &[bool]) -> (usize, usize, usize) {
    a.iter()
        .zip(b.iter())
        .fold((0, 0, 0), |(tp, fp, fn_), (p, t)| match (p, t) {
            (true, true) => (tp + 1, fp, fn_),
            (true, false) => (tp, fp + 1, fn_),
            (false, true) => (tp, fp, fn_ + 1),
            _ => (tp, fp, fn_),
        })
}

pub fn binary_f1(predictions: &[bool], targets: &[bool], beta: f64) -> anyhow::Result<F1PrecRec> {
    if predictions.len() != targets.len() {
        return Err(anyhow!("different number of predictions and targets"));
    }
    let (tp, fp, fn_) = _count_tp_fp_fn(predictions, targets);
    Ok(_f1(tp, fp, fn_, beta))
}

#[pyfunction(name = "binary_f1", signature = (predictions, targets, beta = 1.0))]
fn binary_f1_py(
    predictions: Vec<bool>,
    targets: Vec<bool>,
    beta: f64,
) -> anyhow::Result<F1PrecRec> {
    binary_f1(&predictions, &targets, beta)
}

fn _correction_f1(
    input_sequences: &[impl AsRef<str>],
    predicted_sequences: &[impl AsRef<str>],
    target_sequences: &[impl AsRef<str>],
    beta: f64,
    sequence_averaged: bool,
    f: impl Fn(&str, &str, &str) -> anyhow::Result<F1Sets> + Sync,
) -> anyhow::Result<(F1PrecRec, TpFpFn)> {
    let input_sequences = as_ref_slice_to_vec(input_sequences);
    let predicted_sequences = as_ref_slice_to_vec(predicted_sequences);
    let target_sequences = as_ref_slice_to_vec(target_sequences);
    if !(input_sequences.len() == target_sequences.len()
        && input_sequences.len() == predicted_sequences.len())
    {
        return Err(anyhow!(
            "expected the same number of input, target, and predicted sequences"
        ));
    };
    let sets = (0..input_sequences.len())
        .into_par_iter()
        .map(|idx| {
            let input = &input_sequences[idx];
            let predicted = &predicted_sequences[idx];
            let target = &target_sequences[idx];
            f(
                &normalize(&clean(input, true), Normalization::NFKC, true),
                &normalize(&clean(predicted, true), Normalization::NFKC, true),
                &normalize(&clean(target, true), Normalization::NFKC, true),
            )
        })
        .collect::<anyhow::Result<Vec<F1Sets>>>()?;
    let tp_fp_fn = TpFpFn::new(sets);
    if sequence_averaged {
        Ok((tp_fp_fn.sequence_averaged_f1(beta), tp_fp_fn))
    } else {
        Ok((tp_fp_fn.micro_f1(beta), tp_fp_fn))
    }
}

#[inline]
fn _group_words(
    input: &str,
    predicted: &str,
    matching_in_pred: &HashSet<usize>,
    use_graphemes: bool,
) -> HashSet<usize> {
    let input_cs = CS::new(input, use_graphemes);
    let pred_cs = CS::new(predicted, use_graphemes);
    let ops = edit::operations(input, predicted, use_graphemes, false, true);
    let input_words = word_boundaries(input, use_graphemes);
    let mut merged_with_next = HashSet::new();
    let mut num_whitespaces_inserted = HashMap::new();
    for (op, input_idx, pred_idx) in ops {
        let mut word_idx = 0;
        while word_idx < input_words.len() {
            if input_idx <= input_words[word_idx].1 {
                break;
            }
            word_idx += 1;
        }

        if op == EditOperation::Delete && input_cs.get_char(input_idx).is_whitespace() {
            merged_with_next.insert(word_idx);
        }

        if op == EditOperation::Insert && pred_cs.get_char(pred_idx).is_whitespace() {
            num_whitespaces_inserted.insert(
                word_idx,
                num_whitespaces_inserted
                    .get(&word_idx)
                    .copied()
                    .unwrap_or(0usize)
                    + 1,
            );
        }
    }

    let mut correct = HashSet::new();
    let mut input_idx = 0;
    let mut pred_idx = 0;
    while input_idx < input_words.len() {
        let mut merged_word = HashSet::from([input_idx]);
        let mut total_spaces_inserted = num_whitespaces_inserted
            .get(&input_idx)
            .copied()
            .unwrap_or(0);
        while merged_with_next.contains(&input_idx) {
            input_idx += 1;
            merged_word.insert(input_idx);
            total_spaces_inserted += num_whitespaces_inserted
                .get(&input_idx)
                .copied()
                .unwrap_or(0);
        }

        if (pred_idx..=pred_idx + total_spaces_inserted).all(|idx| matching_in_pred.contains(&idx))
        {
            for word_idx in merged_word {
                correct.insert(word_idx);
            }
        }

        input_idx += 1;
        pred_idx += total_spaces_inserted + 1;
    }
    assert!(
        input_idx == input_words.len()
            && pred_idx == word_boundaries(predicted, use_graphemes).len()
    );
    correct
}

#[inline]
fn _spelling_correction_tp_fp_fn(
    input: &str,
    predicted: &str,
    target: &str,
    use_graphemes: bool,
) -> F1Sets {
    let (_, misspelled) = edited_words(input, target);
    let (changed, _) = edited_words(input, predicted);
    let (matching_indices, _, _) = match_words(predicted, target, false);
    let matching_in_pred: HashSet<usize> = matching_indices
        .iter()
        .map(|(pred_idx, _)| *pred_idx)
        .collect();
    let restored: HashSet<usize> = matching_indices
        .iter()
        .map(|(_, tgt_idx)| *tgt_idx)
        .collect();
    let correct = _group_words(input, predicted, &matching_in_pred, use_graphemes);
    let tps = misspelled.intersection(&restored);
    let fps = changed.difference(&correct);
    let fns = misspelled.difference(&restored);
    (
        misspelled.is_empty() && changed.is_empty(),
        tps.copied().collect(),
        fps.copied().collect(),
        fns.copied().collect(),
    )
}

pub fn spelling_correction_f1(
    input_sequences: &[impl AsRef<str>],
    predicted_sequences: &[impl AsRef<str>],
    target_sequences: &[impl AsRef<str>],
    beta: f64,
    sequence_averaged: bool,
    use_graphemes: bool,
) -> anyhow::Result<(F1PrecRec, TpFpFn)> {
    let input_sequences = as_ref_slice_to_vec(input_sequences);
    let predicted_sequences = as_ref_slice_to_vec(predicted_sequences);
    let target_sequences = as_ref_slice_to_vec(target_sequences);
    _correction_f1(
        &input_sequences,
        &predicted_sequences,
        &target_sequences,
        beta,
        sequence_averaged,
        |input, predicted, target| {
            Ok(_spelling_correction_tp_fp_fn(
                input,
                predicted,
                target,
                use_graphemes,
            ))
        },
    )
}

#[pyfunction(
    name = "spelling_correction_f1", 
    signature = (input_sequences, predicted_sequences, target_sequences, beta=1.0, sequence_averaged=true, use_graphemes=true)
)]
fn spelling_correction_f1_py(
    input_sequences: Vec<String>,
    predicted_sequences: Vec<String>,
    target_sequences: Vec<String>,
    beta: f64,
    sequence_averaged: bool,
    use_graphemes: bool,
) -> anyhow::Result<(F1PrecRec, TpFpFn)> {
    spelling_correction_f1(
        &input_sequences,
        &predicted_sequences,
        &target_sequences,
        beta,
        sequence_averaged,
        use_graphemes,
    )
}

pub enum WhitespaceCorrectionMode {
    Insertions,
    Deletions,
    InsertionsAndDeletions,
}

impl<'a> FromPyObject<'a> for WhitespaceCorrectionMode {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let s: String = ob.extract()?;
        let mode = match s.as_str() {
            "insertions" => WhitespaceCorrectionMode::Insertions,
            "deletions" => WhitespaceCorrectionMode::Deletions,
            "insertion_and_deletions" => WhitespaceCorrectionMode::InsertionsAndDeletions,
            k => return Err(py_invalid_type_error(k, "whitespace correction mode")),
        };
        Ok(mode)
    }
}

#[inline]
fn _whitespace_ops_to_set(
    ops: &[WhitespaceOperation],
    mode: &WhitespaceCorrectionMode,
) -> HashSet<usize> {
    ops.iter()
        .enumerate()
        .filter_map(|(idx, op)| match (*op, mode) {
            (
                WhitespaceOperation::Insert | WhitespaceOperation::Delete,
                WhitespaceCorrectionMode::InsertionsAndDeletions,
            )
            | (WhitespaceOperation::Insert, WhitespaceCorrectionMode::Insertions)
            | (WhitespaceOperation::Delete, WhitespaceCorrectionMode::Deletions) => Some(idx),
            _ => None,
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
) -> anyhow::Result<F1Sets> {
    let gt_ops = operations(input, target, use_graphemes)?;
    let pred_ops = operations(input, predicted, use_graphemes)?;
    if gt_ops.len() != pred_ops.len() {
        return Err(anyhow!(
            "different number of operations from input to target than from input to prediction"
        ));
    }
    let gt_ops = _whitespace_ops_to_set(&gt_ops, mode);
    let pred_ops = _whitespace_ops_to_set(&pred_ops, mode);
    Ok((
        gt_ops.is_empty() && pred_ops.is_empty(),
        gt_ops.intersection(&pred_ops).copied().collect(),
        pred_ops.difference(&gt_ops).copied().collect(),
        gt_ops.difference(&pred_ops).copied().collect(),
    ))
}

pub fn whitespace_correction_f1(
    input_sequences: &[impl AsRef<str>],
    predicted_sequences: &[impl AsRef<str>],
    target_sequences: &[impl AsRef<str>],
    beta: f64,
    sequence_averaged: bool,
    mode: WhitespaceCorrectionMode,
    use_graphemes: bool,
) -> anyhow::Result<(F1PrecRec, TpFpFn)> {
    _correction_f1(
        input_sequences,
        predicted_sequences,
        target_sequences,
        beta,
        sequence_averaged,
        |input, predicted, target| {
            _whitespace_correction_tp_fp_fn(input, predicted, target, &mode, use_graphemes)
        },
    )
}

#[pyfunction(
    name = "whitespace_correction_f1", 
    signature = (input_sequences, predicted_sequences, target_sequences, beta=1.0, sequence_averaged=true, mode=WhitespaceCorrectionMode::InsertionsAndDeletions, use_graphemes=true)
)]
fn whitespace_correction_f1_py(
    input_sequences: Vec<String>,
    predicted_sequences: Vec<String>,
    target_sequences: Vec<String>,
    beta: f64,
    sequence_averaged: bool,
    mode: WhitespaceCorrectionMode,
    use_graphemes: bool,
) -> anyhow::Result<(F1PrecRec, TpFpFn)> {
    whitespace_correction_f1(
        &input_sequences,
        &predicted_sequences,
        &target_sequences,
        beta,
        sequence_averaged,
        mode,
        use_graphemes,
    )
}

/// A submodule containing functions to calculate various text correction metrics.
pub(super) fn add_submodule(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "metrics")?;
    m.add_function(wrap_pyfunction!(mean_edit_distance_py, m)?)?;
    m.add_function(wrap_pyfunction!(mean_normalized_edit_distance_py, m)?)?;
    m.add_function(wrap_pyfunction!(binary_f1_py, m)?)?;
    m.add_function(wrap_pyfunction!(accuracy_py, m)?)?;
    m.add_function(wrap_pyfunction!(spelling_correction_f1_py, m)?)?;
    m.add_function(wrap_pyfunction!(whitespace_correction_f1_py, m)?)?;
    parent_module.add_submodule(m)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::metrics::{
        accuracy, binary_f1, mean_edit_distance, mean_normalized_edit_distance,
        spelling_correction_f1, whitespace_correction_f1, WhitespaceCorrectionMode,
    };

    const EPS: f64 = 1e-8;

    #[test]
    fn test_binary_f1() {
        let (f1, prec, rec) = binary_f1(
            &[true, true, false, false],
            &[true, false, true, false],
            1.0,
        )
        .unwrap();
        assert!((f1 - 0.5).abs() < EPS);
        assert!((prec - 0.5).abs() < EPS);
        assert!((rec - 0.5).abs() < EPS);
    }

    const PRED_SEQUENCES: [&str; 4] = [
        "this is a test",
        "we do no match",
        "Just a wrong sequence",
        "one last examples",
    ];
    const TARGET_SEQUENCES: [&str; 4] = [
        "this is a test",
        "we do not match",
        "just a wrong sequence",
        "one last example",
    ];

    #[test]
    fn test_spelling_correction_f1() {
        // this test case is taken from Matthias Hertel's masters thesis
        // https://ad-publications.cs.uni-freiburg.de/theses/Master_Matthias_Hertel_2019.pdf
        let ipt = "Te cute cteats delicious fi sh.";
        let tgt = "The cute cat eats delicious fish.";
        let pred = "The cute act eats delicate fi sh.";

        let ((f1, prec, rec), _) =
            spelling_correction_f1(&[ipt], &[pred], &[tgt], 1.0, false, true).unwrap();
        assert!((f1 - 0.5).abs() < EPS);
        assert!((prec - 0.5).abs() < EPS);
        assert!((rec - 0.5).abs() < EPS);
    }

    #[test]
    fn test_mean_edit_distance() {
        let med = mean_edit_distance(&PRED_SEQUENCES, &TARGET_SEQUENCES, true).unwrap();
        assert!((med - (1.0 + 1.0 + 1.0 + 0.0) / 4.0).abs() < EPS);
    }

    #[test]
    fn test_mean_normalized_edit_distance() {
        let mned = mean_normalized_edit_distance(&PRED_SEQUENCES, &TARGET_SEQUENCES, true).unwrap();
        assert!((mned - (1.0 / 15.0 + 1.0 / 21.0 + 1.0 / 17.0 + 0.0) / 4.0).abs() < EPS);
    }

    #[test]
    fn test_accuracy() {
        let acc = accuracy(&[true, false, true], &[true, false, false]).unwrap();
        assert!((acc - 0.666).abs() < 1e-3);
        let acc = accuracy(&PRED_SEQUENCES, &TARGET_SEQUENCES).unwrap();
        assert!((acc - 0.25).abs() < EPS);
    }

    const WC_INPUT_SEQUENCES: [&str; 4] = [
        "thisisatest",
        "we do not ma tch",
        "just awrong seq uence",
        "o n e l a s t e x a m p l e",
    ];
    const WC_PRED_SEQUENCES: [&str; 4] = [
        "this is a test",
        "we do not match",
        "justa wrong sequence",
        "onelast example",
    ];
    const WC_TARGET_SEQUENCES: [&str; 4] = [
        "this is a test",
        "we do not match",
        "just a wrong sequence",
        "one last example",
    ];

    #[test]
    fn test_whitespace_correction_f1() {
        let ((f1, prec, rec), _) = whitespace_correction_f1(
            &WC_INPUT_SEQUENCES,
            &WC_PRED_SEQUENCES,
            &WC_TARGET_SEQUENCES,
            1.0,
            false,
            WhitespaceCorrectionMode::InsertionsAndDeletions,
            true,
        )
        .unwrap();
        assert!((f1 - 0.9444).abs() < 1e-4);
        assert!((prec - (3.0 + 1.0 + 2.0 + 11.0) / (3.0 + 1.0 + 3.0 + 12.0)).abs() < EPS);
        assert!((rec - 1.0).abs() < EPS);
        let ((f1, prec, rec), _) = whitespace_correction_f1(
            &WC_INPUT_SEQUENCES,
            &WC_PRED_SEQUENCES,
            &WC_TARGET_SEQUENCES,
            1.0,
            true,
            WhitespaceCorrectionMode::InsertionsAndDeletions,
            true,
        )
        .unwrap();
        assert!((f1 - 0.9391).abs() < 1e-4);
        assert!((prec - 0.8958).abs() < 1e-4);
        assert!((rec - 1.0).abs() < EPS);
    }
}
