use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fmt::Display;
use std::fs::File;
use std::io::{BufReader, Write};
use std::ops::Add;
use std::path::Path;

pub trait SerializeMsgPack
where
    Self: Sized + Serialize + DeserializeOwned,
{
    fn save(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let mut file = File::create(path)?;
        let buf = rmp_serde::to_vec(self)?;
        file.write_all(&buf)?;
        Ok(())
    }

    fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let deserialized = rmp_serde::from_read(reader)?;
        Ok(deserialized)
    }
}

impl<T> SerializeMsgPack for T where T: Sized + Serialize + DeserializeOwned {}

pub(crate) type Matrix<T> = Vec<Vec<T>>;

pub fn progress_bar(msg: &str, size: u64, hidden: bool) -> ProgressBar {
    let pb = ProgressBar::new(size)
        .with_style(
            ProgressStyle::with_template(
                "{msg}: {wide_bar} [{pos}/{len}] [{elapsed_precise}|{eta_precise}]",
            )
            .unwrap(),
        )
        .with_message(msg.to_string());
    if hidden {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }
    pb
}

#[inline]
pub(crate) fn accumulate_with<T, F>(values: &[T], acc_fn: F) -> Vec<T>
where
    T: Clone,
    F: Fn(&T, &T) -> T,
{
    if values.is_empty() {
        return vec![];
    }
    let mut total_v = values.first().unwrap().clone();
    let mut cum_values = vec![total_v.clone()];
    for v in &values[1..] {
        total_v = acc_fn(&total_v, v);
        cum_values.push(total_v.clone());
    }
    cum_values
}

#[cfg(feature = "benchmark-utils")]
#[inline]
pub fn accumulate_with_pub<T, F>(values: &[T], acc_fn: F) -> Vec<T>
where
    T: Clone,
    F: Fn(&T, &T) -> T,
{
    accumulate_with(values, acc_fn)
}

#[inline]
pub(crate) fn accumulate<T>(values: &[T]) -> Vec<T>
where
    T: Add<T, Output = T> + Clone,
{
    accumulate_with(values, |acc, v| acc.clone() + v.clone())
}

#[cfg(feature = "benchmark-utils")]
#[inline]
pub fn accumulate_pub<T>(values: &[T]) -> Vec<T>
where
    T: Add<T, Output = T> + Clone,
{
    accumulate(values)
}

#[inline]
pub(crate) fn run_length_encode<T>(values: &[T]) -> Vec<(T, usize)>
where
    T: PartialEq + Clone,
{
    if values.is_empty() {
        return vec![];
    }
    let mut rle = vec![];
    let mut val = values.first().unwrap();
    let mut count = 1;
    for v in &values[1..] {
        if v == val {
            count += 1;
        } else {
            rle.push((val.clone(), count));
            val = v;
            count = 1;
        }
    }
    rle.push((val.clone(), count));
    rle
}

#[cfg(feature = "benchmark-utils")]
#[inline]
pub fn run_length_encode_pub<T>(values: &[T]) -> Vec<(T, usize)>
where
    T: PartialEq + Clone,
{
    run_length_encode(values)
}

#[inline]
pub(crate) fn run_length_decode<T>(values: &[(T, usize)]) -> Vec<T>
where
    T: Clone,
{
    let mut decoded = vec![];
    for (v, count) in values {
        for _ in 0..*count {
            decoded.push(v.clone())
        }
    }
    decoded
}

#[cfg(feature = "benchmark-utils")]
#[inline]
pub fn run_length_decode_pub<T>(values: &[(T, usize)]) -> Vec<T>
where
    T: Clone,
{
    run_length_decode(values)
}

#[inline]
pub fn find_subsequences_of_max_size_k<T, SeqSize>(
    values: &[T],
    k: usize,
    size_fn: SeqSize,
) -> Vec<(usize, usize)>
where
    SeqSize: Fn(&[T]) -> usize,
{
    // fast forward to first valid starting element
    let mut start = 0;
    while start < values.len() && size_fn(&values[start..start + 1]) > k {
        start += 1;
    }
    if start >= values.len() {
        return vec![];
    }
    let mut end = start + 1;
    let mut prev_subsequence_size = size_fn(&values[start..end]);
    let mut subsequences: Vec<(usize, usize)> = vec![];
    while start < values.len() && end <= values.len() {
        let subsequence_size = size_fn(&values[start..end]);
        match (prev_subsequence_size <= k, subsequence_size <= k) {
            (_, true) => {
                if end >= values.len() {
                    subsequences.push((start, end));
                }
                end += 1
            }
            (true, false) => {
                subsequences.push((start, end - 1));
                start += 1;
            }
            (false, false) => {
                start += 1;
                end = end.max(start + 1);
            }
        };
        prev_subsequence_size = subsequence_size;
    }
    subsequences
}

#[inline]
pub(crate) fn as_ref_slice_to_vec(values: &[impl AsRef<str>]) -> Vec<&str> {
    values.iter().map(|s| s.as_ref()).collect()
}

pub(crate) fn py_required_key_error(key_name: impl Display, value_name: impl Display) -> PyErr {
    PyTypeError::new_err(format!(
        "could not find required key \"{key_name}\" for \
            {value_name}"
    ))
}

pub(crate) fn py_invalid_type_error(name: impl Display, type_name: impl Display) -> PyErr {
    PyTypeError::new_err(format!("\"{name}\" is not a valid {type_name} type"))
}

#[cfg(test)]
mod tests {
    use crate::utils::{
        accumulate, find_subsequences_of_max_size_k, run_length_decode, run_length_encode,
    };

    #[test]
    fn test_accumulate() {
        let accum = accumulate(&vec![1, 4, 4, 2]);
        assert_eq!(accum, vec![1, 5, 9, 11]);
        let accum = accumulate(&vec![0.5, -0.5, 2.0, 3.5]);
        assert_eq!(accum, vec![0.5, 0.0, 2.0, 5.5]);
        let empty: Vec<i32> = Vec::new();
        assert_eq!(accumulate(&empty), empty);
    }

    #[test]
    fn test_run_length_encoding() {
        let values = vec![1, 1, 1, 2, 2, 1, 4, 4, 5];
        let enc = run_length_encode(&values);
        assert_eq!(enc, vec![(1, 3), (2, 2), (1, 1), (4, 2), (5, 1)]);
        assert_eq!(values, run_length_decode(&enc));
    }

    #[test]
    fn test_find_subsequences_of_max_size_k() {
        let sum_fn = |items: &[usize]| items.iter().sum::<usize>();
        let mut values: Vec<usize> = vec![14, 50, 80, 100];
        let sub_seqs = find_subsequences_of_max_size_k(&values, 32, sum_fn);
        assert_eq!(sub_seqs, vec![(0, 1)]);
        let sub_seqs = find_subsequences_of_max_size_k(&values, 13, sum_fn);
        assert_eq!(sub_seqs, vec![]);
        let sub_seqs = find_subsequences_of_max_size_k(&values, 70, sum_fn);
        assert_eq!(sub_seqs, vec![(0, 2)]);
        values.reverse();
        let sub_seqs = find_subsequences_of_max_size_k(&values, 32, sum_fn);
        assert_eq!(sub_seqs, vec![(3, 4)]);
        let values: Vec<usize> = vec![14, 50, 10, 100];
        let sub_seqs = find_subsequences_of_max_size_k(&values, 32, sum_fn);
        assert_eq!(sub_seqs, vec![(0, 1), (2, 3)]);
        let values: Vec<usize> = vec![10, 50, 10, 10, 100];
        let sub_seqs = find_subsequences_of_max_size_k(&values, 70, sum_fn);
        assert_eq!(sub_seqs, vec![(0, 3), (1, 4)]);
        let values: Vec<usize> = vec![10, 50, 10, 10, 70];
        let sub_seqs = find_subsequences_of_max_size_k(&values, 70, sum_fn);
        assert_eq!(sub_seqs, vec![(0, 3), (1, 4), (4, 5)]);
        let values: Vec<usize> = vec![10, 50, 10, 10, 70, 10];
        let sub_seqs = find_subsequences_of_max_size_k(&values, 70, sum_fn);
        assert_eq!(sub_seqs, vec![(0, 3), (1, 4), (4, 5), (5, 6)]);
        let values: Vec<usize> = vec![10, 50, 10, 10, 65, 5];
        let sub_seqs = find_subsequences_of_max_size_k(&values, 70, sum_fn);
        assert_eq!(sub_seqs, vec![(0, 3), (1, 4), (4, 6)]);
        let values: Vec<usize> = vec![50, 50, 50, 50, 50, 50];
        let sub_seqs = find_subsequences_of_max_size_k(&values, 20, sum_fn);
        assert_eq!(sub_seqs, vec![]);
        let values: Vec<usize> = vec![50, 50, 50, 20, 50, 50];
        let sub_seqs = find_subsequences_of_max_size_k(&values, 20, sum_fn);
        assert_eq!(sub_seqs, vec![(3, 4)]);
    }
}
