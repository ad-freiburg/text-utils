use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use num::traits::Num;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

pub(crate) type Matrix<T> = Vec<Vec<T>>;

pub(crate) fn get_progress_bar(size: u64, hidden: bool) -> ProgressBar {
    let pb = ProgressBar::new(size)
        .with_style(
            ProgressStyle::with_template(
                "{msg}: {wide_bar} [{pos}/{len}] [{elapsed_precise}|{eta_precise}]",
            )
            .unwrap(),
        )
        .with_message("matching words");
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
    T: Num + Copy,
{
    accumulate_with(values, |acc, v| *acc + *v)
}

#[cfg(feature = "benchmark-utils")]
#[inline]
pub fn accumulate_pub<T>(values: &[T]) -> Vec<T>
where
    T: Num + Copy,
{
    accumulate(values)
}

#[inline]
pub(crate) fn constrain<T>(value: T, min: T, max: T) -> T
where
    T: Num + PartialOrd,
{
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
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
pub(crate) fn as_ref_slice_to_vec(values: &[impl AsRef<str>]) -> Vec<&str> {
    values.iter().map(|s| s.as_ref()).collect()
}

pub(crate) fn py_required_key_error(key_name: &str, value_name: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "could not find required key \"{key_name}\" for \
            {value_name}"
    ))
}

pub(crate) fn py_invalid_type_error(name: &str, type_name: &str) -> PyErr {
    PyTypeError::new_err(format!("\"{name}\" is not a valid {type_name} type"))
}

#[cfg(test)]
mod tests {
    use crate::utils::{accumulate, run_length_decode, run_length_encode};

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
}
