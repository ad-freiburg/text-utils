use crate::text::match_words;
use crate::unicode::{CharString, Character, CS};
use anyhow::anyhow;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyString;
use std::collections::HashSet;

use crate::utils::py_invalid_type_error;

#[derive(Copy, Clone, Debug)]
enum EditOp {
    None,
    Keep,
    Insert,
    Delete,
    Replace,
    Swap,
}

#[inline]
fn _calculate_edit_matrices(
    a: CharString,
    b: CharString,
    with_swap: bool,
    spaces_insert_delete_only: bool,
) -> (Vec<usize>, Vec<EditOp>) {
    let rows = a.len() + 1;
    let cols = b.len() + 1;
    let mut d = vec![0; rows * cols];
    let mut ops = vec![EditOp::None; rows * cols];

    // initialize matrices
    d[0] = 0;
    ops[0] = EditOp::Keep;
    for i in 1..=a.len() {
        d[i * cols] = i;
        ops[i * cols] = EditOp::Delete;
    }
    for j in 1..=b.len() {
        d[j] = j;
        ops[j] = EditOp::Insert;
    }
    let a_chars: Vec<Character> = a.chars().collect();
    let b_chars: Vec<Character> = b.chars().collect();
    for (a_idx, a_char) in a_chars.iter().enumerate() {
        for (b_idx, b_char) in b_chars.iter().enumerate() {
            // string indices are offset by -1
            let i = a_idx + 1;
            let j = b_idx + 1;

            let mut costs = vec![
                (d[(i - 1) * cols + j] + 1, EditOp::Delete),
                (d[i * cols + j - 1] + 1, EditOp::Insert),
            ];
            if a_char == b_char {
                costs.push((d[(i - 1) * cols + j - 1], EditOp::Keep));
            } else {
                // chars are not equal, only allow replacement if no space is involved
                // or we are allowed to replace spaces
                if !spaces_insert_delete_only
                    || (!a_char.is_whitespace() && !b_char.is_whitespace())
                {
                    costs.push((d[(i - 1) * cols + j - 1] + 1, EditOp::Replace));
                }
            }
            // check if we can swap chars, that is if we are allowed to swap
            // and if the chars to swap match
            if with_swap && i > 1 && j > 1 && a_char == &b_chars[j - 2] && &a_chars[i - 2] == b_char
            {
                // we can swap the chars, but only allow swapping if no space is involved
                // or we are allowed to swap spaces
                if !spaces_insert_delete_only
                    || (!a_char.is_whitespace() && !a_chars[i - 2].is_whitespace())
                {
                    costs.push((d[(i - 2) * cols + j - 2] + 1, EditOp::Swap));
                }
            }

            let (min_cost, min_op) = costs
                .iter()
                .min_by(|(cost_1, _), (cost_2, _)| cost_1.cmp(cost_2))
                .expect("should not happen");
            d[i * cols + j] = *min_cost;
            ops[i * cols + j] = *min_op;
        }
    }
    (d, ops)
}

#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum EditOperation {
    Insert,
    Delete,
    Replace,
    Swap,
}

impl<'a> FromPyObject<'a> for EditOperation {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        let s: String = ob.extract()?;
        let edit_op = match s.as_str() {
            "i" | "insert" => EditOperation::Insert,
            "d" | "delete" => EditOperation::Delete,
            "r" | "replace" => EditOperation::Replace,
            "s" | "swap" => EditOperation::Swap,
            k => return Err(py_invalid_type_error(k, "edit operation")),
        };
        Ok(edit_op)
    }
}

impl<'py> IntoPyObject<'py> for EditOperation {
    type Target = PyString;
    type Output = Bound<'py, Self::Target>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            EditOperation::Insert => "i",
            EditOperation::Delete => "d",
            EditOperation::Replace => "r",
            EditOperation::Swap => "s",
        }
        .into_pyobject(py)
    }
}

#[pyfunction(signature = (a, b, use_graphemes = true, with_swap = true, spaces_insert_delete_only = false))]
pub fn operations(
    a: &str,
    b: &str,
    use_graphemes: bool,
    with_swap: bool,
    spaces_insert_delete_only: bool,
) -> Vec<(EditOperation, usize, usize)> {
    let a_cs = CS::new(a, use_graphemes);
    let b_cs = CS::new(b, use_graphemes);
    let mut i = a_cs.len();
    let mut j = b_cs.len();
    let cols = b_cs.len() + 1;
    let (_, ops) = _calculate_edit_matrices(a_cs, b_cs, with_swap, spaces_insert_delete_only);
    // backtrace
    // edit operations => 0 -> insert, 1 -> delete, 2 -> replace, 3 -> swap
    let mut edit_ops = vec![];
    while i > 0 || j > 0 {
        let op = &ops[i * cols + j];
        match op {
            EditOp::None => {
                panic!("should not happen")
            }
            EditOp::Keep => {
                i -= 1;
                j -= 1;
            }
            EditOp::Insert => {
                j -= 1;
                edit_ops.push((EditOperation::Insert, i, j));
            }
            EditOp::Delete => {
                i -= 1;
                edit_ops.push((EditOperation::Delete, i, j));
            }
            EditOp::Replace => {
                i -= 1;
                j -= 1;
                edit_ops.push((EditOperation::Replace, i, j));
            }
            EditOp::Swap => {
                i -= 2;
                j -= 2;
                edit_ops.push((EditOperation::Swap, i, j));
            }
        }
    }
    edit_ops.reverse();
    edit_ops
}

#[pyfunction(signature = (a, b, use_graphemes = true, with_swap = true, spaces_insert_delete_only = false, normalized = false))]
pub fn distance(
    a: &str,
    b: &str,
    use_graphemes: bool,
    with_swap: bool,
    spaces_insert_delete_only: bool,
    normalized: bool,
) -> f64 {
    let a_cs = CS::new(a, use_graphemes);
    let b_cs = CS::new(b, use_graphemes);
    let norm = if normalized {
        a_cs.len().max(b_cs.len()) as f64
    } else {
        1.0
    };
    let (d, _) = _calculate_edit_matrices(a_cs, b_cs, with_swap, spaces_insert_delete_only);
    d.last().copied().unwrap_or(0) as f64 / norm
}

pub fn distances(
    a: &[impl AsRef<str>],
    b: &[impl AsRef<str>],
    use_graphemes: bool,
    with_swap: bool,
    spaces_insert_delete_only: bool,
    normalized: bool,
) -> anyhow::Result<Vec<f64>> {
    if a.len() != b.len() {
        return Err(anyhow!(
            "a and b must have the same length, but got {} and {}",
            a.len(),
            b.len()
        ));
    }
    Ok(a.iter()
        .zip(b.iter())
        .map(|(a, b)| {
            distance(
                a.as_ref(),
                b.as_ref(),
                use_graphemes,
                with_swap,
                spaces_insert_delete_only,
                normalized,
            )
        })
        .collect())
}

#[pyfunction(name = "distances", signature = (a, b, use_graphemes = true, with_swap = true, spaces_insert_delete_only = false, normalized = false))]
fn distances_py(
    a: Vec<PyBackedStr>,
    b: Vec<PyBackedStr>,
    use_graphemes: bool,
    with_swap: bool,
    spaces_insert_delete_only: bool,
    normalized: bool,
) -> anyhow::Result<Vec<f64>> {
    distances(
        &a,
        &b,
        use_graphemes,
        with_swap,
        spaces_insert_delete_only,
        normalized,
    )
}

#[pyfunction(signature = (a, b, use_graphemes = true, with_swap = true, spaces_insert_delete_only = false, normalized = false))]
pub fn prefix_distance(
    a: &str,
    b: &str,
    use_graphemes: bool,
    with_swap: bool,
    spaces_insert_delete_only: bool,
    normalized: bool,
) -> f64 {
    let a_cs = CS::new(a, use_graphemes);
    let b_cs = CS::new(b, use_graphemes);
    let i = a_cs.len();
    let cols = b_cs.len() + 1;
    let norm = if normalized { a_cs.len() as f64 } else { 1.0 };
    let (d, _) = _calculate_edit_matrices(a_cs, b_cs, with_swap, spaces_insert_delete_only);

    // find minimum in last row
    d[i * cols..(i + 1) * cols]
        .iter()
        .min()
        .copied()
        .unwrap_or(0) as f64
        / norm
}

#[pyfunction]
pub fn edited_words(a: &str, b: &str) -> (HashSet<usize>, HashSet<usize>) {
    let (matching_words, a_len, b_len) = match_words(a, b, false);
    let a_words = HashSet::from_iter(0..a_len);
    let unedited_a_words: HashSet<usize> = matching_words
        .iter()
        .map(|(a_word_idx, _)| *a_word_idx)
        .collect();
    let b_words = HashSet::from_iter(0..b_len);
    let unedited_b_words: HashSet<usize> = matching_words
        .iter()
        .map(|(_, b_word_idx)| *b_word_idx)
        .collect();
    (
        a_words.difference(&unedited_a_words).cloned().collect(),
        b_words.difference(&unedited_b_words).cloned().collect(),
    )
}

/// A submodule for calculating the edit distance and operations between strings.
pub(super) fn add_submodule(py: Python, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m_name = "edit";
    let m = PyModule::new(py, m_name)?;
    m.add_function(wrap_pyfunction!(distance, m.clone())?)?;
    m.add_function(wrap_pyfunction!(distances_py, m.clone())?)?;
    m.add_function(wrap_pyfunction!(prefix_distance, m.clone())?)?;
    m.add_function(wrap_pyfunction!(operations, m.clone())?)?;
    m.add_function(wrap_pyfunction!(edited_words, m.clone())?)?;
    parent_module.add_submodule(&m)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};

    use crate::edit::{distance, edited_words, operations, EditOperation};
    use std::collections::HashSet;

    #[test]
    fn test_edit_operations() {
        let ed_ops = operations("this is a test", "tihsi s a test", false, true, false);
        assert_eq!(
            ed_ops,
            vec![(EditOperation::Swap, 1, 1), (EditOperation::Swap, 4, 4)]
        );
        let ed_ops = operations("this is a test", "tihsi s a test", false, true, true);
        assert_eq!(
            ed_ops,
            vec![
                (EditOperation::Swap, 1, 1),
                (EditOperation::Insert, 4, 4),
                (EditOperation::Delete, 5, 6)
            ]
        );
        let ed_ops = operations("this is a test", "tihsi s a test", false, false, false);
        assert_eq!(
            ed_ops,
            vec![
                (EditOperation::Insert, 1, 1),
                (EditOperation::Insert, 2, 3),
                (EditOperation::Delete, 3, 5),
                (EditOperation::Delete, 5, 6)
            ]
        );
        let ed_ops = operations("this is a test", "tihsi s a test", false, false, true);
        assert_eq!(
            ed_ops,
            vec![
                (EditOperation::Insert, 1, 1),
                (EditOperation::Insert, 2, 3),
                (EditOperation::Delete, 3, 5),
                (EditOperation::Delete, 5, 6)
            ]
        );
        let ed_ops = operations("thyis is a texst", "thxis is a teyst", false, false, false);
        assert_eq!(
            ed_ops,
            vec![
                (EditOperation::Replace, 2, 2),
                (EditOperation::Replace, 13, 13)
            ]
        );

        let mut rng = rand_chacha::ChaCha8Rng::from_os_rng();

        let mut rand_string = |min_size: usize, max_size: usize| -> String {
            let size: usize = rng.random_range(min_size..=max_size);
            (0..size)
                .map(|_| char::from_u32(rng.random_range(97..=100)).unwrap())
                .collect()
        };
        // test that edit operations are returned always in sorted order
        // with some randomly created ascii (characters a-d) strings
        for _ in 0..1000 {
            let rand_a: String = rand_string(8, 128);
            let rand_b: String = rand_string(8, 128);
            let ops = operations(&rand_a, &rand_b, true, false, false);
            assert!(ops
                .iter()
                .enumerate()
                .map(|(idx, (_, a_op_idx, b_op_idx))| {
                    let a_idx_geq = idx == 0 || (*a_op_idx >= ops[idx - 1].1);
                    let b_idx_geq = idx == 0 || (*b_op_idx >= ops[idx - 1].2);
                    a_idx_geq && b_idx_geq
                })
                .all(|b| b))
        }
    }

    #[test]
    fn test_edit_distance() {
        let ed = distance(
            "this is a test",
            "tihsi s a test",
            false,
            true,
            false,
            false,
        );
        assert_eq!(ed as usize, 2);
        let ed = distance("this is a test", "tihsi s a test", false, true, true, false);
        assert_eq!(ed as usize, 3);
        let ed = distance(
            "this is a test",
            "tihsi s a test",
            false,
            false,
            false,
            false,
        );
        assert_eq!(ed as usize, 4);
        let ed = distance(
            "this is a test",
            "tihsi s a test",
            false,
            false,
            true,
            false,
        );
        assert_eq!(ed as usize, 4);
    }

    #[test]
    fn test_edited_words() {
        let a_sequences = vec![
            "th s is a test",
            "we do no match",
            "Just a rwong sequence",
            "one last example",
            "final last example",
        ];
        let b_sequences = vec![
            "this is a test",
            "we do not match",
            "just a wrong sequence",
            "one last last last example here",
            "last example here",
        ];
        let (edited_a, edited_b) = edited_words(a_sequences[0], b_sequences[0]);
        assert_eq!(edited_a, HashSet::from([0, 1]));
        assert_eq!(edited_b, HashSet::from([0]));
        let (edited_a, edited_b) = edited_words(a_sequences[1], b_sequences[1]);
        assert_eq!(edited_a, HashSet::from([2]));
        assert_eq!(edited_b, HashSet::from([2]));
        let (edited_a, edited_b) = edited_words(a_sequences[2], b_sequences[2]);
        assert_eq!(edited_a, HashSet::from([0, 2]));
        assert_eq!(edited_b, HashSet::from([0, 2]));
        let (edited_a, edited_b) = edited_words(a_sequences[3], b_sequences[3]);
        assert_eq!(edited_a, HashSet::from([]));
        assert_eq!(edited_b, HashSet::from([1, 2, 5]));
        let (edited_a, edited_b) = edited_words(a_sequences[4], b_sequences[4]);
        assert_eq!(edited_a, HashSet::from([0]));
        assert_eq!(edited_b, HashSet::from([2]));
    }
}
