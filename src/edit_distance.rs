use indicatif::ParallelProgressIterator;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::{get_progress_bar, Matrix};

#[derive(Copy, Clone, Debug)]
enum EditOp {
    None,
    Keep,
    Insert,
    Delete,
    Replace,
    Swap,
}

fn calculate_edit_matrices(
    a: &str,
    b: &str,
    with_swap: bool,
    spaces_insert_delete_only: bool,
) -> (Matrix<usize>, Matrix<EditOp>) {
    let a_chars = a.chars().collect::<Vec<char>>();
    let b_chars = b.chars().collect::<Vec<char>>();

    let mut d: Matrix<usize> = vec![vec![0; b_chars.len() + 1]; a_chars.len() + 1];
    let mut ops: Matrix<EditOp> = vec![vec![EditOp::None; b_chars.len() + 1]; a_chars.len() + 1];

    // initialize matrices
    ops[0][0] = EditOp::Keep;
    d[0][0] = 0;
    for i in 1..=a_chars.len() {
        d[i][0] = i;
        ops[i][0] = EditOp::Delete;
    }
    for j in 1..=b_chars.len() {
        d[0][j] = j;
        ops[0][j] = EditOp::Insert;
    }

    for (a_idx, &a_char) in a_chars.iter().enumerate() {
        for (b_idx, &b_char) in b_chars.iter().enumerate() {
            // string indices are offset by -1
            let i = a_idx + 1;
            let j = b_idx + 1;

            let mut costs = vec![
                (d[i - 1][j] + 1, EditOp::Delete), (d[i][j - 1] + 1, EditOp::Insert),
            ];
            if a_char == b_char {
                costs.push((d[i - 1][j - 1], EditOp::Keep));
            } else {
                // chars are not equal, only allow replacement if no space is involved
                // or we are allowed to replace spaces
                if !spaces_insert_delete_only || (!a_char.is_whitespace() && !b_char.is_whitespace()) {
                    costs.push((d[i - 1][j - 1] + 1, EditOp::Replace));
                }
            }
            // check if we can swap chars, that is if we are allowed to swap
            // and if the chars to swap match
            if with_swap && i > 1 && j > 1 && a_char == b_chars[j - 2] && a_chars[i - 2] == b_char {
                // we can swap the chars, but only allow swapping if no space is involved
                // or we are allowed to swap spaces
                if !spaces_insert_delete_only || (!a_char.is_whitespace() && !a_chars[i - 2].is_whitespace()) {
                    costs.push((d[i - 2][j - 2] + 1, EditOp::Swap));
                }
            }

            let (min_cost, min_op) = costs.iter().min_by(
                |(cost_1, _), (cost_2, _)| { cost_1.cmp(cost_2) }
            ).expect("should not happen");
            d[i][j] = *min_cost;
            ops[i][j] = *min_op;
        }
    }
    (d, ops)
}

pub fn edit_operations(
    a: &str,
    b: &str,
    with_swap: bool,
    spaces_insert_delete_only: bool,
) -> Vec<(u8, usize, usize)> {
    let (_, ops) = calculate_edit_matrices(a, b, with_swap, spaces_insert_delete_only);
    // backtrace
    // edit operations => 0 -> insert, 1 -> delete, 2 -> replace, 3 -> swap
    let mut edit_ops = vec![];
    let mut i = ops.len() - 1;
    let mut j = ops[0].len() - 1;
    while i > 0 || j > 0 {
        let op = &ops[i][j];
        match op {
            EditOp::None => { panic!("should not happen") }
            EditOp::Keep => {
                i -= 1;
                j -= 1;
            }
            EditOp::Insert => {
                j -= 1;
                edit_ops.push((0, i, j));
            }
            EditOp::Delete => {
                i -= 1;
                edit_ops.push((1, i, j));
            }
            EditOp::Replace => {
                i -= 1;
                j -= 1;
                edit_ops.push((2, i, j));
            }
            EditOp::Swap => {
                i -= 2;
                j -= 2;
                edit_ops.push((3, i, j));
            }
        }
    }
    edit_ops.reverse();
    edit_ops
}

#[pyfunction]
fn edit_operations_py(
    a: &str,
    b: &str,
    with_swap: bool,
    spaces_insert_delete_only: bool,
) -> PyResult<Vec<(u8, usize, usize)>> {
    Ok(edit_operations(&a, &b, with_swap, spaces_insert_delete_only))
}

pub fn batch_edit_operations(
    a_list: &Vec<&str>,
    b_list: &Vec<&str>,
    with_swap: bool,
    spaces_insert_delete_only: bool,
    batch_size: usize,
    show_progress: bool,
) -> Vec<Vec<(u8, usize, usize)>> {
    assert_eq!(a_list.len(), b_list.len(), "lists don't have the same length");
    let pb = get_progress_bar(
        ((a_list.len() + batch_size - 1) / batch_size).max(1) as u64,
        !show_progress,
    );
    a_list
        .par_chunks(batch_size)
        .zip(b_list.par_chunks(batch_size))
        .progress_with(pb)
        .map(|(a_chunk, b_chunk)| {
            a_chunk.iter().zip(b_chunk.iter()).map(|(a, b)| {
                edit_operations(a, b, with_swap, spaces_insert_delete_only)
            }).collect::<Vec<Vec<(u8, usize, usize)>>>()
        })
        .flatten()
        .collect()
}

#[pyfunction]
fn batch_edit_operations_py(
    a_list: Vec<&str>,
    b_list: Vec<&str>,
    with_swap: bool,
    spaces_insert_delete_only: bool,
    batch_size: usize,
    show_progress: bool,
) -> PyResult<Vec<Vec<(u8, usize, usize)>>> {
    Ok(batch_edit_operations(
        &a_list,
        &b_list,
        with_swap,
        spaces_insert_delete_only,
        batch_size,
        show_progress,
    ))
}

pub fn edit_distance(
    a: &str,
    b: &str,
    with_swap: bool,
    spaces_insert_delete_only: bool,
) -> usize {
    let (d, _) = calculate_edit_matrices(a, b, with_swap, spaces_insert_delete_only);
    d[d.len() - 1][d[0].len() - 1]
}

#[pyfunction]
fn edit_distance_py(
    a: &str,
    b: &str,
    with_swap: bool,
    spaces_insert_delete_only: bool,
) -> PyResult<usize> {
    Ok(edit_distance(a, b, with_swap, spaces_insert_delete_only))
}

pub fn batch_edit_distance(
    a_list: &Vec<&str>,
    b_list: &Vec<&str>,
    with_swap: bool,
    spaces_insert_delete_only: bool,
    batch_size: usize,
    show_progress: bool,
) -> Vec<usize> {
    assert_eq!(a_list.len(), b_list.len(), "lists don't have the same length");
    let pb = get_progress_bar(
        ((a_list.len() + batch_size - 1) / batch_size).max(1) as u64,
        !show_progress,
    );
    a_list
        .par_chunks(batch_size)
        .zip(b_list.par_chunks(batch_size))
        .progress_with(pb)
        .map(|(a_chunk, b_chunk)| {
            a_chunk.iter().zip(b_chunk.iter()).map(|(a, b)| {
                edit_distance(a, b, with_swap, spaces_insert_delete_only)
            }).collect::<Vec<usize>>()
        })
        .flatten()
        .collect()
}

#[pyfunction]
fn batch_edit_distance_py(
    a_list: Vec<&str>,
    b_list: Vec<&str>,
    with_swap: bool,
    spaces_insert_delete_only: bool,
    batch_size: usize,
    show_progress: bool,
) -> PyResult<Vec<usize>> {
    Ok(batch_edit_distance(&a_list, &b_list, with_swap, spaces_insert_delete_only, batch_size, show_progress))
}

pub(super) fn add_submodule(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "edit_distance")?;
    m.add_function(wrap_pyfunction!(edit_distance_py, m)?)?;
    m.add_function(wrap_pyfunction!(batch_edit_distance_py, m)?)?;
    m.add_function(wrap_pyfunction!(edit_operations_py, m)?)?;
    m.add_function(wrap_pyfunction!(batch_edit_operations_py, m)?)?;
    parent_module.add_submodule(m)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::edit_distance::{edit_distance, edit_operations};

    #[test]
    fn test_edit_operations() {
        let ed_ops = edit_operations(
            "this is a test",
            "tihsi s a test",
            true,
            false,
        );
        assert_eq!(ed_ops, vec![(3, 1, 1), (3, 4, 4)]);
        let ed_ops = edit_operations(
            "this is a test",
            "tihsi s a test",
            true,
            true,
        );
        assert_eq!(ed_ops, vec![(3, 1, 1), (0, 4, 4), (1, 5, 6)]);
        let ed_ops = edit_operations(
            "this is a test",
            "tihsi s a test",
            false,
            false,
        );
        assert_eq!(ed_ops, vec![(0, 1, 1), (0, 2, 3), (1, 3, 5), (1, 5, 6)]);
        let ed_ops = edit_operations(
            "this is a test",
            "tihsi s a test",
            false,
            true,
        );
        assert_eq!(ed_ops, vec![(0, 1, 1), (0, 2, 3), (1, 3, 5), (1, 5, 6)]);
    }

    #[test]
    fn test_edit_distance() {
        let ed = edit_distance(
            "this is a test",
            "tihsi s a test",
            true,
            false,
        );
        assert_eq!(ed, 2);
        let ed = edit_distance(
            "this is a test",
            "tihsi s a test",
            true,
            true,
        );
        assert_eq!(ed, 3);
        let ed = edit_distance(
            "this is a test",
            "tihsi s a test",
            false,
            false,
        );
        assert_eq!(ed, 4);
        let ed = edit_distance(
            "this is a test",
            "tihsi s a test",
            false,
            true,
        );
        assert_eq!(ed, 4);
    }
}