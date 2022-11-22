use pyo3::prelude::*;
use crate::unicode::{Character, CharString, CS};

use crate::utils::{Matrix};

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
) -> (Matrix<usize>, Matrix<EditOp>) {
    let mut d: Matrix<usize> = vec![vec![0; b.len() + 1]; a.len() + 1];
    let mut ops: Matrix<EditOp> = vec![vec![EditOp::None; b.len() + 1]; a.len() + 1];

    // initialize matrices
    ops[0][0] = EditOp::Keep;
    d[0][0] = 0;
    for i in 1..=a.len() {
        d[i][0] = i;
        ops[i][0] = EditOp::Delete;
    }
    for j in 1..=b.len() {
        d[0][j] = j;
        ops[0][j] = EditOp::Insert;
    }
    let a_chars: Vec<Character> = a.chars().collect();
    let b_chars: Vec<Character> = b.chars().collect();
    for (a_idx, a_char) in a_chars.iter().enumerate() {
        for (b_idx, b_char) in b_chars.iter().enumerate() {
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
                if !spaces_insert_delete_only
                    || (!a_char.is_whitespace() && !b_char.is_whitespace()) {
                    costs.push((d[i - 1][j - 1] + 1, EditOp::Replace));
                }
            }
            // check if we can swap chars, that is if we are allowed to swap
            // and if the chars to swap match
            if with_swap
                && i > 1
                && j > 1
                && a_char == &b_chars[j - 2]
                && &a_chars[i - 2] == b_char {
                // we can swap the chars, but only allow swapping if no space is involved
                // or we are allowed to swap spaces
                if !spaces_insert_delete_only
                    || (!a_char.is_whitespace() && !a_chars[i - 2].is_whitespace()) {
                    costs.push((d[i - 2][j - 2] + 1, EditOp::Swap));
                }
            }

            let (min_cost, min_op) = costs.
                iter()
                .min_by(|(cost_1, _), (cost_2, _)| {
                    cost_1.cmp(cost_2)
                })
                .expect("should not happen");
            d[i][j] = *min_cost;
            ops[i][j] = *min_op;
        }
    }
    (d, ops)
}

pub fn edit_operations(
    a: &str,
    b: &str,
    use_graphemes: bool,
    with_swap: bool,
    spaces_insert_delete_only: bool,
) -> Vec<(u8, usize, usize)> {
    let (_, ops) = _calculate_edit_matrices(
        CS::new(a, use_graphemes),
        CS::new(b, use_graphemes),
        with_swap,
        spaces_insert_delete_only,
    );
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

#[pyfunction(
use_graphemes = "true",
with_swap = "true",
spaces_insert_delete_only = "false"
)]
#[pyo3(name = "edit_operations")]
fn edit_operations_py(
    a: &str,
    b: &str,
    use_graphemes: bool,
    with_swap: bool,
    spaces_insert_delete_only: bool,
) -> PyResult<Vec<(u8, usize, usize)>> {
    Ok(edit_operations(&a, &b, use_graphemes, with_swap, spaces_insert_delete_only))
}

pub fn edit_distance(
    a: &str,
    b: &str,
    use_graphemes: bool,
    with_swap: bool,
    spaces_insert_delete_only: bool,
) -> usize {
    let (d, _) = _calculate_edit_matrices(
        CS::new(a, use_graphemes),
        CS::new(b, use_graphemes),
        with_swap,
        spaces_insert_delete_only,
    );
    d[d.len() - 1][d[0].len() - 1]
}

#[pyfunction(
use_graphemes = "true",
with_swap = "true",
spaces_insert_delete_only = "false"
)]
#[pyo3(name = "edit_distance")]
fn edit_distance_py(
    a: &str,
    b: &str,
    use_graphemes: bool,
    with_swap: bool,
    spaces_insert_delete_only: bool,
) -> PyResult<usize> {
    Ok(edit_distance(
        a,
        b,
        use_graphemes,
        with_swap,
        spaces_insert_delete_only,
    ))
}

pub(super) fn add_submodule(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "edit_distance")?;
    m.add_function(wrap_pyfunction!(edit_distance_py, m)?)?;
    m.add_function(wrap_pyfunction!(edit_operations_py, m)?)?;
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
            false,
            true,
            false,
        );
        assert_eq!(ed_ops, vec![(3, 1, 1), (3, 4, 4)]);
        let ed_ops = edit_operations(
            "this is a test",
            "tihsi s a test",
            false,
            true,
            true,
        );
        assert_eq!(ed_ops, vec![(3, 1, 1), (0, 4, 4), (1, 5, 6)]);
        let ed_ops = edit_operations(
            "this is a test",
            "tihsi s a test",
            false,
            false,
            false,
        );
        assert_eq!(ed_ops, vec![(0, 1, 1), (0, 2, 3), (1, 3, 5), (1, 5, 6)]);
        let ed_ops = edit_operations(
            "this is a test",
            "tihsi s a test",
            false,
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
            false,
            true,
            false,
        );
        assert_eq!(ed, 2);
        let ed = edit_distance(
            "this is a test",
            "tihsi s a test",
            false,
            true,
            true,
        );
        assert_eq!(ed, 3);
        let ed = edit_distance(
            "this is a test",
            "tihsi s a test",
            false,
            false,
            false,
        );
        assert_eq!(ed, 4);
        let ed = edit_distance(
            "this is a test",
            "tihsi s a test",
            false,
            false,
            true,
        );
        assert_eq!(ed, 4);
    }
}