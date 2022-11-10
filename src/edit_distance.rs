use indicatif::ParallelProgressIterator;
use pyo3::prelude::*;
use pyo3::types::*;
use rayon::prelude::*;

use crate::utils;

#[derive(Copy, Clone, Debug)]
enum EditOp {
    None,
    Keep,
    Insert,
    Delete,
    Replace,
    Swap,
}

type Matrix<T> = Vec<Vec<T>>;

fn calculate_edit_matrices(
    a: &String,
    b: &String,
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
                if !spaces_insert_delete_only || (a_char != ' ' && b_char != ' ') {
                    costs.push((d[i - 1][j - 1] + 1, EditOp::Replace));
                }
            }
            // check if we can swap chars, that is if we are allowed to swap
            // and if the chars to swap match
            if with_swap && i > 1 && j > 1 && a_char == b_chars[j - 2] && a_chars[i - 2] == b_char {
                // we can swap the chars, but only allow swapping if no space is involved
                // or we are allowed to swap spaces
                if !spaces_insert_delete_only || (a_char != ' ' && a_chars[i - 2] != ' ') {
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

fn calculate_edit_operations(
    a: &String,
    b: &String,
    with_swap: bool,
    spaces_insert_delete_only: bool,
) -> Vec<(String, usize, usize)> {
    let (_, ops) = calculate_edit_matrices(a, b, with_swap, spaces_insert_delete_only);
    // backtrace
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
                edit_ops.push(("i".to_string(), i, j));
            }
            EditOp::Delete => {
                i -= 1;
                edit_ops.push(("d".to_string(), i, j));
            }
            EditOp::Replace => {
                i -= 1;
                j -= 1;
                edit_ops.push(("r".to_string(), i, j));
            }
            EditOp::Swap => {
                i -= 2;
                j -= 2;
                edit_ops.push(("s".to_string(), i, j));
            }
        }
    }
    edit_ops.reverse();
    edit_ops
}

fn calculate_edit_distance(
    a: &String,
    b: &String,
    with_swap: bool,
    spaces_insert_delete_only: bool,
) -> usize {
    let (d, _) = calculate_edit_matrices(a, b, with_swap, spaces_insert_delete_only);
    d[d.len() - 1][d[0].len() - 1]
}

#[derive(Copy, Clone, Debug)]
enum MatchOp {
    None,
    Delete,
    Insert,
    Match,
    NoMatch,
}

fn calculate_word_matching(
    a: &String,
    b: &String,
) -> Vec<(usize, usize)> {
    let a_words = a.split_whitespace().collect::<Vec<&str>>();
    let b_words = b.split_whitespace().collect::<Vec<&str>>();

    let mut d: Matrix<usize> = vec![vec![0; b_words.len() + 1]; a_words.len() + 1];
    let mut ops: Matrix<MatchOp> = vec![vec![MatchOp::None; b_words.len() + 1]; a_words.len() + 1];

    // initialize matrices
    ops[0][0] = MatchOp::NoMatch;
    for i in 1..=a_words.len() {
        ops[i][0] = MatchOp::Delete;
    }
    for j in 1..=b_words.len() {
        ops[0][j] = MatchOp::Insert;
    }

    for (a_idx, &a_word) in a_words.iter().enumerate() {
        for (b_idx, &b_word) in b_words.iter().enumerate() {
            // string indices are offset by -1
            let i = a_idx + 1;
            let j = b_idx + 1;

            let values = vec![
                (d[i - 1][j], MatchOp::Delete),
                (d[i][j - 1], MatchOp::Insert),
                (d[i - 1][j - 1] + ((a_word == b_word) as usize),
                 if a_word == b_word { MatchOp::Match } else { MatchOp::NoMatch }),
            ];

            let (max_value, max_op) = values.iter().max_by(|(v_1, _), (v_2, _)| {
                v_1.cmp(v_2)
            }).expect("should not happen");
            d[i][j] = *max_value;
            ops[i][j] = *max_op;
        }
    }

    // backtrace
    let mut matches = vec![];
    let mut i = a_words.len();
    let mut j = b_words.len();
    while i > 0 || j > 0 {
        let op = &ops[i][j];
        match op {
            MatchOp::None => { panic!("should not happen") }
            MatchOp::Delete => { i -= 1; }
            MatchOp::Insert => { j -= 1; }
            MatchOp::Match => {
                i -= 1;
                j -= 1;
                matches.push((i, j));
            }
            MatchOp::NoMatch => {
                i -= 1;
                j -= 1;
            }
        }
    }
    matches.reverse();
    matches
}


#[pyfunction]
fn edit_distance(
    a: String,
    b: String,
    with_swap: bool,
    spaces_insert_delete_only: bool,
) -> PyResult<usize> {
    Ok(calculate_edit_distance(&a, &b, with_swap, spaces_insert_delete_only))
}

#[pyfunction]
fn batch_edit_distance(
    a_list: Vec<String>,
    b_list: Vec<String>,
    with_swap: bool,
    spaces_insert_delete_only: bool,
    batch_size: usize,
    show_progress: bool
) -> PyResult<Vec<usize>> {
    assert_eq!(a_list.len(), b_list.len());
    let pb = utils::get_progress_bar(
        ((a_list.len() + batch_size - 1) / batch_size).max(1) as u64,
        !show_progress
    );
    let edit_distances = a_list
        .par_chunks(batch_size)
        .zip(b_list.par_chunks(batch_size))
        .progress_with(pb)
        .map(|(a_chunk, b_chunk)| {
            a_chunk.iter().zip(b_chunk.iter()).map(|(a, b)| {
                calculate_edit_distance(a, b, with_swap, spaces_insert_delete_only)
            }).collect::<Vec<usize>>()
        })
        .flatten()
        .collect();
    Ok(edit_distances)
}

#[pyfunction]
fn edit_operations(
    a: String,
    b: String,
    with_swap: bool,
    spaces_insert_delete_only: bool,
) -> PyResult<Vec<(String, usize, usize)>> {
    Ok(calculate_edit_operations(&a, &b, with_swap, spaces_insert_delete_only))
}

#[pyfunction]
fn batch_edit_operations(
    a_list: Vec<String>,
    b_list: Vec<String>,
    with_swap: bool,
    spaces_insert_delete_only: bool,
    batch_size: usize,
    show_progress: bool
) -> PyResult<Vec<Vec<(String, usize, usize)>>> {
    assert_eq!(a_list.len(), b_list.len());
    let pb = utils::get_progress_bar(
        ((a_list.len() + batch_size - 1) / batch_size).max(1) as u64,
        !show_progress
    );
    let edit_operations = a_list
        .par_chunks(batch_size)
        .zip(b_list.par_chunks(batch_size))
        .progress_with(pb)
        .map(|(a_chunk, b_chunk)| {
            a_chunk.iter().zip(b_chunk.iter()).map(|(a, b)| {
                calculate_edit_operations(a, b, with_swap, spaces_insert_delete_only)
            }).collect::<Vec<Vec<(String, usize, usize)>>>()
        })
        .flatten()
        .collect();
    Ok(edit_operations)
}

#[pyfunction]
fn match_words(
    a: String,
    b: String,
) -> PyResult<Vec<(usize, usize)>> {
    Ok(calculate_word_matching(&a, &b))
}

#[pyfunction]
fn batch_match_words(
    a_list: Vec<String>,
    b_list: Vec<String>,
    batch_size: usize,
    show_progress: bool
) -> PyResult<Vec<Vec<(usize, usize)>>> {
    assert_eq!(a_list.len(), b_list.len());
    let pb = utils::get_progress_bar(
        ((a_list.len() + batch_size - 1) / batch_size).max(1) as u64,
        !show_progress
    );
    let word_matchings = a_list
        .par_chunks(batch_size)
        .zip(b_list.par_chunks(batch_size))
        .progress_with(pb)
        .map(|(a_chunk, b_chunk)| {
            a_chunk.iter().zip(b_chunk.iter()).map(|(a, b)| {
                calculate_word_matching(a, b)
            }).collect::<Vec<Vec<(usize, usize)>>>()
        })
        .flatten()
        .collect();
    Ok(word_matchings)
}

pub fn add_functions(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(edit_distance, m)?)?;
    m.add_function(wrap_pyfunction!(batch_edit_distance, m)?)?;
    m.add_function(wrap_pyfunction!(edit_operations, m)?)?;
    m.add_function(wrap_pyfunction!(batch_edit_operations, m)?)?;
    m.add_function(wrap_pyfunction!(match_words, m)?)?;
    m.add_function(wrap_pyfunction!(batch_match_words, m)?)?;

    Ok(())
}