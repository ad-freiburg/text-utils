use indicatif::ParallelProgressIterator;
use itertools::Itertools;
use pyo3::prelude::*;
use rayon::prelude::*;
use crate::unicode::CS;

use crate::utils::{get_progress_bar, Matrix};

pub fn clean(s: &str) -> String {
    s.split_whitespace().join(" ")
}

#[pyfunction]
fn clean_py(s: &str) -> PyResult<String> {
    Ok(clean(s))
}

pub fn batch_clean(
    list: &Vec<&str>,
    batch_size: usize,
    show_progress: bool,
) -> Vec<String> {
    let pb = get_progress_bar(
        ((list.len() + batch_size - 1) / batch_size).max(1) as u64,
        !show_progress,
    );
    list
        .par_chunks(batch_size)
        .progress_with(pb)
        .map(|chunk| {
            chunk.iter().map(|s| {
                clean(s)
            }).collect::<Vec<String>>()
        })
        .flatten()
        .collect()
}

#[pyfunction]
fn batch_clean_py(
    list: Vec<&str>,
    batch_size: usize,
    show_progress: bool,
) -> PyResult<Vec<String>> {
    Ok(batch_clean(&list, batch_size, show_progress))
}

pub fn word_boundaries(s: &str, graphemes: bool) -> Vec<(usize, usize)> {
    let cs: CS = if graphemes {
        CS::with_graphemes(s)
    } else {
        CS::with_chars(s)
    };
    _word_boundaries(cs.chars().map(|c| c.str))
}

fn _word_boundaries<'a>(chars: impl Iterator<Item=&'a str>) -> Vec<(usize, usize)> {
    let mut boundaries = vec![];
    let mut start: Option<usize> = None;
    let mut num_elements = 0;
    for (idx, char) in chars.enumerate() {
        match (char.chars().all(char::is_whitespace), start) {
            (true, Some(start_idx)) => {
                boundaries.push((start_idx, idx));
                start = None;
            }
            (false, None) => start = Some(idx),
            _ => ()
        }
        num_elements += 1;
    }
    // add potential last word (not captured by the for loop above)
    if start.is_some() && start.unwrap() < num_elements {
        boundaries.push((start.unwrap(), num_elements));
    }
    boundaries
}

#[pyfunction]
fn word_boundaries_py(s: &str, graphemes: bool) -> PyResult<Vec<(usize, usize)>> {
    Ok(word_boundaries(s, graphemes))
}


pub fn batch_word_boundaries(
    list: &Vec<&str>,
    graphemes: bool,
    batch_size: usize,
    show_progress: bool,
) -> Vec<Vec<(usize, usize)>> {
    let pb = get_progress_bar(
        ((list.len() + batch_size - 1) / batch_size).max(1) as u64,
        !show_progress,
    );
    list
        .par_chunks(batch_size)
        .progress_with(pb)
        .map(|chunk| {
            chunk.iter().map(|&s| {
                word_boundaries(s, graphemes)
            }).collect::<Vec<Vec<(usize, usize)>>>()
        })
        .flatten()
        .collect()
}

#[pyfunction]
fn batch_word_boundaries_py(
    list: Vec<&str>,
    graphemes: bool,
    batch_size: usize,
    show_progress: bool,
) -> PyResult<Vec<Vec<(usize, usize)>>> {
    Ok(
        batch_word_boundaries(&list, graphemes, batch_size, show_progress)
    )
}


fn str_match(a: &str, b: &str, ignore_case: bool) -> bool {
    a == b || (ignore_case && a.to_lowercase() == b.to_lowercase())
}

pub fn match_words(
    a: &str,
    b: &str,
    ignore_case: bool,
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

            let matching = str_match(a_word, b_word, ignore_case);
            let values = vec![
                (d[i - 1][j], MatchOp::Delete),
                (d[i][j - 1], MatchOp::Insert),
                (d[i - 1][j - 1] + (matching as usize),
                 if matching { MatchOp::Match } else { MatchOp::NoMatch }),
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

#[derive(Copy, Clone, Debug)]
enum MatchOp {
    None,
    Delete,
    Insert,
    Match,
    NoMatch,
}

#[pyfunction]
fn match_words_py(
    a: &str,
    b: &str,
    ignore_case: bool,
) -> PyResult<Vec<(usize, usize)>> {
    Ok(match_words(a, b, ignore_case))
}

pub fn batch_match_words(
    a_list: &Vec<&str>,
    b_list: &Vec<&str>,
    ignore_case: bool,
    batch_size: usize,
    show_progress: bool,
) -> Vec<Vec<(usize, usize)>> {
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
                match_words(a, b, ignore_case)
            }).collect::<Vec<Vec<(usize, usize)>>>()
        })
        .flatten()
        .collect()
}

#[pyfunction]
fn batch_match_words_py(
    a_list: Vec<&str>,
    b_list: Vec<&str>,
    ignore_case: bool,
    batch_size: usize,
    show_progress: bool,
) -> PyResult<Vec<Vec<(usize, usize)>>> {
    Ok(batch_match_words(&a_list, &b_list, ignore_case, batch_size, show_progress))
}

pub fn substring(s: &str, start: usize, end: usize) {
    let num_chars = s.chars().count();
    assert!(start < num_chars && end < num_chars && start <= end);
}

pub fn possible_character_substrings(s: &str, max_chars: usize) -> Vec<(usize, usize)> {
    let num_chars = s.chars().count();
    (0..1.max(num_chars - max_chars + 1))
        .map(|i| (i, num_chars.min(i + max_chars)))
        .collect()
}

pub(super) fn add_submodule(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "text")?;
    m.add_function(wrap_pyfunction!(word_boundaries_py, m)?)?;
    m.add_function(wrap_pyfunction!(batch_word_boundaries_py, m)?)?;
    m.add_function(wrap_pyfunction!(clean_py, m)?)?;
    m.add_function(wrap_pyfunction!(batch_clean_py, m)?)?;
    m.add_function(wrap_pyfunction!(match_words_py, m)?)?;
    m.add_function(wrap_pyfunction!(batch_match_words_py, m)?)?;
    parent_module.add_submodule(m)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::text::{clean, match_words, word_boundaries};

    #[test]
    fn test_clean() {
        let text = "  this\t is \n a test sentence  ";
        assert_eq!(clean(text), "this is a test sentence");
    }

    #[test]
    fn test_word_boundaries() {
        let text = "this is a test sentence";
        assert_eq!(
            word_boundaries(text, true),
            vec![(0, 4), (5, 7), (8, 9), (10, 14), (15, 23)]
        );
        let text = "  this\t is \n a test sentence  ";
        assert_eq!(
            word_boundaries(text, true),
            vec![(2, 6), (8, 10), (13, 14), (15, 19), (20, 28)]
        );
    }

    #[test]
    fn test_match_words() {
        let matches = match_words(
            "this is a test",
            "This is also a test",
            false,
        );
        assert_eq!(matches, vec![(1, 1), (2, 3), (3, 4)]);
        let matches = match_words(
            "this is a test",
            "This is also a test",
            true,
        );
        assert_eq!(matches, vec![(0, 0), (1, 1), (2, 3), (3, 4)]);
    }
}
