use itertools::Itertools;
use pyo3::prelude::*;
use crate::unicode::CS;

use crate::utils::Matrix;

#[inline]
pub fn clean(s: &str) -> String {
    s.split_whitespace().join(" ")
}

#[pyfunction]
fn clean_py(s: &str) -> PyResult<String> {
    Ok(clean(s))
}

pub fn word_boundaries(str: &str, use_graphemes: bool) -> Vec<(usize, usize)> {
    let mut boundaries = vec![];
    let mut start: Option<usize> = None;
    let mut num_elements = 0;
    for (idx, char) in CS::new(str, use_graphemes)
        .chars()
        .enumerate() {
        match (char.is_whitespace(), start) {
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
fn word_boundaries_py(s: &str, use_graphemes: bool) -> PyResult<Vec<(usize, usize)>> {
    Ok(word_boundaries(s, use_graphemes))
}

#[inline]
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

pub fn possible_character_substrings(
    str: &str,
    use_graphemes: bool,
    mut max_chars: usize,
) -> Vec<(usize, usize)> {
    if str.is_empty() {
        return vec![(0, 0)];
    }
    let cs = CS::new(str, use_graphemes);
    let num_chars = cs.len();
    max_chars = max_chars.min(num_chars);
    (0..num_chars - max_chars + 1)
        .map(|start_char| {
            let end_char = num_chars.min(start_char + max_chars);
            let start_byte = cs.cum_cluster_lengths[start_char] - cs.cluster_lengths[start_char];
            let end_byte = cs.cum_cluster_lengths[end_char - 1];
            (start_byte, end_byte)
        })
        .collect()
}

pub fn possible_byte_substrings(
    str: &str,
    use_graphemes: bool,
    max_bytes: usize,
) -> Vec<(usize, usize)> {
    if str.is_empty() {
        return vec![(0, 0)];
    }
    let cs = CS::new(str, use_graphemes);
    if *cs.cum_cluster_lengths.last().expect("should not happen") <= max_bytes {
        return vec![(0, cs.len())];
    }
    let mut start = 0;
    while start < cs.len() && cs.cluster_lengths[start] > max_bytes {
        start += 1;
    }
    if start >= cs.len() {
        return vec![];
    }
    let mut end = start;
    let mut substrings = vec![];
    while start < cs.len() && end < cs.len() {
        let next_end_v = if end + 1 < cs.len() { cs.cluster_lengths[end + 1] } else { 0 };
        if next_end_v > max_bytes {
            substrings.push((start, end + 1));
            start = end + 2;
            end = start;
        } else {
            let cum_next_end_v = cs.cum_cluster_lengths[end] + next_end_v;
            let cum_up_to_start = cs.cum_cluster_lengths[start] - cs.cluster_lengths[start];
            if cum_next_end_v - cum_up_to_start > max_bytes {
                if substrings.is_empty() || substrings.last().unwrap().1 < end + 1 {
                    substrings.push((start, end + 1));
                }
                start += 1;
            } else {
                end += 1;
            }
        }
    }
    if start != end {
        substrings.push((start, end))
    }
    // convert character substrings to byte indices
    substrings
        .into_iter()
        .map(|(start_char, end_char)| {
            let start_byte = cs.cum_cluster_lengths[start_char] - cs.cluster_lengths[start_char];
            let end_byte = cs.cum_cluster_lengths[end_char - 1];
            (start_byte, end_byte)
        })
        .collect()
}

pub(super) fn add_submodule(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "text")?;
    m.add_function(wrap_pyfunction!(word_boundaries_py, m)?)?;
    m.add_function(wrap_pyfunction!(clean_py, m)?)?;
    m.add_function(wrap_pyfunction!(match_words_py, m)?)?;
    parent_module.add_submodule(m)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::text::{clean, match_words, possible_byte_substrings, possible_character_substrings, word_boundaries};

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

    #[test]
    fn test_possible_character_substrings() {
        let s = "a test";
        let v = possible_character_substrings(s, true, 4);
        assert_eq!(v, vec![(0, 4), (1, 5), (2, 6)]);
        let v = possible_character_substrings(s, true, 10);
        assert_eq!(v, vec![(0, 6)]);
        let s = "a täst";
        let v = possible_character_substrings(s, true, 4);
        assert_eq!(v, vec![(0, 5), (1, 6), (2, 7)]);
        let s = "नमस्ते";
        let v = possible_character_substrings(s, true, 2);
        assert_eq!(v, vec![(0, 6), (3, 12), (6, 18)]);
        let v = possible_character_substrings(s, false, 2);
        assert_eq!(v, vec![(0, 6), (3, 9), (6, 12), (9, 15), (12, 18)]);
        let v = possible_character_substrings("", true, 4);
        assert_eq!(v, vec![(0, 0)]);
    }

    #[test]
    fn test_possible_byte_substrings() {
        let s = "a test";
        let v = possible_byte_substrings(s, true, 4);
        assert_eq!(v, vec![(0, 4), (1, 5), (2, 6)]);
        let v = possible_byte_substrings(s, true, 10);
        assert_eq!(v, vec![(0, 6)]);
        let s = "a täst";
        let v = possible_byte_substrings(s, true, 4);
        assert_eq!(v, vec![(0, 3), (1, 5), (2, 6), (3, 7)]);
        let s = "नमस्ते";
        let v = possible_byte_substrings(s, true, 2);
        assert_eq!(v, vec![]);
        let v = possible_byte_substrings(s, true, 6);
        assert_eq!(v, vec![(0, 6), (6, 12), (12, 18)]);
        let v = possible_byte_substrings(s, false, 6);
        assert_eq!(v, vec![(0, 6), (3, 9), (6, 12), (9, 15), (12, 18)]);
        let v = possible_byte_substrings("", true, 4);
        assert_eq!(v, vec![(0, 0)]);
    }
}
