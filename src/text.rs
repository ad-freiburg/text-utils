use std::collections::HashSet;
use itertools::Itertools;
use pyo3::prelude::*;
use rand::Rng;

use crate::unicode::{CS};
use crate::utils::Matrix;

#[inline]
pub fn clean(s: &str) -> String {
    s.split_whitespace().join(" ")
}

#[pyfunction]
#[pyo3(name = "clean")]
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
#[pyo3(name = "word_boundaries")]
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
#[pyo3(name = "match_words")]
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
) -> Vec<(usize, usize, usize)> {
    if str.is_empty() {
        return vec![(0, 0, 0)];
    }
    let cs = CS::new(str, use_graphemes);
    let num_chars = cs.len();
    max_chars = max_chars.min(num_chars);
    (0..num_chars - max_chars + 1)
        .map(|start_char| {
            let end_char = num_chars.min(start_char + max_chars);
            let (
                start_byte,
                end_byte
            ) = cs.char_range_to_byte_range(start_char, end_char);
            (start_byte, end_byte, end_char - start_char)
        })
        .collect()
}

pub fn possible_byte_substrings(
    str: &str,
    use_graphemes: bool,
    max_bytes: usize,
) -> Vec<(usize, usize, usize)> {
    if str.is_empty() {
        return vec![(0, 0, 0)];
    }
    let cs = CS::new(str, use_graphemes);
    if *cs.cum_cluster_lengths.last().expect("should not happen") <= max_bytes {
        return vec![(0, cs.len(), cs.len())];
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
            let (
                start_byte,
                end_byte
            ) = cs.char_range_to_byte_range(start_char, end_char);
            (start_byte, end_byte, end_char - start_char)
        })
        .collect()
}

#[derive(Clone)]
pub enum CharEdit {
    Insert(Vec<String>),
    Delete,
    Replace(Vec<String>),
    Swap,
}

impl CharEdit {
    pub fn ascii_edits() -> Vec<Self> {
        let ascii: Vec<String> = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            .chars()
            .map(|c| c.to_string())
            .collect();
        vec![
            Self::Insert(ascii.clone()),
            Self::Delete,
            Self::Replace(ascii),
            Self::Swap,
        ]
    }
}

pub fn edit_word<R: Rng>(
    word: &str,
    use_graphemes: bool,
    rng: &mut R,
    edits: &[CharEdit],
    exclude_indices: Option<HashSet<usize>>,
) -> (String, HashSet<usize>) {
    let mut exclude_indices = exclude_indices.unwrap_or(HashSet::new());
    let cs = CS::new(word, use_graphemes);
    assert!(cs.chars().all(|c| !c.is_whitespace()), "edit word should only be called \
    on strings that do not contain whitespace");
    if edits.is_empty() || cs.len() == 0 {
        return (word.to_string(), exclude_indices);
    }
    let edit = &edits[rng.gen_range(0..edits.len())];
    match edit {
        CharEdit::Insert(insertions) if !insertions.is_empty() => {
            let insert_indices: Vec<usize> = (0..=cs.len())
                .filter_map(|idx| {
                    if exclude_indices.contains(&idx)
                        || (idx > 0 && exclude_indices.contains(&(idx - 1))) {
                        None
                    } else {
                        Some(idx)
                    }
                }).collect();
            let insert_str = &insertions[rng.gen_range(0..insertions.len())];
            if insert_indices.is_empty() || insert_str.is_empty() {
                return (cs.str.to_string(), exclude_indices);
            }
            let insert_idx = insert_indices[rng.gen_range(0..insert_indices.len())];
            let insert_len = CS::new(insert_str, use_graphemes).len();
            // we inserted some string, so the length of the word changed
            // adjust excluded indices to the right of the insertion accordingly
            exclude_indices = exclude_indices
                .into_iter()
                .map(|idx| {
                    if idx >= insert_idx {
                        idx + insert_len
                    } else {
                        idx
                    }
                })
                .collect();
            // add newly inserted indices to excluded indices
            for l in 0..insert_len {
                exclude_indices.insert(insert_idx + l);
            }
            (
                cs.sub(0, insert_idx).to_string()
                    + insert_str.as_str()
                    + cs.sub(insert_idx, cs.len()),
                exclude_indices
            )
        }
        CharEdit::Delete if cs.len() > 1 => {
            let delete_indices: Vec<usize> = (0..cs.len())
                .filter_map(|idx| {
                    if exclude_indices.contains(&idx) {
                        None
                    } else {
                        Some(idx)
                    }
                })
                .collect();
            if delete_indices.is_empty() {
                return (cs.str.to_string(), exclude_indices);
            }
            let delete_idx = delete_indices[rng.gen_range(0..delete_indices.len())];
            // we deleted a character, so the length of the word changed
            // adjust the excluded indices to the right of the delete idx accordingly
            exclude_indices = exclude_indices
                .into_iter()
                .map(|idx| {
                    if idx > delete_idx {
                        idx - 1
                    } else {
                        idx
                    }
                })
                .collect();
            (
                cs.sub(0, delete_idx).to_string()
                    + cs.sub(delete_idx + 1, cs.len()),
                exclude_indices
            )
        }
        CharEdit::Replace(replacements) if !replacements.is_empty() => {
            let replace_indices: Vec<usize> = (0..cs.len())
                .filter_map(|idx| {
                    if exclude_indices.contains(&idx) {
                        None
                    } else {
                        Some(idx)
                    }
                })
                .collect();
            if replace_indices.is_empty() {
                return (cs.str.to_string(), exclude_indices);
            }
            let replace_idx = replace_indices[rng.gen_range(0..replace_indices.len())];
            // look for a replacement that is not equal to the current character
            // start at a random idx in the replacement list and go through it at most once
            let mut replacement_idx = rng.gen_range(0..replacements.len());
            let mut replacement = cs.get(replace_idx).to_string();
            for _ in 0..replacements.len() {
                if cs.get(replace_idx) != &replacements[replacement_idx] {
                    replacement = replacements[replacement_idx].clone();
                    break;
                }
                replacement_idx = (replacement_idx + 1) % replacements.len();
            }
            if replacement.is_empty() {
                return (cs.str.to_string(), exclude_indices);
            }
            let replacement_len = CS::new(&replacement, use_graphemes).len();
            // shift all indices that come after the replacement by length of the replacement
            // string - 1
            exclude_indices = exclude_indices
                .into_iter()
                .map(|idx| {
                    if idx > replace_idx {
                        idx + replacement_len - 1
                    } else {
                        idx
                    }
                })
                .collect();
            // add replaced indices to the excluded indices
            for l in 0..replacement_len {
                exclude_indices.insert(replace_idx + l);
            }
            (
                cs.sub(0, replace_idx).to_string()
                    + replacement.as_str()
                    + cs.sub(replace_idx + 1, cs.len()),
                exclude_indices
            )
        }
        CharEdit::Swap if cs.len() > 1 => {
            let swap_indices: Vec<usize> = (0..cs.len() - 1)
                .filter_map(|idx| {
                    if exclude_indices.contains(&idx)
                        || exclude_indices.contains(&(idx + 1)) {
                        None
                    } else {
                        Some(idx)
                    }
                })
                .collect();
            if swap_indices.is_empty() {
                return (cs.str.to_string(), exclude_indices);
            }
            let swap_idx = swap_indices[rng.gen_range(0..swap_indices.len())];
            // length of word did not change, just add the two swapped indices to
            // the excluded indices
            exclude_indices.insert(swap_idx);
            exclude_indices.insert(swap_idx + 1);
            (
                cs.sub(0, swap_idx).to_string()
                    + cs.get(swap_idx + 1)
                    + cs.get(swap_idx)
                    + cs.sub(swap_idx + 2, cs.len()),
                exclude_indices
            )
        }
        _ => (cs.str.to_string(), exclude_indices)
    }
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
        assert_eq!(v, vec![(0, 4, 4), (1, 5, 4), (2, 6, 4)]);
        let v = possible_character_substrings(s, true, 10);
        assert_eq!(v, vec![(0, 6, 6)]);
        let s = "a täst";
        let v = possible_character_substrings(s, true, 4);
        assert_eq!(v, vec![(0, 5, 4), (1, 6, 4), (2, 7, 4)]);
        let s = "नमस्ते";
        let v = possible_character_substrings(s, true, 2);
        assert_eq!(v, vec![(0, 6, 2), (3, 12, 2), (6, 18, 2)]);
        let v = possible_character_substrings(s, false, 2);
        assert_eq!(v, vec![(0, 6, 2), (3, 9, 2), (6, 12, 2), (9, 15, 2), (12, 18, 2)]);
        let v = possible_character_substrings("", true, 4);
        assert_eq!(v, vec![(0, 0, 0)]);
    }

    #[test]
    fn test_possible_byte_substrings() {
        let s = "a test";
        let v = possible_byte_substrings(s, true, 4);
        assert_eq!(v, vec![(0, 4, 4), (1, 5, 4), (2, 6, 4)]);
        let v = possible_byte_substrings(s, true, 10);
        assert_eq!(v, vec![(0, 6, 6)]);
        let s = "a täst";
        let v = possible_byte_substrings(s, true, 4);
        assert_eq!(v, vec![(0, 3, 3), (1, 5, 3), (2, 6, 3), (3, 7, 3)]);
        let s = "नमस्ते";
        let v = possible_byte_substrings(s, true, 2);
        assert_eq!(v, vec![]);
        let v = possible_byte_substrings(s, true, 6);
        assert_eq!(v, vec![(0, 6, 2), (6, 12, 1), (12, 18, 1)]);
        let v = possible_byte_substrings(s, false, 6);
        assert_eq!(v, vec![(0, 6, 2), (3, 9, 2), (6, 12, 2), (9, 15, 2), (12, 18, 2)]);
        let v = possible_byte_substrings("", true, 4);
        assert_eq!(v, vec![(0, 0, 0)]);
    }
}
