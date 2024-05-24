use anyhow::anyhow;
use pyo3::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::unicode::{Character, CS};
use crate::utils::{find_subsequences_of_max_size_k, Matrix};

#[pyfunction(signature = (s, use_graphemes = true))]
#[inline]
pub fn clean(s: &str, use_graphemes: bool) -> String {
    let cs = CS::new(s, use_graphemes);
    let mut output = String::new();
    let mut last_was_whitespace = false;
    for char in cs.chars() {
        if char.is_whitespace() {
            last_was_whitespace = true;
            continue;
        } else if last_was_whitespace && !output.is_empty() {
            output.push(' ');
        }
        last_was_whitespace = false;
        // char.is_whitespace() is only true for characters that
        // contain only whitespace unicode code points, so we trim
        // remaining whitespaces here again. it should be enough to trim here because
        // whitespaces should never occurr in the middle of a grapheme cluster with
        // > 2 code points
        output.push_str(char.str.trim());
    }
    output
}

#[pyfunction(signature = (s, use_graphemes = true))]
#[inline]
pub fn word_boundaries(s: &str, use_graphemes: bool) -> Vec<(usize, usize)> {
    let mut boundaries = vec![];
    let mut start: Option<usize> = None;
    let mut num_elements = 0;
    for (idx, char) in CS::new(s, use_graphemes).chars().enumerate() {
        match (char.is_whitespace(), start) {
            (true, Some(start_idx)) => {
                boundaries.push((start_idx, idx));
                start = None;
            }
            (false, None) => start = Some(idx),
            _ => (),
        }
        num_elements += 1;
    }
    // add potential last word (not captured by the for loop above)
    if let Some(start) = start {
        if start < num_elements {
            boundaries.push((start, num_elements));
        }
    }
    boundaries
}

#[inline]
fn str_match_fn(ignore_case: bool) -> impl Fn(&str, &str) -> bool {
    if ignore_case {
        |a: &str, b: &str| a.to_lowercase() == b.to_lowercase()
    } else {
        |a: &str, b: &str| a == b
    }
}

#[inline]
pub fn match_words_with(
    a: &str,
    b: &str,
    word_match_fn: impl Fn(&str, &str) -> bool,
) -> (Vec<(usize, usize)>, usize, usize) {
    let a_words = a.split_ascii_whitespace().collect::<Vec<&str>>();
    let b_words = b.split_ascii_whitespace().collect::<Vec<&str>>();

    let mut d: Matrix<usize> = vec![vec![0; b_words.len() + 1]; a_words.len() + 1];
    let mut ops: Matrix<MatchOp> = vec![vec![MatchOp::None; b_words.len() + 1]; a_words.len() + 1];

    // initialize matrices
    ops[0][0] = MatchOp::NoMatch;
    for op in ops.iter_mut().skip(1) {
        op[0] = MatchOp::Delete;
    }
    for op in ops[0].iter_mut().skip(1) {
        *op = MatchOp::Insert;
    }

    for (a_idx, &a_word) in a_words.iter().enumerate() {
        for (b_idx, &b_word) in b_words.iter().enumerate() {
            // string indices are offset by -1
            let i = a_idx + 1;
            let j = b_idx + 1;

            let matching = word_match_fn(a_word, b_word);
            let values = [
                (d[i - 1][j], MatchOp::Delete),
                (d[i][j - 1], MatchOp::Insert),
                (
                    d[i - 1][j - 1] + (matching as usize),
                    if matching {
                        MatchOp::Match
                    } else {
                        MatchOp::NoMatch
                    },
                ),
            ];

            let (max_value, max_op) = values
                .iter()
                .max_by(|(v_1, _), (v_2, _)| v_1.cmp(v_2))
                .expect("should not happen");
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
            MatchOp::None => {
                panic!("should not happen")
            }
            MatchOp::Delete => {
                i -= 1;
            }
            MatchOp::Insert => {
                j -= 1;
            }
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
    (matches, a_words.len(), b_words.len())
}

#[pyfunction(signature = (a, b, ignore_case = false))]
#[inline]
pub fn match_words(a: &str, b: &str, ignore_case: bool) -> (Vec<(usize, usize)>, usize, usize) {
    match_words_with(a, b, str_match_fn(ignore_case))
}

#[derive(Copy, Clone, Debug)]
enum MatchOp {
    None,
    Delete,
    Insert,
    Match,
    NoMatch,
}

#[inline]
pub fn possible_character_substrings(
    str: &str,
    mut max_chars: usize,
    use_graphemes: bool,
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
            let (start_byte, end_byte) = cs.char_range_to_byte_range(start_char, end_char);
            (start_byte, end_byte, end_char - start_char)
        })
        .collect()
}

#[inline]
pub fn possible_byte_substrings(
    str: &str,
    max_bytes: usize,
    use_graphemes: bool,
) -> Vec<(usize, usize, usize)> {
    if str.is_empty() {
        return vec![(0, 0, 0)];
    }
    let cs = CS::new(str, use_graphemes);
    let chars: Vec<Character> = cs.chars().collect();
    let subsequences = find_subsequences_of_max_size_k(&chars, max_bytes, |sub_chars| {
        sub_chars.iter().map(|c| c.byte_len()).sum()
    });
    // convert character subsequences to byte indices
    subsequences
        .into_iter()
        .map(|(start_char, end_char)| {
            let (start_byte, end_byte) = cs.char_range_to_byte_range(start_char, end_char);
            (start_byte, end_byte, end_char - start_char)
        })
        .collect()
}

pub type WordParts<'a> = Vec<(&'a str, usize)>;
pub fn split_words(s: &str) -> Vec<(&str, Option<WordParts>)> {
    let re = Regex::new(r"\b[\p{Alphabetic}\p{M}\p{Pc}\p{Join_Control}]+\b").unwrap();
    s.split_whitespace()
        .map(|word| {
            let parts: Vec<_> = re
                .find_iter(word)
                .map(|m| (m.as_str(), m.start()))
                .collect();
            (word, if !parts.is_empty() { Some(parts) } else { None })
        })
        .collect()
}

pub(crate) static SPLIT_WORD_WHITESPACE_PATTERN: &str = r"\s+\S+|^\S+";

#[pyfunction(signature = (s, with_leading_whitespace = false))]
pub fn count_words_whitespace(s: &str, with_leading_whitespace: bool) -> HashMap<&str, usize> {
    let mut counts = HashMap::new();
    let pattern = Regex::new(SPLIT_WORD_WHITESPACE_PATTERN).unwrap();
    for word in pattern.find_iter(s) {
        let word = if with_leading_whitespace {
            word.as_str()
        } else {
            word.as_str().trim_start()
        };
        if let Some(count) = counts.get_mut(word) {
            *count += 1;
        } else {
            counts.insert(word, 1);
        }
    }
    counts
}

pub fn file_size(path: impl AsRef<Path>) -> anyhow::Result<(usize, usize)> {
    let metadata = fs::metadata(path.as_ref())?;
    if !metadata.is_file() {
        return Err(anyhow!("{} is not a file", path.as_ref().display()));
    }
    let num_lines = BufReader::new(File::open(path)?).lines().count();
    Ok((num_lines, metadata.len() as usize))
}

#[pyfunction(name = "file_size")]
fn file_size_py(path: &str) -> PyResult<(usize, usize)> {
    let path = Path::new(path);
    let (num_lines, size) = file_size(path)?;
    Ok((num_lines, size))
}

/// A submodule containing useful functions on text, like cleaning text or
/// calculating word boundaries of all words in a text.
pub(super) fn add_submodule(py: Python, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(py, "text")?;
    m.add_function(wrap_pyfunction!(word_boundaries, m.clone())?)?;
    m.add_function(wrap_pyfunction!(clean, m.clone())?)?;
    m.add_function(wrap_pyfunction!(match_words, m.clone())?)?;
    m.add_function(wrap_pyfunction!(file_size_py, m.clone())?)?;
    m.add_function(wrap_pyfunction!(count_words_whitespace, m.clone())?)?;
    parent_module.add_submodule(&m)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::text::{
        clean, count_words_whitespace, match_words, possible_byte_substrings,
        possible_character_substrings, split_words, word_boundaries,
    };
    use std::collections::HashMap;

    #[test]
    fn test_clean() {
        let text = "  this\t is \n a test sentence  ";
        assert_eq!(clean(text, true), "this is a test sentence");
    }

    #[test]
    fn test_split_words() {
        let text = "This is 123 a 'socalled' unit-test!";
        let words = split_words(text);
        assert_eq!(
            words,
            vec![
                ("This", Some(vec![("This", 0)])),
                ("is", Some(vec![("is", 0)])),
                ("123", None),
                ("a", Some(vec![("a", 0)])),
                ("'socalled'", Some(vec![("socalled", 1)])),
                ("unit-test!", Some(vec![("unit", 0), ("test", 5)]))
            ]
        );
    }

    #[test]
    fn test_count_words() {
        let text = "this\t is \n a test sentence, it is  ";
        let counts = count_words_whitespace(text, false);
        let expected: HashMap<_, usize> = HashMap::from_iter(vec![
            ("this", 1),
            ("is", 2),
            ("a", 1),
            ("test", 1),
            ("sentence,", 1),
            ("it", 1),
        ]);
        assert!(
            counts.len() == expected.len()
                && counts
                    .iter()
                    .all(|(k, v)| expected.contains_key(k) && expected[k] == *v)
        );
        let counts = count_words_whitespace(text, true);
        let expected: HashMap<_, usize> = HashMap::from_iter(vec![
            ("this", 1),
            ("\t is", 1),
            (" is", 1),
            (" \n a", 1),
            (" test", 1),
            (" sentence,", 1),
            (" it", 1),
        ]);
        assert!(
            counts.len() == expected.len()
                && counts
                    .iter()
                    .all(|(k, v)| expected.contains_key(k) && expected[k] == *v)
        );
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
        let (matches, a_len, b_len) = match_words("this is a test", "This is also a test", false);
        assert_eq!(matches, vec![(1, 1), (2, 3), (3, 4)]);
        assert_eq!(a_len, 4);
        assert_eq!(b_len, 5);
        let (matches, a_len, b_len) = match_words("this is a test", "This is also a test", true);
        assert_eq!(matches, vec![(0, 0), (1, 1), (2, 3), (3, 4)]);
        assert_eq!(a_len, 4);
        assert_eq!(b_len, 5);
    }

    #[test]
    fn test_possible_character_substrings() {
        let s = "a test";
        let v = possible_character_substrings(s, 4, true);
        assert_eq!(v, vec![(0, 4, 4), (1, 5, 4), (2, 6, 4)]);
        let v = possible_character_substrings(s, 10, true);
        assert_eq!(v, vec![(0, 6, 6)]);
        let s = "a täst";
        let v = possible_character_substrings(s, 4, true);
        assert_eq!(v, vec![(0, 5, 4), (1, 6, 4), (2, 7, 4)]);
        let s = "नमस्ते";
        let v = possible_character_substrings(s, 2, true);
        assert_eq!(v, vec![(0, 6, 2), (3, 12, 2), (6, 18, 2)]);
        let v = possible_character_substrings(s, 2, false);
        assert_eq!(
            v,
            vec![(0, 6, 2), (3, 9, 2), (6, 12, 2), (9, 15, 2), (12, 18, 2)]
        );
        let v = possible_character_substrings("", 4, true);
        assert_eq!(v, vec![(0, 0, 0)]);
    }

    #[test]
    fn test_possible_byte_substrings() {
        let s = "a test";
        let v = possible_byte_substrings(s, 4, true);
        assert_eq!(v, vec![(0, 4, 4), (1, 5, 4), (2, 6, 4)]);
        let v = possible_byte_substrings(s, 10, true);
        assert_eq!(v, vec![(0, 6, 6)]);
        let s = "a täst";
        let v = possible_byte_substrings(s, 4, true);
        assert_eq!(v, vec![(0, 3, 3), (1, 5, 3), (2, 6, 3), (3, 7, 3)]);
        let s = "नमस्ते";
        let v = possible_byte_substrings(s, 2, true);
        assert_eq!(v, vec![]);
        let v = possible_byte_substrings(s, 6, true);
        assert_eq!(v, vec![(0, 6, 2), (6, 12, 1), (12, 18, 1)]);
        let v = possible_byte_substrings(s, 6, false);
        assert_eq!(
            v,
            vec![(0, 6, 2), (3, 9, 2), (6, 12, 2), (9, 15, 2), (12, 18, 2)]
        );
        let v = possible_byte_substrings("", 4, true);
        assert_eq!(v, vec![(0, 0, 0)]);
    }
}
