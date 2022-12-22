use pyo3::prelude::*;
use rand::Rng;
use std::collections::HashSet;

use crate::unicode::{Character, CS};
use crate::utils::{find_subsequences_of_max_size_k, Matrix};

#[pyfunction]
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

#[pyfunction(use_graphemes = "true")]
#[inline]
pub fn word_boundaries(str: &str, use_graphemes: bool) -> Vec<(usize, usize)> {
    let mut boundaries = vec![];
    let mut start: Option<usize> = None;
    let mut num_elements = 0;
    for (idx, char) in CS::new(str, use_graphemes).chars().enumerate() {
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
    if start.is_some() && start.unwrap() < num_elements {
        boundaries.push((start.unwrap(), num_elements));
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
            let values = vec![
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

#[pyfunction(ignore_case = "false")]
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

pub fn edit_word(
    word: &str,
    use_graphemes: bool,
    rng: &mut impl Rng,
    edits: &[CharEdit],
    exclude_indices: Option<HashSet<usize>>,
) -> (String, HashSet<usize>) {
    let mut exclude_indices = exclude_indices.unwrap_or_default();
    let cs = CS::new(word, use_graphemes);
    assert!(
        cs.chars().all(|c| !c.is_whitespace()),
        "edit word should only be called \
    on strings that do not contain whitespace"
    );
    if edits.is_empty() || cs.len() == 0 {
        return (word.to_string(), exclude_indices);
    }
    let edit = &edits[rng.gen_range(0..edits.len())];
    match edit {
        CharEdit::Insert(insertions) if !insertions.is_empty() => {
            let insert_indices: Vec<usize> = (0..=cs.len())
                .filter(|idx| {
                    !exclude_indices.contains(idx)
                        && (*idx == 0 || !exclude_indices.contains(&(idx - 1)))
                })
                .collect();
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
                exclude_indices,
            )
        }
        CharEdit::Delete if cs.len() > 1 => {
            let delete_indices: Vec<usize> = (0..cs.len())
                .filter(|idx| !exclude_indices.contains(idx))
                .collect();
            if delete_indices.is_empty() {
                return (cs.str.to_string(), exclude_indices);
            }
            let delete_idx = delete_indices[rng.gen_range(0..delete_indices.len())];
            // we deleted a character, so the length of the word changed
            // adjust the excluded indices to the right of the delete idx accordingly
            exclude_indices = exclude_indices
                .into_iter()
                .map(|idx| if idx > delete_idx { idx - 1 } else { idx })
                .collect();
            (
                cs.sub(0, delete_idx).to_string() + cs.sub(delete_idx + 1, cs.len()),
                exclude_indices,
            )
        }
        CharEdit::Replace(replacements) if !replacements.is_empty() => {
            let replace_indices: Vec<usize> = (0..cs.len())
                .filter(|idx| !exclude_indices.contains(idx))
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
                if cs.get(replace_idx) != replacements[replacement_idx] {
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
                exclude_indices,
            )
        }
        CharEdit::Swap if cs.len() > 1 => {
            let swap_indices: Vec<usize> = (0..cs.len() - 1)
                .filter(|idx| {
                    !exclude_indices.contains(idx) && !exclude_indices.contains(&(idx + 1))
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
                exclude_indices,
            )
        }
        _ => (cs.str.to_string(), exclude_indices),
    }
}

/// A submodule containing useful functions on text, like cleaning text or
/// calculating word boundaries of all words in a text.
pub(super) fn add_submodule(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "text")?;
    m.add_function(wrap_pyfunction!(word_boundaries, m)?)?;
    m.add_function(wrap_pyfunction!(clean, m)?)?;
    m.add_function(wrap_pyfunction!(match_words, m)?)?;
    parent_module.add_submodule(m)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::text::CharEdit::{Delete, Insert, Replace, Swap};
    use crate::text::{
        clean, edit_word, match_words, possible_byte_substrings, possible_character_substrings,
        word_boundaries,
    };
    use crate::unicode::CS;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::HashSet;

    #[test]
    fn test_clean() {
        let text = "  this\t is \n a test sentence  ";
        assert_eq!(clean(text, true), "this is a test sentence");
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

    #[test]
    fn test_edit_word() {
        let w = "täst";
        let mut rng = ChaCha8Rng::from_entropy();
        // test deletion of characters
        let edits = vec![Delete];
        let (ew, excluded) = edit_word(w, true, &mut rng, &edits, None);
        assert!(ew.len() < w.len());
        assert_eq!(excluded.len(), 0);
        // test with excluded indices --> ä should be removed
        let (ew, _) = edit_word(w, true, &mut rng, &edits, Some(HashSet::from([0, 2, 3])));
        assert_eq!(&ew, "tst");
        // test deletion for word with 1 or fewer characters
        let (ew, excluded) = edit_word("t", true, &mut rng, &edits, None);
        assert_eq!(ew.len(), 1);
        assert_eq!(excluded.len(), 0);
        // test insertion of characters
        let edits = vec![Insert(vec!["w".to_string()])];
        let (ew, excluded) = edit_word(w, true, &mut rng, &edits, None);
        assert!(ew.len() > w.len());
        assert_eq!(excluded.len(), 1);
        let ex_idx = *excluded
            .into_iter()
            .collect::<Vec<usize>>()
            .first()
            .unwrap();
        assert_eq!(CS::new(&ew, true).get(ex_idx), "w");
        // test with excluded indices --> w should be inserted at beginning
        let (ew, excluded) = edit_word(w, true, &mut rng, &edits, Some(HashSet::from([1, 2, 3])));
        assert_eq!(&ew, "wtäst");
        assert_eq!(excluded, HashSet::from([0, 2, 3, 4]));
        // test swapping of characters
        let edits = vec![Swap];
        let (ew, excluded) = edit_word(w, true, &mut rng, &edits, None);
        assert_eq!(ew.len(), w.len());
        assert_eq!(excluded.len(), 2);
        // test with excluded indices --> ä should be swapped with s
        let (ew, excluded) = edit_word(w, true, &mut rng, &edits, Some(HashSet::from([0, 3])));
        assert_eq!(&ew, "tsät");
        assert_eq!(excluded, HashSet::from([0, 1, 2, 3]));
        // test replacement of characters
        let edits = vec![Replace(vec!["bla".to_string()])];
        let (ew, excluded) = edit_word(w, true, &mut rng, &edits, None);
        assert!(ew.len() > w.len());
        assert_eq!(excluded.len(), 3);
        // test with excluded indices --> s should be replaced with bla
        let (ew, excluded) = edit_word(w, true, &mut rng, &edits, Some(HashSet::from([0, 1, 3])));
        assert_eq!(&ew, "täblat");
        assert_eq!(excluded, HashSet::from([0, 1, 2, 3, 4, 5]));
    }
}
