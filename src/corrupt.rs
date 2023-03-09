use std::collections::{HashMap, HashSet};

use crate::unicode::CS;
use pyo3::prelude::*;
use rand::Rng;

pub fn replace_word(
    word: &str,
    rng: &mut impl Rng,
    replacements: &HashMap<String, Vec<String>>,
) -> String {
    if let Some(replace) = replacements.get(word) {
        replace[rng.gen_range(0..replace.len())].to_string()
    } else {
        word.to_string()
    }
}

pub type CanEditFn = dyn Fn(&CS, &usize) -> bool;
pub type EditOptionsFn = dyn Fn(&CS, &usize) -> Vec<String>;

const ASCII_CHARS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
const ASCII_LEFT_PUNCT: &str = "\"'<([{";
const ASCII_DASH: &str = "-";
const ASCII_RIGHT_PUNCT: &str = "\"'>)]}";
const ASCII_END_PUNCT: &str = ",.:;?!";
const ASCII_OTHER_PUNCT: &str = "~#%&*+\\/|@";

pub fn alpha_punct_delete_fn(allow_full_delete: bool) -> Box<CanEditFn> {
    Box::new(move |cs, &idx| {
        assert!(!cs.is_empty());
        let char = cs.get_char(idx);
        // only delete punctuation and alphabetic characters
        (allow_full_delete || cs.len() > 1) && (char.is_alphabetic() || char.is_punctuation())
    })
}

pub fn insert_fn(insertions: Vec<String>) -> Box<EditOptionsFn> {
    Box::new(move |_, _| insertions.clone())
}

pub fn ascii_insert_fn() -> Box<EditOptionsFn> {
    Box::new(|cs, &idx| {
        assert!(!cs.is_empty());
        let char = cs.get_char(idx);
        let mut inserts: Vec<_> = Vec::new();
        if idx == 0 && !char.is_punctuation() {
            inserts.extend(ASCII_LEFT_PUNCT.chars().map(|c| c.to_string()));
        } else if idx == cs.len() && !cs.get_char(idx - 1).is_punctuation() {
            inserts.extend(ASCII_RIGHT_PUNCT.chars().map(|c| c.to_string()));
            inserts.extend(ASCII_END_PUNCT.chars().map(|c| c.to_string()));
        } else {
            inserts.extend(ASCII_CHARS.chars().map(|c| c.to_string()));
            inserts.extend(ASCII_OTHER_PUNCT.chars().map(|c| c.to_string()));
            inserts.push(ASCII_DASH.to_string());
        }
        inserts
    })
}

pub fn replace_fn(replacements: Vec<String>) -> Box<EditOptionsFn> {
    Box::new(move |cs, &idx| {
        let char = cs.get_char(idx);
        replacements
            .iter()
            .filter_map(|c| {
                if c != char.str {
                    Some(c.to_string())
                } else {
                    None
                }
            })
            .collect()
    })
}

pub fn ascii_replace_fn() -> Box<EditOptionsFn> {
    Box::new(|cs, &idx| {
        assert!(!cs.is_empty());
        let char = cs.get_char(idx);
        let is_punct = char.is_punctuation();
        let filter = |c: char| {
            let c_str = c.to_string();
            if c_str != char.str {
                Some(c_str)
            } else {
                None
            }
        };

        let mut replacements: Vec<_> = ASCII_CHARS.chars().filter_map(filter).collect();
        if idx == 0 && is_punct {
            replacements.extend(ASCII_LEFT_PUNCT.chars().filter_map(filter));
        } else if idx == cs.len() - 1 && is_punct {
            replacements.extend(ASCII_RIGHT_PUNCT.chars().filter_map(filter));
            replacements.extend(ASCII_END_PUNCT.chars().filter_map(filter));
        } else {
            replacements.extend(ASCII_OTHER_PUNCT.chars().filter_map(filter));
            replacements.push(ASCII_DASH.to_string());
        }
        replacements
    })
}

pub fn alpha_swap_fn() -> Box<CanEditFn> {
    Box::new(|cs, &idx| {
        assert!(cs.len() > 1);
        let char1 = cs.get_char(idx);
        let char2 = cs.get_char(idx + 1);
        // only swap alphabetic characters
        char1.is_alphabetic() && char2.is_alphabetic()
    })
}

pub fn edit_word(
    word: &str,
    use_graphemes: bool,
    rng: &mut impl Rng,
    insert: Option<Box<EditOptionsFn>>,
    delete: Option<Box<CanEditFn>>,
    replace: Option<Box<EditOptionsFn>>,
    swap: Option<Box<CanEditFn>>,
    exclude_indices: Option<HashSet<usize>>,
) -> (String, HashSet<usize>) {
    let mut exclude_indices = exclude_indices.unwrap_or_default();
    let cs = CS::new(word, use_graphemes);
    assert!(
        cs.chars().all(|c| !c.is_whitespace()),
        "edit word should only be called \
    on strings that do not contain whitespace"
    );
    let edits = vec![
        insert.is_some(),
        delete.is_some(),
        replace.is_some(),
        swap.is_some(),
    ];
    if edits.iter().all(|e| !e) || cs.is_empty() {
        return (word.to_string(), exclude_indices);
    }
    let edit_indices: Vec<_> = edits
        .into_iter()
        .enumerate()
        .filter_map(|(idx, e)| if e { Some(idx) } else { None })
        .collect();
    let edit_idx = &edit_indices[rng.gen_range(0..edit_indices.len())];
    match edit_idx {
        0 => {
            let insert_indices: Vec<usize> = (0..=cs.len())
                .filter(|idx| {
                    let excluded = exclude_indices.contains(idx)
                        || (*idx > 0 && exclude_indices.contains(&(idx - 1)));
                    let can_insert = !insert.as_ref().unwrap()(&cs, idx).is_empty();
                    !excluded && can_insert
                })
                .collect();
            if insert_indices.is_empty() {
                return (cs.str.to_string(), exclude_indices);
            }
            let insert_idx = insert_indices[rng.gen_range(0..insert_indices.len())];
            let possible_insertions = insert.as_ref().unwrap()(&cs, &insert_idx);
            assert!(!possible_insertions.is_empty());
            let insert_str = &possible_insertions[rng.gen_range(0..possible_insertions.len())];
            let insert_len = CS::new(insert_str.as_ref(), use_graphemes).len();
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
                    + insert_str.as_ref()
                    + cs.sub(insert_idx, cs.len()),
                exclude_indices,
            )
        }
        1 => {
            let delete_indices: Vec<usize> = (0..cs.len())
                .filter(|idx| {
                    let excluded = exclude_indices.contains(idx);
                    let can_delete = delete.as_ref().unwrap()(&cs, idx);
                    !excluded && can_delete
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
                .map(|idx| if idx > delete_idx { idx - 1 } else { idx })
                .collect();
            (
                cs.sub(0, delete_idx).to_string() + cs.sub(delete_idx + 1, cs.len()),
                exclude_indices,
            )
        }
        2 => {
            let replace_indices: Vec<usize> = (0..cs.len())
                .filter(|idx| {
                    let excluded = exclude_indices.contains(idx);
                    let can_replace = !replace.as_ref().unwrap()(&cs, idx).is_empty();
                    !excluded && can_replace
                })
                .collect();
            if replace_indices.is_empty() {
                return (cs.str.to_string(), exclude_indices);
            }
            let replace_idx = replace_indices[rng.gen_range(0..replace_indices.len())];
            let possible_replacements = replace.as_ref().unwrap()(&cs, &replace_idx);
            assert!(!possible_replacements.is_empty());
            let replacement = &possible_replacements[rng.gen_range(0..possible_replacements.len())];
            let replacement_len = CS::new(replacement.as_ref(), use_graphemes).len();
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
                    + replacement.as_ref()
                    + cs.sub(replace_idx + 1, cs.len()),
                exclude_indices,
            )
        }
        3 if cs.len() > 1 => {
            let swap_indices: Vec<usize> = (0..cs.len() - 1)
                .filter(|idx| {
                    let excluded =
                        exclude_indices.contains(idx) || exclude_indices.contains(&(idx + 1));
                    let can_swap = swap.as_ref().unwrap()(&cs, idx);
                    !excluded && can_swap
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

/// A submodule containing functions to corrupt text
pub(super) fn add_submodule(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "corrupt")?;
    parent_module.add_submodule(m)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use crate::{
        corrupt::{alpha_punct_delete_fn, alpha_swap_fn, edit_word, insert_fn, replace_fn},
        unicode::CS,
    };

    #[test]
    fn test_edit_word() {
        let w = "täst";
        let mut rng = ChaCha8Rng::from_entropy();
        // test deletion of characters
        let (ew, excluded) = edit_word(
            w,
            true,
            &mut rng,
            None,
            Some(alpha_punct_delete_fn(false)),
            None,
            None,
            None,
        );
        assert!(ew.len() < w.len());
        assert_eq!(excluded.len(), 0);
        // test with excluded indices --> ä should be removed
        let (ew, _) = edit_word(
            w,
            true,
            &mut rng,
            None,
            Some(alpha_punct_delete_fn(false)),
            None,
            None,
            Some(HashSet::from([0, 2, 3])),
        );
        assert_eq!(&ew, "tst");
        // test deletion for word with 1 or fewer characters
        let (ew, excluded) = edit_word(
            "t",
            true,
            &mut rng,
            None,
            Some(alpha_punct_delete_fn(false)),
            None,
            None,
            None,
        );
        assert_eq!(ew.len(), 1);
        assert_eq!(excluded.len(), 0);
        // test insertion of characters
        let (ew, excluded) = edit_word(
            w,
            true,
            &mut rng,
            Some(insert_fn(vec!["w".to_string()])),
            None,
            None,
            None,
            None,
        );
        assert!(ew.len() > w.len());
        assert_eq!(excluded.len(), 1);
        let ex_idx = *excluded
            .into_iter()
            .collect::<Vec<usize>>()
            .first()
            .unwrap();
        assert_eq!(CS::new(&ew, true).get(ex_idx), "w");
        // test with excluded indices --> w should be inserted at beginning
        let (ew, excluded) = edit_word(
            w,
            true,
            &mut rng,
            Some(insert_fn(vec!["w".to_string()])),
            None,
            None,
            None,
            Some(HashSet::from([1, 2, 3])),
        );
        assert_eq!(&ew, "wtäst");
        assert_eq!(excluded, HashSet::from([0, 2, 3, 4]));
        // test swapping of characters
        let (ew, excluded) = edit_word(
            w,
            true,
            &mut rng,
            None,
            None,
            None,
            Some(alpha_swap_fn()),
            None,
        );
        assert_eq!(ew.len(), w.len());
        assert_eq!(excluded.len(), 2);
        // test with excluded indices --> ä should be swapped with s
        let (ew, excluded) = edit_word(
            w,
            true,
            &mut rng,
            None,
            None,
            None,
            Some(alpha_swap_fn()),
            Some(HashSet::from([0, 3])),
        );
        assert_eq!(&ew, "tsät");
        assert_eq!(excluded, HashSet::from([0, 1, 2, 3]));
        // test replacement of characters
        let (ew, excluded) = edit_word(
            w,
            true,
            &mut rng,
            None,
            None,
            Some(replace_fn(vec!["bla".to_string()])),
            None,
            None,
        );
        assert!(ew.len() > w.len());
        assert_eq!(excluded.len(), 3);
        // test with excluded indices --> s should be replaced with bla
        let (ew, excluded) = edit_word(
            w,
            true,
            &mut rng,
            None,
            None,
            Some(replace_fn(vec!["bla".to_string()])),
            None,
            Some(HashSet::from([0, 1, 3])),
        );
        assert_eq!(&ew, "täblat");
        assert_eq!(excluded, HashSet::from([0, 1, 2, 3, 4, 5]));
    }
}
