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

pub const ASCII_CHARS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
pub const ASCII_LEFT_PUNCT: &str = "([{";
pub const ASCII_DASH: &str = "-";
pub const ASCII_QUOTE: &str = "\"'";
pub const ASCII_RIGHT_PUNCT: &str = ")]}";
pub const ASCII_END_PUNCT: &str = ",.:;?!";
pub const ASCII_OTHER_PUNCT: &str = "~#%&*+\\/|@";

pub fn ascii_chars() -> Vec<String> {
    ASCII_CHARS.chars().map(|c| c.to_string()).collect()
}

pub trait GetEdits {
    fn get_edits<'e>(&'e self, cs: &CS, idx: &usize) -> Vec<&'e str>;
}

pub trait CanEdit {
    fn can_edit(&self, cs: &CS, idx: &usize) -> bool;
}

#[derive(Default, Clone, Debug)]
pub struct PunctuationEdits {
    pub start: Vec<String>,
    pub end: Vec<String>,
    pub conn: Vec<String>,
}

#[derive(Default, Clone, Debug)]
pub struct InsertEdits {
    pub punct: PunctuationEdits,
    pub insertions: Vec<String>,
}

#[derive(Default, Clone, Debug)]
pub struct ReplaceEdits {
    pub punct: PunctuationEdits,
    pub replacements: Vec<String>,
}

impl GetEdits for InsertEdits {
    fn get_edits<'e>(&'e self, cs: &CS, idx: &usize) -> Vec<&'e str> {
        // let s = cs.get(*idx);
        let mut edits = self.insertions.iter().map(|s| s.as_str()).collect();
        edits
    }
}

impl GetEdits for ReplaceEdits {
    fn get_edits<'e>(&'e self, cs: &CS, idx: &usize) -> Vec<&'e str> {
        // let s = cs.get(*idx);
        let mut edits = self.replacements.iter().map(|s| s.as_str()).collect();
        edits
    }
}

pub struct DeleteEdits<F = fn(&str) -> bool> {
    pub full_delete: bool,
    pub can_delete: F,
}

impl<F> CanEdit for DeleteEdits<F>
where
    F: Fn(&str) -> bool,
{
    fn can_edit(&self, cs: &CS, idx: &usize) -> bool {
        if !self.full_delete && cs.len() <= 1 {
            return false;
        }
        let s = cs.get(*idx);
        (self.can_delete)(s)
    }
}

pub struct SwapEdits<F = fn(&str, &str) -> bool> {
    pub can_swap: F,
}

impl<F> CanEdit for SwapEdits<F>
where
    F: Fn(&str, &str) -> bool,
{
    fn can_edit(&self, cs: &CS, idx: &usize) -> bool {
        if cs.len() <= 1 || *idx >= cs.len() - 1 {
            return false;
        }
        let s = cs.get(*idx);
        let s_next = cs.get(idx + 1);
        (self.can_swap)(s, s_next)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn edit_word(
    word: &str,
    use_graphemes: bool,
    rng: &mut impl Rng,
    insert: Option<&impl GetEdits>,
    delete: Option<&impl CanEdit>,
    replace: Option<&impl GetEdits>,
    swap: Option<&impl CanEdit>,
    exclude_indices: Option<HashSet<usize>>,
) -> (String, HashSet<usize>) {
    let mut edit_indices = vec![];
    if insert.is_some() {
        edit_indices.push(0);
    }
    if delete.is_some() {
        edit_indices.push(1);
    }
    if replace.is_some() {
        edit_indices.push(2);
    }
    if swap.is_some() {
        edit_indices.push(3);
    }
    if edit_indices.is_empty() {
        return (word.to_string(), exclude_indices.unwrap_or_default());
    }
    let edit_idx = edit_indices[rng.gen_range(0..edit_indices.len())];

    let mut exclude_indices = exclude_indices.unwrap_or_default();
    let cs = CS::new(word, use_graphemes);

    match edit_idx {
        0 => {
            let insertions: Vec<_> = (0..=cs.len())
                .filter_map(|idx| {
                    let excluded = exclude_indices.contains(&idx)
                        || (idx > 0 && exclude_indices.contains(&(idx - 1)));
                    if excluded {
                        return None;
                    }
                    let insertions = insert.as_ref().unwrap().get_edits(&cs, &idx);
                    if insertions.len() > 0 {
                        Some((idx, insertions))
                    } else {
                        None
                    }
                })
                .collect();
            if insertions.is_empty() {
                return (cs.str.to_string(), exclude_indices);
            }
            let (insert_idx, insertions) = &insertions[rng.gen_range(0..insertions.len())];
            let insertion = insertions[rng.gen_range(0..insertions.len())];
            let insert_len = CS::new(insertion, use_graphemes).len();
            // we inserted some string, so the length of the word changed
            // adjust excluded indices to the right of the insertion accordingly
            exclude_indices = exclude_indices
                .into_iter()
                .map(|idx| {
                    if idx >= *insert_idx {
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
                cs.sub(0, *insert_idx).to_string() + insertion + cs.sub(*insert_idx, cs.len()),
                exclude_indices,
            )
        }
        1 => {
            let delete_indices: Vec<usize> = (0..cs.len())
                .filter(|idx| {
                    let excluded = exclude_indices.contains(idx);
                    let can_delete = delete.as_ref().unwrap().can_edit(&cs, idx);
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
            let replacements: Vec<_> = (0..cs.len())
                .filter_map(|idx| {
                    if exclude_indices.contains(&idx) {
                        return None;
                    }
                    let replacements = replace.as_ref().unwrap().get_edits(&cs, &idx);
                    if replacements.len() > 0 {
                        Some((idx, replacements))
                    } else {
                        None
                    }
                })
                .collect();
            if replacements.is_empty() {
                return (cs.str.to_string(), exclude_indices);
            }
            let (replace_idx, replacements) = &replacements[rng.gen_range(0..replacements.len())];
            let replacement = replacements[rng.gen_range(0..replacements.len())];
            let replacement_len = CS::new(replacement, use_graphemes).len();
            // shift all indices that come after the replacement by length of the replacement
            // string - 1
            exclude_indices = exclude_indices
                .into_iter()
                .map(|idx| {
                    if idx > *replace_idx {
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
                cs.sub(0, *replace_idx).to_string()
                    + replacement
                    + cs.sub(replace_idx + 1, cs.len()),
                exclude_indices,
            )
        }
        3 if cs.len() > 1 => {
            let swap_indices: Vec<usize> = (0..cs.len() - 1)
                .filter(|idx| {
                    let excluded =
                        exclude_indices.contains(idx) || exclude_indices.contains(&(idx + 1));
                    let can_swap = swap.as_ref().unwrap().can_edit(&cs, idx);
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
        corrupt::{edit_word, DeleteEdits, InsertEdits, ReplaceEdits, SwapEdits},
        unicode::CS,
    };

    #[test]
    fn test_edit_word() {
        let insert = InsertEdits {
            insertions: vec!["w".to_string()],
            ..Default::default()
        };
        let replace = ReplaceEdits {
            replacements: vec!["bla".to_string()],
            ..Default::default()
        };
        fn can_delete(_: &str) -> bool {
            true
        }
        let delete = DeleteEdits {
            full_delete: false,
            can_delete,
        };
        fn can_swap(_: &str, _: &str) -> bool {
            true
        }
        let swap = SwapEdits { can_swap };
        let w = "täst";
        let mut rng = ChaCha8Rng::from_entropy();
        // test deletion of characters
        let (ew, excluded) = edit_word(
            w,
            true,
            &mut rng,
            None as Option<&InsertEdits>,
            Some(&delete),
            None as Option<&ReplaceEdits>,
            None as Option<&SwapEdits>,
            None,
        );
        assert!(ew.len() < w.len());
        assert_eq!(excluded.len(), 0);
        // test with excluded indices --> ä should be removed
        let (ew, _) = edit_word(
            w,
            true,
            &mut rng,
            None as Option<&InsertEdits>,
            Some(&delete),
            None as Option<&ReplaceEdits>,
            None as Option<&SwapEdits>,
            Some(HashSet::from([0, 2, 3])),
        );
        assert_eq!(&ew, "tst");
        // test deletion for word with 1 or fewer characters
        let (ew, excluded) = edit_word(
            "t",
            true,
            &mut rng,
            None as Option<&InsertEdits>,
            Some(&delete),
            None as Option<&ReplaceEdits>,
            None as Option<&SwapEdits>,
            None,
        );
        assert_eq!(ew.len(), 1);
        assert_eq!(excluded.len(), 0);
        // test insertion of characters
        let (ew, excluded) = edit_word(
            w,
            true,
            &mut rng,
            Some(&insert),
            None as Option<&DeleteEdits>,
            None as Option<&ReplaceEdits>,
            None as Option<&SwapEdits>,
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
            Some(&insert),
            None as Option<&DeleteEdits>,
            None as Option<&ReplaceEdits>,
            None as Option<&SwapEdits>,
            Some(HashSet::from([1, 2, 3])),
        );
        assert_eq!(&ew, "wtäst");
        assert_eq!(excluded, HashSet::from([0, 2, 3, 4]));
        // test swapping of characters
        let (ew, excluded) = edit_word(
            w,
            true,
            &mut rng,
            None as Option<&InsertEdits>,
            None as Option<&DeleteEdits>,
            None as Option<&ReplaceEdits>,
            Some(&swap),
            None,
        );
        assert_eq!(ew.len(), w.len());
        assert_eq!(excluded.len(), 2);
        // test with excluded indices --> ä should be swapped with s
        let (ew, excluded) = edit_word(
            w,
            true,
            &mut rng,
            None as Option<&InsertEdits>,
            None as Option<&DeleteEdits>,
            None as Option<&ReplaceEdits>,
            Some(&swap),
            Some(HashSet::from([0, 3])),
        );
        assert_eq!(&ew, "tsät");
        assert_eq!(excluded, HashSet::from([0, 1, 2, 3]));
        // test replacement of characters
        let (ew, excluded) = edit_word(
            w,
            true,
            &mut rng,
            None as Option<&InsertEdits>,
            None as Option<&DeleteEdits>,
            Some(&replace),
            None as Option<&SwapEdits>,
            None,
        );
        assert!(ew.len() > w.len());
        assert_eq!(excluded.len(), 3);
        // test with excluded indices --> s should be replaced with bla
        let (ew, excluded) = edit_word(
            w,
            true,
            &mut rng,
            None as Option<&InsertEdits>,
            None as Option<&DeleteEdits>,
            Some(&replace),
            None as Option<&SwapEdits>,
            Some(HashSet::from([0, 1, 3])),
        );
        assert_eq!(&ew, "täblat");
        assert_eq!(excluded, HashSet::from([0, 1, 2, 3, 4, 5]));
    }
}
