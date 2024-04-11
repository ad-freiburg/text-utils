use crate::corrupt::{
    edit_word, DeleteEdits, EditsAndWeights, InsertEdits, ReplaceEdits, SwapEdits,
};
use crate::data::{TextData, TextDataInfo};
use crate::dictionary::Dictionary;
use crate::text::{self, split_words};
use crate::text::{possible_byte_substrings, possible_character_substrings};
use crate::unicode::{is_alphabetic, is_punctuation, normalize, Normalization, CS};
use crate::utils::{py_invalid_type_error, py_required_key_error};
use crate::whitespace::{find_substring_ignoring_whitespace, full, remove};
use anyhow::anyhow;
use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use super::utils::{chain, switch};

pub type PreprocessingFn = dyn Send
    + Sync
    + 'static
    + Fn(TextData, TextDataInfo) -> anyhow::Result<(TextData, TextDataInfo)>;

#[derive(PartialEq, Debug, Clone)]
pub enum PreprocessingFnConfig {
    // do nothing
    None,
    // apply multiple processings after each other
    Chain(Vec<PreprocessingFnConfig>),
    // clean the sequences (remove spurious whitespaces)
    Clean(bool),
    // normalize the sequence
    // (using a unicode normalization scheme, see https://en.wikipedia.org/wiki/Unicode_equivalence#Normalization)
    Normalize(Normalization, bool),
    // overwrite target text with input text
    Overwrite,
    // switch between multiple preprocessing functions
    Switch(Vec<PreprocessingFnConfig>, Vec<f64>),
    // delete all whitespaces
    NoWhitespaces(bool),
    // insert whitespaces between all characters
    FullWhitespaces(bool),
    // delete and insert whitespaces with certain probabilities
    WhitespaceCorruption(f64, f64, bool),
    // extract substrings from text
    CharSubstring(usize, bool),
    ByteSubstring(usize, bool),
    // randomly edit and replace words in text
    SpellingCorruption(f64, bool, SpellingCorruptionMode),
    // mark inputs with additional info
    Mark(String, String),
    // add prefix to input sequence
    Prefix(String),
    // decode from json
    JsonDecode(bool, bool),
    // concatenate input and target sequences with a separator
    Concatenate(String),
}

impl<'a> FromPyObject<'a> for PreprocessingFnConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let Some(preprocessing_type) = d.get_item("type")? else {
            return Err(py_required_key_error("type", "preprocessing config"));
        };
        let preprocessing_type: String = preprocessing_type.extract()?;
        let preprocessing_config = match preprocessing_type.as_str() {
            "none" => PreprocessingFnConfig::None,
            "chain" => {
                let Some(configs) = d.get_item("configs")? else {
                    return Err(py_required_key_error("configs", "chain config"));
                };
                let configs: Vec<PreprocessingFnConfig> = configs.extract()?;
                PreprocessingFnConfig::Chain(configs)
            }
            "clean" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes")? {
                    value.extract()?
                } else {
                    true
                };

                PreprocessingFnConfig::Clean(use_graphemes)
            }
            "normalize" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes")? {
                    value.extract()?
                } else {
                    true
                };

                let Some(scheme) = d.get_item("scheme")? else {
                    return Err(py_required_key_error("scheme", "normalization config"));
                };
                PreprocessingFnConfig::Normalize(scheme.extract()?, use_graphemes)
            }
            "overwrite" => PreprocessingFnConfig::Overwrite,
            "no_whitespaces" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes")? {
                    value.extract()?
                } else {
                    true
                };
                PreprocessingFnConfig::NoWhitespaces(use_graphemes)
            }
            "full_whitespaces" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes")? {
                    value.extract()?
                } else {
                    true
                };
                PreprocessingFnConfig::FullWhitespaces(use_graphemes)
            }
            "whitespace_corruption" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes")? {
                    value.extract()?
                } else {
                    true
                };

                let iw_p = if let Some(value) = d.get_item("insert_whitespace_prob")? {
                    value.extract()?
                } else {
                    0.
                };
                let dw_p = if let Some(value) = d.get_item("delete_whitespace_prob")? {
                    value.extract()?
                } else {
                    0.
                };
                PreprocessingFnConfig::WhitespaceCorruption(iw_p, dw_p, use_graphemes)
            }
            "switch" => {
                let Some(configs) = d.get_item("configs")? else {
                    return Err(py_required_key_error("configs", "switch config"));
                };
                let Some(probs) = d.get_item("probabilities")? else {
                    return Err(py_required_key_error("probabilities", "switch config"));
                };
                PreprocessingFnConfig::Switch(configs.extract()?, probs.extract()?)
            }
            "char_substring" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes")? {
                    value.extract()?
                } else {
                    true
                };
                let Some(max_chars) = d.get_item("max_chars")? else {
                    return Err(py_required_key_error("max_chars", "char substring config"));
                };
                PreprocessingFnConfig::CharSubstring(max_chars.extract()?, use_graphemes)
            }
            "byte_substring" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes")? {
                    value.extract()?
                } else {
                    true
                };
                let Some(max_bytes) = d.get_item("max_bytes")? else {
                    return Err(py_required_key_error("max_bytes", "byte substring config"));
                };
                PreprocessingFnConfig::ByteSubstring(max_bytes.extract()?, use_graphemes)
            }
            "spelling_corruption" => {
                let Some(p) = d.get_item("prob")? else {
                    return Err(py_required_key_error("prob", "spelling corruption config"));
                };
                let full_delete = if let Some(value) = d.get_item("allow_full_delete")? {
                    value.extract()?
                } else {
                    true
                };
                let Some(mode) = d.get_item("mode")? else {
                    return Err(py_required_key_error("mode", "spelling corruption config"));
                };
                PreprocessingFnConfig::SpellingCorruption(
                    p.extract()?,
                    full_delete,
                    mode.extract()?,
                )
            }
            "mark" => {
                let Some(key) = d.get_item("key")? else {
                    return Err(py_required_key_error("key", "mark config"));
                };
                let Some(value) = d.get_item("value")? else {
                    return Err(py_required_key_error("value", "mark config"));
                };
                PreprocessingFnConfig::Mark(key.extract()?, value.extract()?)
            }
            "prefix" => {
                let Some(prefix) = d.get_item("prefix")? else {
                    return Err(py_required_key_error("prefix", "prefix config"));
                };
                PreprocessingFnConfig::Prefix(prefix.extract()?)
            }
            "json_decode" => {
                let decode_input = d
                    .get_item("input")?
                    .map(|value| value.extract())
                    .transpose()?
                    .unwrap_or(false);
                let decode_target = d
                    .get_item("target")?
                    .map(|value| value.extract())
                    .transpose()?
                    .unwrap_or(false);
                PreprocessingFnConfig::JsonDecode(decode_input, decode_target)
            }
            "concatenate" => {
                let separator = if let Some(sep) = d.get_item("separator")? {
                    sep.extract()?
                } else {
                    "".to_string()
                };
                PreprocessingFnConfig::Concatenate(separator)
            }
            k => {
                return Err(py_invalid_type_error(k, "preprocessing"));
            }
        };
        Ok(preprocessing_config)
    }
}

fn apply_to_text<F: Fn(&str) -> anyhow::Result<String> + Send + Sync + 'static>(
    f: F,
) -> Box<PreprocessingFn> {
    Box::new(move |item, info| {
        Ok((
            TextData {
                input: f(&item.input)?,
                ..item
            },
            info,
        ))
    })
}

fn overwrite_target_from_input() -> Box<PreprocessingFn> {
    Box::new(|item, info| {
        Ok((
            TextData {
                target: item.input.clone(),
                ..item
            },
            info,
        ))
    })
}

fn corrupt_whitespace(iw_p: f64, dw_p: f64, use_graphemes: bool) -> Box<PreprocessingFn> {
    let iw_p = iw_p.clamp(0., 1.);
    let dw_p = dw_p.clamp(0., 1.);
    assert!(
        iw_p > 0. || dw_p > 0.,
        "at least one of insert whitespace or delete whitespace probability must be greater 0"
    );
    Box::new(move |item, info| {
        let mut rng = ChaCha8Rng::seed_from_u64(info.seed);
        let cs = CS::new(&item.input, use_graphemes);
        let input = cs
            .chars()
            .enumerate()
            .map(|(idx, c)| {
                let r: f64 = rng.gen();
                if c.is_whitespace() {
                    if r < dw_p {
                        "".to_string()
                    } else {
                        c.str.to_string()
                    }
                } else if r < iw_p && idx > 0 && !cs.get_char(idx - 1).unwrap().is_whitespace() {
                    " ".to_string() + c.str
                } else {
                    c.str.to_string()
                }
            })
            .join("");
        Ok((TextData { input, ..item }, info))
    })
}

fn substring<F: Fn(&str) -> anyhow::Result<Vec<(usize, usize, usize)>> + Send + Sync + 'static>(
    name: String,
    substring_fn: F,
    use_graphemes: bool,
) -> Box<PreprocessingFn> {
    Box::new(move |item, info| {
        let mut rng = ChaCha8Rng::seed_from_u64(info.seed);
        let possible_substrings = substring_fn(&item.input)?;
        let idx = rng.gen_range(0..possible_substrings.len());
        let (start, end, _) = possible_substrings[idx];
        let input = item.input[start..end].to_string();
        let target = match find_substring_ignoring_whitespace(&item.target, &input, use_graphemes) {
            Some(s) => s.trim().to_string(),
            None => {
                return Err(anyhow!(
                    "input and target sequences can only differ in \
            whitespaces when applying the {} substring preprocessing",
                    name
                ))
            }
        };
        Ok((TextData { target, input }, info))
    })
}

fn char_substring(max_length: usize, use_graphemes: bool) -> Box<PreprocessingFn> {
    substring(
        "character".to_string(),
        move |s| Ok(possible_character_substrings(s, max_length, use_graphemes)),
        use_graphemes,
    )
}

fn byte_substring(max_length: usize, use_graphemes: bool) -> Box<PreprocessingFn> {
    substring(
        "byte".to_string(),
        move |s| Ok(possible_byte_substrings(s, max_length, use_graphemes)),
        use_graphemes,
    )
}

#[derive(PartialEq, Debug, Clone)]
pub enum SpellingCorruptionMode {
    Artificial(f64, f64, Option<PathBuf>),
    Realistic(PathBuf),
    Mixed(f64, f64, f64, Option<PathBuf>, PathBuf),
}

impl IntoPy<PyObject> for SpellingCorruptionMode {
    fn into_py(self, _py: Python<'_>) -> PyObject {
        todo!()
    }
}

impl<'a> FromPyObject<'a> for SpellingCorruptionMode {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let Some(corruption_type) = d.get_item("type")? else {
            return Err(py_required_key_error("type", "spelling corruption mode"));
        };
        let corruption_type: String = corruption_type.extract()?;
        let corruption_mode = match corruption_type.as_str() {
            "artificial" => {
                let Some(prob) = d.get_item("char_edit_prob")? else {
                    return Err(py_required_key_error(
                        "char_edit_prob",
                        "artificial spelling corruption mode",
                    ));
                };
                let prob: f64 = prob.extract()?;
                let path = d
                    .get_item("characters_file")?
                    .map(|path| path.extract().expect("characters_file must be a string"));
                let temp = d
                    .get_item("temperature")?
                    .map(|temp| temp.extract().expect("temperature must be a float"))
                    .unwrap_or(2.0);
                SpellingCorruptionMode::Artificial(prob, temp, path)
            }
            "realistic" => {
                let Some(path) = d.get_item("misspellings_file")? else {
                    return Err(py_required_key_error(
                        "misspellings_file",
                        "realistic spelling corruption mode",
                    ));
                };
                SpellingCorruptionMode::Realistic(path.extract()?)
            }
            "mixed" => {
                let Some(prob) = d.get_item("char_edit_prob")? else {
                    return Err(py_required_key_error(
                        "char_edit_prob",
                        "mixed spelling corruption mode",
                    ));
                };
                let prob: f64 = prob.extract()?;
                let Some(art_prob) = d.get_item("artificial_prob")? else {
                    return Err(py_required_key_error(
                        "artificial_prob",
                        "mixed spelling corruption mode",
                    ));
                };
                let art_prob: f64 = art_prob.extract()?;
                let art_temp = d
                    .get_item("artificial_temperature")?
                    .map(|temp| temp.extract().expect("temperature must be a float"))
                    .unwrap_or(2.0);
                let char_path = d
                    .get_item("characters_file")?
                    .map(|path| path.extract().expect("characters_file must be a string"));
                let Some(missp_path) = d.get_item("misspellings_file")? else {
                    return Err(py_required_key_error(
                        "misspellings_file",
                        "mixed spelling corruption mode",
                    ));
                };
                SpellingCorruptionMode::Mixed(
                    art_prob,
                    prob,
                    art_temp,
                    char_path,
                    missp_path.extract()?,
                )
            }
            k => {
                return Err(py_invalid_type_error(k, "spelling corruption mode"));
            }
        };
        Ok(corruption_mode)
    }
}

fn corrupt_spelling(
    prob: f64,
    allow_full_delete: bool,
    mode: SpellingCorruptionMode,
) -> Box<PreprocessingFn> {
    let prob = prob.clamp(0.0, 1.0);
    assert!(prob > 0.0, "corruption probability must be greater than 0");
    // extract the probabilities for each corruption mode
    let (art_p, real_p, art_char_edit_p, art_temp) = match mode {
        SpellingCorruptionMode::Artificial(art_char_p, art_temp, _) => {
            (prob, 0.0, art_char_p, art_temp)
        }
        SpellingCorruptionMode::Realistic(..) => (0.0, prob, 0.0, 1.0),
        SpellingCorruptionMode::Mixed(art_p, art_char_p, art_temp, ..) => {
            let art_p = art_p.clamp(0.0, 1.0);
            (art_p * prob, (1.0 - art_p) * prob, art_char_p, art_temp)
        }
    };
    // only delete alphabetic chars or punctuation
    fn can_delete(s: &str) -> bool {
        is_alphabetic(s) || is_punctuation(s)
    }
    let delete = DeleteEdits {
        full_delete: allow_full_delete,
        can_delete,
    };
    // only swap alphabetic chars
    fn can_swap(a: &str, b: &str) -> bool {
        is_alphabetic(a) && is_alphabetic(b)
    }
    let swap = SwapEdits { can_swap };
    // character n-grams we use for insertions and replacements
    // must have a minimum relative frequency of 0.01%
    let min_rel_freq = 1.0 / 10_000.0;
    let (insertions, replacements) = match &mode {
        SpellingCorruptionMode::Artificial(.., Some(char_file))
        | SpellingCorruptionMode::Mixed(.., Some(char_file), _) => {
            let dict = Dictionary::load(char_file).expect("could not load character dictionary");
            let total_freq = dict.freq_sum as f64;
            let mut insertions = HashMap::new();
            let mut replacements = HashMap::new();
            dict.items()
                .sorted_by_key(|&(_, freq)| Reverse(freq))
                .filter_map(|(s, freq)| {
                    let freq = *freq as f64;
                    let rel_freq = freq / total_freq;
                    if rel_freq < min_rel_freq {
                        return None;
                    }
                    match s.split_whitespace().collect::<Vec<_>>().as_slice() {
                        &[prev, cur, next] => Some((prev, cur, next, freq)),
                        _ => panic!(
                            "character dictionary must contain 3-grams separated by whitespace, got '{s}'"
                        ),
                    }
                })
                .for_each(|(prev, cur, next, freq)| {
                    let ctx = (Cow::Owned(prev.to_string()), Cow::Owned(next.to_string()));
                    let weight = freq.powf(1.0 / art_temp);
                    let (edits, weights) = insertions
                        .entry(ctx.clone())
                        .or_insert_with(|| (vec![], vec![]));
                    edits.push(cur.to_string());
                    weights.push(weight);
                    let (edits, weights) = replacements
                        .entry(ctx)
                        .or_insert_with(|| (vec![], vec![]));
                    edits.push(cur.to_string());
                    weights.push(weight);
                });
            // filter out items in replacements that match the cur char
            let replacements: HashMap<_, _> = replacements
                .into_iter()
                .flat_map(
                    |((prev, next), (replacements, weights)): (
                        (Cow<str>, Cow<str>),
                        EditsAndWeights,
                    )| {
                        replacements
                            .iter()
                            .enumerate()
                            .filter_map(|(idx, cur)| {
                                let ctx = (
                                    Cow::Owned(prev.to_string()),
                                    Cow::Owned(cur.to_string()),
                                    Cow::Owned(next.to_string()),
                                );
                                let mut filtered_replacements = replacements.clone();
                                let rem = filtered_replacements.remove(idx);
                                assert_eq!(&rem, cur);
                                if filtered_replacements.is_empty() {
                                    None
                                } else {
                                    let mut filtered_weights = weights.clone();
                                    filtered_weights.remove(idx);
                                    Some((ctx, (filtered_replacements, filtered_weights)))
                                }
                            })
                            .collect::<Vec<_>>()
                    },
                )
                .collect();
            (Some(insertions), Some(replacements))
        }
        _ => (None, None),
    };
    let insert = insertions.map(|insertions| InsertEdits { insertions });
    let replace = replacements.map(|replacements| ReplaceEdits { replacements });
    let misspellings = match &mode {
        SpellingCorruptionMode::Realistic(.., file) | SpellingCorruptionMode::Mixed(.., file) => {
            let file = File::open(file).expect("could not read misspellings file");
            let reader = BufReader::new(file);
            let misspellings: HashMap<String, Vec<String>> =
                serde_json::from_reader(reader).expect("failed to parse misspelllings file");
            misspellings
        }
        _ => HashMap::new(),
    };
    Box::new(move |item, info| {
        let mut rng = ChaCha8Rng::seed_from_u64(info.seed);
        let words = split_words(&item.input);
        let words: Vec<_> = words
            .into_iter()
            .filter_map(|(word, parts)| {
                let r: f64 = rng.gen();
                if r > real_p + art_p {
                    return Some(word.to_string());
                } else if r < real_p {
                    if let Some(replacements) = misspellings.get(word) {
                        let idx = rng.gen_range(0..replacements.len());
                        return Some(replacements[idx].to_string());
                    } else if let Some(parts) = parts {
                        let replacable_parts: Vec<_> = parts
                            .iter()
                            .enumerate()
                            .filter_map(|(idx, &(part, _))| {
                                misspellings
                                    .get(part)
                                    .map(|replacements| (idx, replacements))
                            })
                            .collect();
                        if !replacable_parts.is_empty() {
                            let (idx, replacements) =
                                replacable_parts[rng.gen_range(0..replacable_parts.len())];
                            let (part, start) = parts[idx];
                            let replacement = &replacements[rng.gen_range(0..replacements.len())];
                            return Some(
                                word[..start].to_string()
                                    + replacement
                                    + &word[start + part.len()..],
                            );
                        }
                    }
                    // fallback to artificial corruption
                    // if there are no replacements for the word itself or one of its parts
                }
                let word_len = CS::split(word, true).count();
                let num_edits = (0..word_len)
                    .filter(|_| rng.gen::<f64>() < art_char_edit_p)
                    .count()
                    .max(1);
                let mut exclude: HashSet<usize> = HashSet::new();
                let mut word = word.to_string();
                for _ in 0..num_edits {
                    let (new_word, new_exclude) = edit_word(
                        &word,
                        true,
                        &mut rng,
                        insert.as_ref(),
                        Some(&delete),
                        replace.as_ref(),
                        Some(&swap),
                        Some(exclude),
                    );
                    word = new_word;
                    exclude = new_exclude;
                }
                if word.is_empty() {
                    None
                } else {
                    Some(word)
                }
            })
            .collect();
        Ok((
            TextData {
                input: words.join(" "),
                ..item
            },
            info,
        ))
    })
}

pub fn mark(key: String, value: String) -> Box<PreprocessingFn> {
    Box::new(move |item, mut info| {
        info.marks.insert(key.clone(), value.clone());
        Ok((item, info))
    })
}

pub fn concatenate(separator: String) -> Box<PreprocessingFn> {
    Box::new(move |mut item, info| {
        item.input = item.input + &separator + &item.target;
        Ok((item, info))
    })
}

pub fn preprocessing(cfg: PreprocessingFnConfig) -> Box<PreprocessingFn> {
    match cfg {
        PreprocessingFnConfig::None => Box::new(|input, info| Ok((input, info))),
        PreprocessingFnConfig::Chain(configs) => {
            let pfns = configs.into_iter().map(preprocessing).collect::<Vec<_>>();
            chain(pfns)
        }
        PreprocessingFnConfig::Clean(use_g) => apply_to_text(move |s| Ok(text::clean(s, use_g))),
        PreprocessingFnConfig::Overwrite => overwrite_target_from_input(),
        PreprocessingFnConfig::Switch(fns, probs) => {
            let pfns = fns.into_iter().map(preprocessing).collect::<Vec<_>>();
            switch(pfns, probs)
        }
        PreprocessingFnConfig::WhitespaceCorruption(iw_p, dw_p, use_g) => {
            corrupt_whitespace(iw_p, dw_p, use_g)
        }
        PreprocessingFnConfig::NoWhitespaces(use_graphemes) => {
            apply_to_text(move |s| Ok(remove(s, use_graphemes)))
        }
        PreprocessingFnConfig::FullWhitespaces(use_graphemes) => {
            apply_to_text(move |s| Ok(full(s, use_graphemes)))
        }
        PreprocessingFnConfig::CharSubstring(max_chars, use_graphemes) => {
            char_substring(max_chars, use_graphemes)
        }
        PreprocessingFnConfig::ByteSubstring(max_bytes, use_graphemes) => {
            byte_substring(max_bytes, use_graphemes)
        }
        PreprocessingFnConfig::Normalize(scheme, use_graphemes) => {
            apply_to_text(move |s| Ok(normalize(s, scheme, use_graphemes)))
        }
        PreprocessingFnConfig::SpellingCorruption(p, full_del, mode) => {
            corrupt_spelling(p, full_del, mode)
        }
        PreprocessingFnConfig::Mark(key, value) => mark(key, value),
        PreprocessingFnConfig::Prefix(prefix) => apply_to_text(move |s| Ok(prefix.clone() + s)),
        PreprocessingFnConfig::JsonDecode(decode_input, decode_target) => {
            Box::new(move |mut item, info| {
                if decode_input {
                    item.input = serde_json::from_str(&item.input)
                        .map_err(|e| anyhow!("failed to decode input text from json: {}", e))?;
                }
                if decode_target {
                    item.target = serde_json::from_str(&item.target)
                        .map_err(|e| anyhow!("failed to decode target text from json: {}", e))?;
                }
                Ok((item, info))
            })
        }
        PreprocessingFnConfig::Concatenate(separator) => concatenate(separator),
    }
}

#[cfg(test)]
mod tests {
    use crate::data::{TextData, TextDataInfo};

    use super::corrupt_whitespace;

    #[test]
    fn test_corrupt_whitespace() -> anyhow::Result<()> {
        let noise_fn = corrupt_whitespace(0.0, 1.0, true);
        let data = TextData::new("a test".to_string(), None);
        let info = TextDataInfo::default();
        let (noised, _) = noise_fn(data.clone(), info.clone())?;
        assert_eq!(&noised.input, "atest");
        let noise_fn = corrupt_whitespace(1.0, 0.0, true);
        let data = TextData::new("a test".to_string(), None);
        let (noised, _) = noise_fn(data.clone(), info.clone())?;
        assert_eq!(&noised.input, "a t e s t");
        let data = TextData::new("Ginsberǵs".to_string(), None);
        let (noised, _) = noise_fn(data.clone(), info.clone())?;
        assert_eq!(&noised.input, "G i n s b e r ǵ s");
        Ok(())
    }
}
