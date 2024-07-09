use crate::corrupt::{
    edit_word, DeleteEdits, EditsAndWeights, InsertEdits, ReplaceEdits, SwapEdits,
};
use crate::data::{TextDataInfo, TrainData};
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

use super::utils::{chain, switch, Chat, ChatTemplate};

pub type PreprocessingFn = dyn Send
    + Sync
    + 'static
    + Fn(TrainData, TextDataInfo) -> anyhow::Result<(TrainData, TextDataInfo)>;

#[derive(PartialEq, Debug, Clone)]
pub enum Part {
    Input,
    Target,
}

impl<'a> FromPyObject<'a> for Part {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let part: String = ob.extract()?;
        match part.as_str() {
            "input" => Ok(Part::Input),
            "target" => Ok(Part::Target),
            k => Err(py_invalid_type_error(k, "part")),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum PreprocessingFnConfig {
    // do nothing
    None,
    // apply multiple processings after each other
    Chain(Vec<PreprocessingFnConfig>),
    // clean the sequences (remove spurious whitespaces)
    Clean(Part, bool),
    // normalize the sequence
    // (using a unicode normalization scheme, see https://en.wikipedia.org/wiki/Unicode_equivalence#Normalization)
    Normalize(Part, Normalization, bool),
    // overwrite target text with input text
    Overwrite(Part),
    // switch between multiple preprocessing functions
    Switch(Vec<PreprocessingFnConfig>, Vec<f64>),
    // delete all whitespaces
    NoWhitespaces(Part, bool),
    // insert whitespaces between all characters
    FullWhitespaces(Part, bool),
    // delete and insert whitespaces with certain probabilities
    WhitespaceCorruption(Part, f64, f64, bool),
    // extract substrings from text
    CharSubstring(usize, bool),
    ByteSubstring(usize, bool),
    // randomly edit and replace words in text
    SpellingCorruption(Part, f64, bool, SpellingCorruptionMode),
    // mark item with additional info
    Mark(String, String),
    // add prefix to input sequence
    Prefix(Part, String),
    // add suffix to input sequence
    Suffix(Part, String),
    // decode from json string
    JsonDecode(Part),
    // decode from json chat
    ChatDecode(Part, Option<String>, Option<ChatTemplate>),
}

impl<'a> FromPyObject<'a> for PreprocessingFnConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let Some(preprocessing_type) = d.get_item("type")? else {
            return Err(py_required_key_error("type", "preprocessing fn config"));
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
                let Some(part) = d.get_item("part")? else {
                    return Err(py_required_key_error("part", "clean config"));
                };
                PreprocessingFnConfig::Clean(part.extract()?, use_graphemes)
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
                let Some(part) = d.get_item("part")? else {
                    return Err(py_required_key_error("part", "normalization config"));
                };
                PreprocessingFnConfig::Normalize(part.extract()?, scheme.extract()?, use_graphemes)
            }
            "overwrite" => {
                let Some(part) = d.get_item("part")? else {
                    return Err(py_required_key_error("part", "overwrite config"));
                };
                PreprocessingFnConfig::Overwrite(part.extract()?)
            }
            "no_whitespaces" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes")? {
                    value.extract()?
                } else {
                    true
                };
                let Some(part) = d.get_item("part")? else {
                    return Err(py_required_key_error("part", "no whitespaces config"));
                };
                PreprocessingFnConfig::NoWhitespaces(part.extract()?, use_graphemes)
            }
            "full_whitespaces" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes")? {
                    value.extract()?
                } else {
                    true
                };
                let Some(part) = d.get_item("part")? else {
                    return Err(py_required_key_error("part", "full whitespaces config"));
                };
                PreprocessingFnConfig::FullWhitespaces(part.extract()?, use_graphemes)
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
                let Some(part) = d.get_item("part")? else {
                    return Err(py_required_key_error(
                        "part",
                        "whitespace corruption config",
                    ));
                };
                PreprocessingFnConfig::WhitespaceCorruption(
                    part.extract()?,
                    iw_p,
                    dw_p,
                    use_graphemes,
                )
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
                let Some(part) = d.get_item("part")? else {
                    return Err(py_required_key_error("part", "spelling corruption config"));
                };
                PreprocessingFnConfig::SpellingCorruption(
                    part.extract()?,
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
                let Some(part) = d.get_item("part")? else {
                    return Err(py_required_key_error("part", "prefix config"));
                };
                PreprocessingFnConfig::Prefix(part.extract()?, prefix.extract()?)
            }
            "suffix" => {
                let Some(suffix) = d.get_item("suffix")? else {
                    return Err(py_required_key_error("suffix", "suffix config"));
                };
                let Some(part) = d.get_item("part")? else {
                    return Err(py_required_key_error("part", "suffix config"));
                };
                PreprocessingFnConfig::Suffix(part.extract()?, suffix.extract()?)
            }
            "json_decode" => {
                let Some(part) = d.get_item("part")? else {
                    return Err(py_required_key_error("part", "json decode config"));
                };
                PreprocessingFnConfig::JsonDecode(part.extract()?)
            }
            "chat_decode" => {
                let Some(part) = d.get_item("part")? else {
                    return Err(py_required_key_error("part", "chat decode config"));
                };
                let separator = d.get_item("separator")?.map(|s| s.extract()).transpose()?;
                let template = d
                    .get_item("chat_template")?
                    .map(|t| t.extract())
                    .transpose()?;
                PreprocessingFnConfig::ChatDecode(part.extract()?, separator, template)
            }
            k => {
                return Err(py_invalid_type_error(k, "preprocessing fn"));
            }
        };
        Ok(preprocessing_config)
    }
}

pub trait TextFn:
    Fn(&str, &TextDataInfo) -> anyhow::Result<String> + Send + Sync + 'static
{
}

impl<F> TextFn for F where
    F: Fn(&str, &TextDataInfo) -> anyhow::Result<String> + Send + Sync + 'static
{
}

fn apply(part: Part, f: impl TextFn) -> Box<PreprocessingFn> {
    Box::new(move |item, info| {
        let item = match part {
            Part::Input => TrainData {
                input: f(&item.input, &info)?,
                ..item
            },
            Part::Target => TrainData {
                target: f(&item.target, &info)?,
                ..item
            },
        };
        Ok((item, info))
    })
}

fn overwrite(part: Part) -> Box<PreprocessingFn> {
    Box::new(move |item, info| {
        let item = match part {
            Part::Input => TrainData {
                input: item.target.clone(),
                ..item
            },
            Part::Target => TrainData {
                target: item.input.clone(),
                ..item
            },
        };
        Ok((item, info))
    })
}

fn corrupt_whitespace(iw_p: f64, dw_p: f64, use_graphemes: bool) -> Box<dyn TextFn> {
    let iw_p = iw_p.clamp(0., 1.);
    let dw_p = dw_p.clamp(0., 1.);
    assert!(
        iw_p > 0. || dw_p > 0.,
        "at least one of insert whitespace or delete whitespace probability must be greater 0"
    );
    Box::new(move |text, info| {
        let mut rng = ChaCha8Rng::seed_from_u64(info.seed);
        let cs = CS::new(text, use_graphemes);
        let corrupted = cs
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
        Ok(corrupted)
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
        Ok((TrainData { target, input }, info))
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
) -> Box<dyn TextFn> {
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
    Box::new(move |text, info| {
        let mut rng = ChaCha8Rng::seed_from_u64(info.seed);
        let words = split_words(text);
        Ok(words
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
            .join(" "))
    })
}

pub fn mark(key: String, value: String) -> Box<PreprocessingFn> {
    Box::new(move |item, mut info| {
        info.marks.insert(key.clone(), value.clone());
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
        PreprocessingFnConfig::Clean(part, use_g) => {
            apply(part, move |s, _| Ok(text::clean(s, use_g)))
        }
        PreprocessingFnConfig::Overwrite(part) => overwrite(part),
        PreprocessingFnConfig::Switch(fns, probs) => {
            let pfns = fns.into_iter().map(preprocessing).collect::<Vec<_>>();
            switch(pfns, probs)
        }
        PreprocessingFnConfig::WhitespaceCorruption(part, iw_p, dw_p, use_g) => {
            apply(part, corrupt_whitespace(iw_p, dw_p, use_g))
        }
        PreprocessingFnConfig::NoWhitespaces(part, use_graphemes) => {
            apply(part, move |s, _| Ok(remove(s, use_graphemes)))
        }
        PreprocessingFnConfig::FullWhitespaces(part, use_graphemes) => {
            apply(part, move |s, _| Ok(full(s, use_graphemes)))
        }
        PreprocessingFnConfig::CharSubstring(max_chars, use_graphemes) => {
            char_substring(max_chars, use_graphemes)
        }
        PreprocessingFnConfig::ByteSubstring(max_bytes, use_graphemes) => {
            byte_substring(max_bytes, use_graphemes)
        }
        PreprocessingFnConfig::Normalize(part, scheme, use_graphemes) => {
            apply(part, move |s, _| Ok(normalize(s, scheme, use_graphemes)))
        }
        PreprocessingFnConfig::SpellingCorruption(part, p, full_del, mode) => {
            apply(part, corrupt_spelling(p, full_del, mode))
        }
        PreprocessingFnConfig::Mark(key, value) => mark(key, value),
        PreprocessingFnConfig::Prefix(part, prefix) => {
            apply(part, move |s, _| Ok(prefix.clone() + s))
        }
        PreprocessingFnConfig::Suffix(part, suffix) => {
            apply(part, move |s, _| Ok(s.to_string() + &suffix))
        }
        PreprocessingFnConfig::JsonDecode(part) => apply(part, move |s, _| {
            serde_json::from_str(s).map_err(|e| anyhow!("failed to decode string from json: {}", e))
        }),
        PreprocessingFnConfig::ChatDecode(part, separator, template) => apply(part, move |s, _| {
            let chat = serde_json::from_str::<Chat>(s)
                .map_err(|e| anyhow!("failed to decode chat from json: {}", e))?;

            if let Some(template) = &template {
                template.format(&chat)
            } else {
                Ok(chat
                    .into_iter()
                    .map(|m| m.text)
                    .join(separator.as_deref().unwrap_or_default()))
            }
        }),
    }
}

#[cfg(test)]
mod tests {
    use crate::data::{utils::ChatTemplate, TextDataInfo, TrainData};

    use super::{corrupt_whitespace, preprocessing, Part, PreprocessingFnConfig};

    #[test]
    fn test_corrupt_whitespace() -> anyhow::Result<()> {
        let noise_fn = corrupt_whitespace(0.0, 1.0, true);
        let info = TextDataInfo::default();
        let s = "a test";
        let noised = noise_fn(s, &info)?;
        assert_eq!(&noised, "atest");
        let noise_fn = corrupt_whitespace(1.0, 0.0, true);
        let noised = noise_fn(s, &info)?;
        assert_eq!(&noised, "a t e s t");
        let s = "Ginsberǵs";
        let noised = noise_fn(s, &info)?;
        assert_eq!(&noised, "G i n s b e r ǵ s");
        Ok(())
    }

    #[test]
    fn test_chat_decode() {
        let chat = r#"[{"role": "user", "text": "Hello"}, {"role": "bot", "text": "Hi"}]"#;
        let data = TrainData {
            input: chat.to_string(),
            target: "".to_string(),
        };
        let chat_fn = preprocessing(PreprocessingFnConfig::ChatDecode(
            Part::Input,
            Some("\n".to_string()),
            None,
        ));
        let (data, _) = chat_fn(data, TextDataInfo::default()).unwrap();
        assert_eq!(data.input, "Hello\nHi");
        let data = TrainData {
            input: chat.to_string(),
            target: "".to_string(),
        };
        let chat_fn = preprocessing(PreprocessingFnConfig::ChatDecode(
            Part::Input,
            None,
            Some(ChatTemplate {
                start: Some("<start>".to_string()),
                roles: vec![("user", "User: {text}\n"), ("bot", "Bot: {text}")]
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
                end: Some("<end>".to_string()),
            }),
        ));
        let (data, _) = chat_fn(data, TextDataInfo::default()).unwrap();
        assert_eq!(data.input, "<start>User: Hello\nBot: Hi<end>");
    }
}
