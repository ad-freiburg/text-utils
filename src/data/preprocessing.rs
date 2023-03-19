use crate::corrupt::{
    alpha_punct_delete_fn, alpha_swap_fn, ascii_insert_fn, ascii_replace_fn, edit_word,
};
use crate::data::{Label, TextData};
use crate::dictionary::Dictionary;
use crate::text::{self, split_words};
use crate::text::{possible_byte_substrings, possible_character_substrings};
use crate::tokenization::{tokenizer, TokenizerConfig};
use crate::unicode::{normalize, Normalization, CS};
use crate::utils::{accumulate, py_invalid_type_error, py_required_key_error};
use crate::whitespace::{find_substring_ignoring_whitespace, full, operations, remove};
use anyhow::anyhow;
use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Geometric};
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::ops::Sub;
use std::path::PathBuf;

pub type PreprocessingFn =
    dyn Send + Sync + 'static + Fn(TextData, Option<u64>) -> anyhow::Result<TextData>;
pub type LabelingFn = dyn Send + Sync + 'static + Fn(&TextData) -> anyhow::Result<Label>;

#[derive(PartialEq, Debug, Clone)]
pub enum PreprocessingConfig {
    // do nothing
    None,
    // apply multiple processings after each other
    Chain(Vec<PreprocessingConfig>),
    // clean the sequences (remove spurious whitespaces)
    Clean(bool),
    // normalize the sequence
    // (using a unicode normalization scheme, see https://en.wikipedia.org/wiki/Unicode_equivalence#Normalization)
    Normalize(Normalization, bool),
    // overwrite original text with processed text
    Overwrite,
    // switch between multiple preprocessing functions
    Switch(Vec<PreprocessingConfig>, Vec<f64>),
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
    // randomly mask full words in text
    MaskCorruption(f64, bool, String, MaskMode),
    // randomly replace the language token with the given default
    LanguageDropout(f64),
}

impl IntoPy<PyObject> for PreprocessingConfig {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let d = PyDict::new(py);
        let preprocessing_type = match self {
            PreprocessingConfig::None => "none",
            PreprocessingConfig::Chain(configs) => {
                let py_configs = PyList::empty(py);
                for config in configs {
                    py_configs.append(config.into_py(py)).unwrap();
                }
                d.set_item("configs", py_configs).unwrap();
                "chain"
            }
            PreprocessingConfig::Clean(use_g) => {
                d.set_item("use_graphemes", use_g).unwrap();
                "clean"
            }
            PreprocessingConfig::Overwrite => "overwrite",
            PreprocessingConfig::Switch(configs, probs) => {
                let py_configs = PyList::empty(py);
                for config in configs {
                    py_configs.append(config.into_py(py)).unwrap();
                }
                d.set_item("configs", py_configs).unwrap();
                d.set_item("probabilities", PyList::new(py, probs)).unwrap();
                "switch"
            }
            PreprocessingConfig::NoWhitespaces(use_g) => {
                d.set_item("use_graphemes", use_g).unwrap();
                "no_whitespaces"
            }
            PreprocessingConfig::FullWhitespaces(use_g) => {
                d.set_item("use_graphemes", use_g).unwrap();
                "full_whitespaces"
            }
            PreprocessingConfig::WhitespaceCorruption(iw_p, dw_p, use_g) => {
                d.set_item("insert_whitespace_prob", iw_p).unwrap();
                d.set_item("delete_whitespace_prob", dw_p).unwrap();
                d.set_item("use_graphemes", use_g).unwrap();
                "whitespace_corruption"
            }
            PreprocessingConfig::CharSubstring(max_chars, use_g) => {
                d.set_item("max_chars", max_chars).unwrap();
                d.set_item("use_graphemes", use_g).unwrap();
                "char_substring"
            }
            PreprocessingConfig::ByteSubstring(max_bytes, use_g) => {
                d.set_item("max_bytes", max_bytes).unwrap();
                d.set_item("use_graphemes", use_g).unwrap();
                "byte_substring"
            }
            PreprocessingConfig::Normalize(scheme, use_g) => {
                d.set_item("scheme", scheme.into_py(py)).unwrap();
                d.set_item("use_graphemes", use_g).unwrap();
                "normalize"
            }
            PreprocessingConfig::LanguageDropout(p) => {
                d.set_item("prob", p).unwrap();
                "language_dropout"
            }
            PreprocessingConfig::SpellingCorruption(p, full_del, mode) => {
                d.set_item("prob", p).unwrap();
                d.set_item("allow_full_delete", full_del).unwrap();
                d.set_item("mode", mode.into_py(py)).unwrap();
                "spelling_corruption"
            }
            PreprocessingConfig::MaskCorruption(p, use_g, mask_token, mode) => {
                d.set_item("prob", p).unwrap();
                d.set_item("use_graphemes", use_g).unwrap();
                d.set_item("mask_token", mask_token).unwrap();
                d.set_item("mode", mode.into_py(py)).unwrap();
                "mask_corruption"
            }
        };
        d.set_item("type", preprocessing_type).unwrap();
        d.to_object(py)
    }
}

impl<'a> FromPyObject<'a> for PreprocessingConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let Some(preprocessing_type) = d.get_item("type") else {
            return Err(py_required_key_error("type", "preprocessing config"));
        };
        let preprocessing_type: String = preprocessing_type.extract()?;
        let preprocessing_config = match preprocessing_type.as_str() {
            "none" => PreprocessingConfig::None,
            "chain" => {
                let Some(configs) = d.get_item("configs") else {
                    return Err(py_required_key_error("configs", "chain config"));
                };
                let configs: Vec<PreprocessingConfig> = configs.extract()?;
                PreprocessingConfig::Chain(configs)
            }
            "clean" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };

                PreprocessingConfig::Clean(use_graphemes)
            }
            "normalize" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };

                let Some(scheme) = d.get_item("scheme") else {
                    return Err(py_required_key_error("scheme", "normalization config"));
                };
                PreprocessingConfig::Normalize(scheme.extract()?, use_graphemes)
            }
            "overwrite" => PreprocessingConfig::Overwrite,
            "no_whitespaces" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };
                PreprocessingConfig::NoWhitespaces(use_graphemes)
            }
            "full_whitespaces" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };
                PreprocessingConfig::FullWhitespaces(use_graphemes)
            }
            "whitespace_corruption" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };

                let iw_p = if let Some(value) = d.get_item("insert_whitespace_prob") {
                    value.extract()?
                } else {
                    0.
                };
                let dw_p = if let Some(value) = d.get_item("delete_whitespace_prob") {
                    value.extract()?
                } else {
                    0.
                };
                PreprocessingConfig::WhitespaceCorruption(iw_p, dw_p, use_graphemes)
            }
            "switch" => {
                let Some(configs) = d.get_item("configs") else {
                    return Err(py_required_key_error("configs", "switch config"));
                };
                let Some(probs) = d.get_item("probabilities") else {
                    return Err(py_required_key_error("probabilities", "switch config"));
                };
                PreprocessingConfig::Switch(
                    configs
                        .extract::<&PyList>()?
                        .iter()
                        .map(|any| any.extract())
                        .collect::<PyResult<Vec<PreprocessingConfig>>>()?,
                    probs
                        .extract::<&PyList>()?
                        .iter()
                        .map(|any| any.extract())
                        .collect::<PyResult<Vec<f64>>>()?,
                )
            }
            "char_substring" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };
                let Some(max_chars) = d.get_item("max_chars") else {
                    return Err(py_required_key_error("max_chars", "char substring config"));
                };
                PreprocessingConfig::CharSubstring(max_chars.extract()?, use_graphemes)
            }
            "byte_substring" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };
                let Some(max_bytes) = d.get_item("max_bytes") else {
                    return Err(py_required_key_error("max_bytes", "byte substring config"));
                };
                PreprocessingConfig::ByteSubstring(max_bytes.extract()?, use_graphemes)
            }
            "language_dropout" => {
                let Some(p) = d.get_item("prob") else {
                    return Err(py_required_key_error("prob", "language dropout config"));
                };
                PreprocessingConfig::LanguageDropout(p.extract()?)
            }
            "spelling_corruption" => {
                let Some(p) = d.get_item("prob") else {
                    return Err(py_required_key_error("prob", "spelling corruption config"));
                };
                let full_delete = if let Some(value) = d.get_item("allow_full_delete") {
                    value.extract()?
                } else {
                    true
                };
                let Some(mode) = d.get_item("mode") else {
                    return Err(py_required_key_error("mode", "spelling corruption config"));
                };
                PreprocessingConfig::SpellingCorruption(p.extract()?, full_delete, mode.extract()?)
            }
            "mask_corruption" => {
                let Some(p) = d.get_item("prob") else {
                    return Err(py_required_key_error("prob", "mask corruption config"));
                };
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };
                let Some(mask_token) = d.get_item("mask_token") else {
                    return Err(py_required_key_error("mask_token", "mask corruption config"));
                };
                let Some(mode) = d.get_item("mode") else {
                    return Err(py_required_key_error("mode", "mask corruption config"));
                };
                PreprocessingConfig::MaskCorruption(
                    p.extract()?,
                    use_graphemes,
                    mask_token.extract()?,
                    mode.extract()?,
                )
            }
            k => {
                return Err(py_invalid_type_error(k, "preprocessing"));
            }
        };
        Ok(preprocessing_config)
    }
}

#[derive(Debug, Clone)]
pub enum LabelingConfig {
    // generate whitespace correction labels given processed and original sequence
    WhitespaceCorrection(bool, TokenizerConfig),
    // generate sequence generation labels (basically just the tokenization) of the processed sequence
    SequenceGeneration(TokenizerConfig),
}

impl<'a> FromPyObject<'a> for LabelingConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let Some(labeling_type) = d.get_item("type") else {
            return Err(py_required_key_error("type", "labeling config"));
        };
        let labeling_type: String = labeling_type.extract()?;
        let labeling_config = match labeling_type.as_str() {
            "whitespace_correction" => {
                let use_graphemes = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };
                let Some(tokenizer_config) = d.get_item("tokenizer") else {
                    return Err(py_required_key_error("tokenizer", "whitespace correction config"));
                };
                LabelingConfig::WhitespaceCorrection(use_graphemes, tokenizer_config.extract()?)
            }
            "sequence_generation" => {
                let Some(tokenizer_config) = d.get_item("tokenizer") else {
                    return Err(py_required_key_error("tokenizer", "sequence generation config"));
                };
                LabelingConfig::SequenceGeneration(tokenizer_config.extract()?)
            }
            k => {
                return Err(py_invalid_type_error(k, "labeling"));
            }
        };
        Ok(labeling_config)
    }
}

fn switch(fns: Vec<PreprocessingConfig>, probs: Vec<f64>) -> Box<PreprocessingFn> {
    let num_fns = fns.len();
    assert!(
        num_fns > 0 && num_fns == probs.len(),
        "expected one or more preprocessing for switch preprocessing and the same \
        number of probabilities"
    );
    // generate cumulative probabilities
    let cum_p: Vec<f64> = accumulate(&probs);
    // probabilities should sum to 1
    assert!(
        cum_p.last().copied().unwrap().sub(1f64).abs() < 1e-5,
        "all switch probabilities should sum to 1"
    );

    let fns: Vec<Box<PreprocessingFn>> = fns.into_iter().map(preprocessing_fn).collect();

    // return new function that switches between multiple preprocessing functions
    // based on the given probability distribution
    Box::new(move |item, seed| -> anyhow::Result<TextData> {
        let mut rng = if let Some(seed) = seed {
            ChaCha8Rng::seed_from_u64(seed)
        } else {
            ChaCha8Rng::from_entropy()
        };
        let r: f64 = rng.gen();
        let mut idx = 0;
        while idx < num_fns - 1 && r > cum_p[idx] {
            idx += 1;
        }
        fns[idx](item, seed)
    })
}

fn apply_to_text<F: Fn(&str) -> anyhow::Result<String> + Send + Sync + 'static>(
    f: F,
) -> Box<PreprocessingFn> {
    Box::new(move |item, _| {
        Ok(TextData {
            processed: f(&item.processed)?,
            ..item
        })
    })
}

fn overwrite_original_from_processed() -> Box<PreprocessingFn> {
    Box::new(|item, _| {
        Ok(TextData {
            original: item.processed.clone(),
            ..item
        })
    })
}

fn corrupt_whitespace(iw_p: f64, dw_p: f64, use_graphemes: bool) -> Box<PreprocessingFn> {
    let iw_p = iw_p.clamp(0., 1.);
    let dw_p = dw_p.clamp(0., 1.);
    assert!(
        iw_p > 0. || dw_p > 0.,
        "at least one of insert whitespace or delete whitespace probability must be greater 0"
    );
    Box::new(move |item, seed| {
        let mut rng = if let Some(seed) = seed {
            ChaCha8Rng::seed_from_u64(seed)
        } else {
            ChaCha8Rng::from_entropy()
        };
        let cs = CS::new(&item.processed, use_graphemes);
        let processed = cs
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
                } else if r < iw_p && idx > 0 && !cs.get_char(idx - 1).is_whitespace() {
                    " ".to_string() + c.str
                } else {
                    c.str.to_string()
                }
            })
            .join("");
        Ok(TextData { processed, ..item })
    })
}

fn substring<F: Fn(&str) -> anyhow::Result<Vec<(usize, usize, usize)>> + Send + Sync + 'static>(
    name: String,
    substring_fn: F,
    use_graphemes: bool,
) -> Box<PreprocessingFn> {
    Box::new(move |item, seed| {
        let mut rng = if let Some(seed) = seed {
            ChaCha8Rng::seed_from_u64(seed)
        } else {
            ChaCha8Rng::from_entropy()
        };
        let possible_substrings = substring_fn(&item.processed)?;
        let idx = rng.gen_range(0..possible_substrings.len());
        let (start, end, _) = possible_substrings[idx];
        let processed = item.processed[start..end].to_string();
        let original =
            match find_substring_ignoring_whitespace(&item.original, &processed, use_graphemes) {
                Some(s) => s.trim().to_string(),
                None => {
                    return Err(anyhow!(
                        "original and processed sequences can only differ in \
            whitespaces when applying the {} substring preprocessing",
                        name
                    ))
                }
            };
        Ok(TextData {
            original,
            processed,
            ..item
        })
    })
}

fn char_substring(max_chars: usize, use_graphemes: bool) -> Box<PreprocessingFn> {
    substring(
        "character".to_string(),
        move |s| Ok(possible_character_substrings(s, max_chars, use_graphemes)),
        use_graphemes,
    )
}

fn byte_substring(max_bytes: usize, use_graphemes: bool) -> Box<PreprocessingFn> {
    substring(
        "byte".to_string(),
        move |s| Ok(possible_byte_substrings(s, max_bytes, use_graphemes)),
        use_graphemes,
    )
}

fn language_dropout(prob: f64) -> Box<PreprocessingFn> {
    let prob = prob.clamp(0.0, 1.0);
    Box::new(move |item, seed| {
        let mut rng = if let Some(seed) = seed {
            ChaCha8Rng::seed_from_u64(seed)
        } else {
            ChaCha8Rng::from_entropy()
        };
        let r: f64 = rng.gen();
        if r < prob {
            Ok(TextData {
                language: None,
                ..item
            })
        } else {
            Ok(item)
        }
    })
}

#[derive(PartialEq, Debug, Clone)]
pub enum SpellingCorruptionMode {
    Artificial(f64, Option<PathBuf>),
    Realistic(PathBuf),
    Mixed(f64, f64, Option<PathBuf>, PathBuf),
}

impl IntoPy<PyObject> for SpellingCorruptionMode {
    fn into_py(self, _py: Python<'_>) -> PyObject {
        todo!()
    }
}

impl<'a> FromPyObject<'a> for SpellingCorruptionMode {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let Some(corruption_type) = d.get_item("type") else {
            return Err(py_required_key_error("type", "spelling corruption mode"));
        };
        let corruption_type: String = corruption_type.extract()?;
        let corruption_mode = match corruption_type.as_str() {
            "artificial" => {
                let Some(prob) = d.get_item("char_edit_prob") else {
                    return Err(py_required_key_error("char_edit_prob", "artificial spelling corruption mode"));
                };
                let prob: f64 = prob.extract()?;
                let path = d
                    .get_item("characters_file")
                    .map(|path| path.extract().expect("characters_file must be a string"));
                SpellingCorruptionMode::Artificial(prob, path)
            }
            "realistic" => {
                let Some(path) = d.get_item("misspellings_file") else {
                    return Err(py_required_key_error("misspellings_file", "realistic spelling corruption mode"));
                };
                SpellingCorruptionMode::Realistic(path.extract()?)
            }
            "mixed" => {
                let Some(prob) = d.get_item("char_edit_prob") else {
                    return Err(py_required_key_error("char_edit_prob", "mixed spelling corruption mode"));
                };
                let prob: f64 = prob.extract()?;
                let Some(art_prob) = d.get_item("artificial_prob") else {
                    return Err(py_required_key_error("artificial_prob", "mixed spelling corruption mode"));
                };
                let art_prob: f64 = art_prob.extract()?;
                let char_path = d
                    .get_item("characters_file")
                    .map(|path| path.extract().expect("characters_file must be a string"));
                let Some(missp_path) = d.get_item("misspellings_file") else {
                    return Err(py_required_key_error("misspellings_file", "mixed spelling corruption mode"));
                };
                SpellingCorruptionMode::Mixed(art_prob, prob, char_path, missp_path.extract()?)
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
    let delete = alpha_punct_delete_fn(allow_full_delete);
    let swap = alpha_swap_fn();
    // characters we use for insertions and replacements
    // must have a minimum relative frequency of 0.01%
    let min_rel_freq = 1.0 / 10000.0;
    let (insert, replace) = match &mode {
        SpellingCorruptionMode::Artificial(.., Some(char_file))
        | SpellingCorruptionMode::Mixed(.., Some(char_file), _) => {
            let dict = Dictionary::load(char_file).expect("could not load character dictionary");
            let total_freq = dict.freq_sum as f64;
            let insert_and_replace_chars: Vec<_> = dict
                .items()
                .sorted_by_key(|&(_, freq)| Reverse(freq))
                .filter_map(|(w, freq)| {
                    let rel_freq = *freq as f64 / total_freq;
                    if rel_freq < min_rel_freq {
                        None
                    } else {
                        Some(w.to_string())
                    }
                })
                .collect();
            (
                ascii_insert_fn(Some(insert_and_replace_chars.clone())),
                ascii_replace_fn(Some(insert_and_replace_chars)),
            )
        }
        _ => (ascii_insert_fn(None), ascii_replace_fn(None)),
    };
    let misspellings = match &mode {
        SpellingCorruptionMode::Realistic(file) | SpellingCorruptionMode::Mixed(.., file) => {
            let file = File::open(file).expect("could not read misspellings file");
            let reader = BufReader::new(file);
            let misspellings: HashMap<String, Vec<String>> =
                serde_json::from_reader(reader).expect("failed to parse misspelllings file");
            misspellings
        }
        _ => HashMap::new(),
    };
    let art_char_edit_p = match &mode {
        SpellingCorruptionMode::Artificial(p, ..) | SpellingCorruptionMode::Mixed(_, p, ..) => {
            p.clamp(0.0, 1.0)
        }
        _ => 0.0,
    };
    let (art_p, real_p) = match mode {
        SpellingCorruptionMode::Artificial(..) => (prob, 0.0),
        SpellingCorruptionMode::Realistic(..) => (0.0, prob),
        SpellingCorruptionMode::Mixed(art_p, ..) => {
            let art_p = art_p.clamp(0.0, 1.0);
            (art_p * prob, (1.0 - art_p) * prob)
        }
    };
    Box::new(move |item, seed| {
        let mut rng = if let Some(seed) = seed {
            ChaCha8Rng::seed_from_u64(seed)
        } else {
            ChaCha8Rng::from_entropy()
        };
        let words = split_words(&item.processed);
        let words: Vec<_> = words
            .into_iter()
            .filter_map(|(word, parts)| {
                let mut word = word.to_string();
                let r: f64 = rng.gen();
                if r < real_p {
                    if let Some(replacements) = misspellings.get(&word) {
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
                if r < real_p + art_p {
                    let word_len = CS::split(&word, true).count();
                    let num_edits = (0..word_len)
                        .filter(|_| rng.gen::<f64>() < art_char_edit_p)
                        .count();
                    let mut exclude: HashSet<usize> = HashSet::new();
                    for _ in 0..num_edits {
                        let (new_word, new_exclude) = edit_word(
                            &word,
                            true,
                            &mut rng,
                            Some(&insert),
                            Some(&delete),
                            Some(&replace),
                            Some(&swap),
                            Some(exclude),
                        );
                        if new_word == word {
                            break;
                        }
                        word = new_word;
                        exclude = new_exclude;
                    }
                }
                if word.is_empty() {
                    None
                } else {
                    Some(word)
                }
            })
            .collect();
        Ok(TextData {
            processed: words.join(" "),
            ..item
        })
    })
}

#[derive(PartialEq, Debug, Clone)]
pub enum MaskMode {
    Word(usize, f64),
    Char(usize, f64),
}

impl IntoPy<PyObject> for MaskMode {
    fn into_py(self, py: Python) -> PyObject {
        let d = PyDict::new(py);
        let mask_type = match self {
            MaskMode::Word(min, p) | MaskMode::Char(min, p) => {
                d.set_item("min", min).unwrap();
                d.set_item("prob", p).unwrap();
                match self {
                    MaskMode::Word(..) => "word",
                    MaskMode::Char(..) => "char",
                }
            }
        };
        d.set_item("type", mask_type).unwrap();
        d.into_py(py)
    }
}

impl<'a> FromPyObject<'a> for MaskMode {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let Some(mask_type) = d.get_item("type") else {
            return Err(py_required_key_error("type", "mask mode config"));
        };
        let mask_type: String = mask_type.extract()?;
        let mask_mode_config = match mask_type.as_str() {
            name @ ("word" | "char") => {
                let Some(min) = d.get_item("min") else {
                    return Err(py_required_key_error("min", format!("{name} mask mode config")));
                };
                let Some(p) = d.get_item("prob") else {
                    return Err(py_required_key_error("prob", format!("{name} mask mode config")));
                };
                let min = min.extract()?;
                let p = p.extract()?;
                if name == "word" {
                    MaskMode::Word(min, p)
                } else {
                    MaskMode::Char(min, p)
                }
            }
            k => return Err(py_invalid_type_error(k, "mask mode config")),
        };
        Ok(mask_mode_config)
    }
}

pub fn corrupt_mask(
    mask_p: f64,
    use_graphemes: bool,
    mask_token: String,
    mode: MaskMode,
) -> Box<PreprocessingFn> {
    let (min, expected_per_mask, geo, is_word) = match mode {
        MaskMode::Word(min, p) | MaskMode::Char(min, p) => (
            min,
            min as f64 + 1.0 / p - 1.0,
            Geometric::new(p).expect("failed to create geometric distribution"),
            matches!(mode, MaskMode::Word(..)),
        ),
    };
    assert!(min > 0, "minimum tokens to mask must be greater than 0");
    // adjust mask probability because it refers to the percentage of tokens expected to be masked
    // but we always mask multiple tokens per mask operation
    let mask_p = mask_p / (expected_per_mask - mask_p);
    Box::new(move |item, seed| {
        let mut rng = if let Some(seed) = seed {
            ChaCha8Rng::seed_from_u64(seed)
        } else {
            ChaCha8Rng::from_entropy()
        };

        let mut tokens: Vec<_> = if is_word {
            split_words(&item.processed)
                .iter()
                .map(|&(w, _)| Some(w))
                .collect()
        } else {
            CS::split(&item.processed, use_graphemes)
                .map(Some)
                .collect()
        };
        if tokens.len() <= 1 {
            return Ok(item);
        }
        let mut i = 0;
        while i < tokens.len() {
            let r: f64 = rng.gen();
            if r > mask_p {
                i += 1;
                continue;
            }
            let num_to_mask = (geo.sample(&mut rng) as usize + min).min(tokens.len() / 2);
            // a single masking should never be more than half of the tokens
            // to allow for more context between the tokens, should only happen
            // for very short sequences or very high corruption values
            // one mask token for all masked characters
            tokens[i] = Some(&mask_token);
            for j in i + 1..(i + num_to_mask).min(tokens.len()) {
                tokens[j] = None;
            }
            i += num_to_mask + 1;
        }
        let processed = tokens
            .into_iter()
            .flatten()
            .join(if is_word { " " } else { "" });
        Ok(TextData { processed, ..item })
    })
}

fn preprocessing_fn(preprocessing: PreprocessingConfig) -> Box<PreprocessingFn> {
    match preprocessing {
        PreprocessingConfig::None => Box::new(|item, _| Ok(item)),
        PreprocessingConfig::Chain(configs) => {
            let pfns = configs
                .into_iter()
                .map(preprocessing_fn)
                .collect::<Vec<_>>();
            Box::new(move |mut item, seed| {
                for pfn in &pfns {
                    item = pfn(item, seed)?;
                }
                Ok(item)
            })
        }
        PreprocessingConfig::Clean(use_g) => apply_to_text(move |s| Ok(text::clean(s, use_g))),
        PreprocessingConfig::Overwrite => overwrite_original_from_processed(),
        PreprocessingConfig::Switch(fns, probs) => switch(fns, probs),
        PreprocessingConfig::WhitespaceCorruption(iw_p, dw_p, use_g) => {
            corrupt_whitespace(iw_p, dw_p, use_g)
        }
        PreprocessingConfig::NoWhitespaces(use_graphemes) => {
            apply_to_text(move |s| Ok(remove(s, use_graphemes)))
        }
        PreprocessingConfig::FullWhitespaces(use_graphemes) => {
            apply_to_text(move |s| Ok(full(s, use_graphemes)))
        }
        PreprocessingConfig::CharSubstring(l, use_graphemes) => char_substring(l, use_graphemes),
        PreprocessingConfig::ByteSubstring(l, use_graphemes) => byte_substring(l, use_graphemes),
        PreprocessingConfig::Normalize(scheme, use_graphemes) => {
            apply_to_text(move |s| Ok(normalize(s, scheme, use_graphemes)))
        }
        PreprocessingConfig::LanguageDropout(p) => language_dropout(p),
        PreprocessingConfig::SpellingCorruption(p, full_del, mode) => {
            corrupt_spelling(p, full_del, mode)
        }
        PreprocessingConfig::MaskCorruption(mask_p, use_graphemes, mask_token, mode) => {
            corrupt_mask(mask_p, use_graphemes, mask_token, mode)
        }
    }
}

pub fn preprocessing(preprocessing: Vec<PreprocessingConfig>) -> Box<PreprocessingFn> {
    // return new function that runs all given preprocessing functions
    // in order
    let fns: Vec<Box<PreprocessingFn>> = preprocessing.into_iter().map(preprocessing_fn).collect();
    Box::new(move |mut item, seed| {
        for f in fns.iter() {
            item = f(item, seed)?;
        }
        Ok(item)
    })
}

fn whitespace_correction_label(
    use_graphemes: bool,
    tokenizer_cfg: TokenizerConfig,
) -> Box<LabelingFn> {
    let tokenizer = tokenizer(tokenizer_cfg)
        .expect("failed to create tokenizer for whitespace correction label function");
    let num_prefix_tokens = tokenizer.num_prefix_tokens();
    let num_suffix_tokens = tokenizer.num_suffix_tokens();
    Box::new(move |item| {
        Ok(Label::SequenceClassification(
            operations(&item.processed, &item.original, use_graphemes)?
                .into_iter()
                .map(|l| l as i32)
                .collect(),
            num_prefix_tokens,
            num_suffix_tokens,
        ))
    })
}

fn sequence_generation_label(tokenizer_cfg: TokenizerConfig) -> Box<LabelingFn> {
    let tokenizer = tokenizer(tokenizer_cfg)
        .expect("failed to create tokenizer for sequence generation label function");
    Box::new(move |item| {
        let tokenization =
            tokenizer.tokenize(&item.original, item.language.as_deref(), None, None, false)?;
        let token_ids = tokenization
            .token_ids
            .into_iter()
            .map(|t| t as i32)
            .collect();
        Ok(Label::SequenceGeneration(
            token_ids,
            tokenizer.pad_token_id(),
        ))
    })
}

pub fn labeling(labeling: LabelingConfig) -> Box<LabelingFn> {
    match labeling {
        LabelingConfig::WhitespaceCorrection(use_graphemes, tokenizer) => {
            whitespace_correction_label(use_graphemes, tokenizer)
        }
        LabelingConfig::SequenceGeneration(tokenizer) => sequence_generation_label(tokenizer),
    }
}

#[cfg(test)]
mod tests {
    use crate::data::TextData;

    use super::corrupt_whitespace;

    #[test]
    fn test_noise_whitespace_preprocessing() -> anyhow::Result<()> {
        let noise_fn = corrupt_whitespace(0.0, 1.0, true);
        let data = TextData::new("a test".to_string(), None, None);
        let noised = noise_fn(data.clone(), Some(0))?;
        assert_eq!(&noised.processed, "atest");
        let noise_fn = corrupt_whitespace(1.0, 0.0, true);
        let data = TextData::new("a test".to_string(), None, None);
        let noised = noise_fn(data.clone(), Some(0))?;
        assert_eq!(&noised.processed, "a t e s t");
        let data = TextData::new("Ginsberǵs".to_string(), None, None);
        let noised = noise_fn(data.clone(), Some(0))?;
        assert_eq!(&noised.processed, "G i n s b e r ǵ s");
        Ok(())
    }
}
