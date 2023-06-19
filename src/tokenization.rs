use crate::text::{clean, count_words_whitespace, file_size, SPLIT_WORD_WHITESPACE_PATTERN};
use crate::unicode::{normalize, Normalization, CS};
use crate::utils::{
    accumulate, progress_bar, py_invalid_type_error, py_required_key_error, SerializeMsgPack,
};
use anyhow::anyhow;
use itertools::Itertools;
use log::info;
use numpy::ndarray::{Array1, Array2};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use regex::{escape, Regex};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::borrow::Borrow;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::hash::Hash;
use std::io::{BufRead, BufReader};
use std::panic;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{sleep, Builder, JoinHandle};
use std::time::Duration;

pub const UNK: &str = "(unk)";
pub const BOS: &str = "(bos)";
pub const EOS: &str = "(eos)";
pub const PAD: &str = "(pad)";
pub const SPECIAL_TOKENS: [&str; 4] = [UNK, BOS, EOS, PAD];
pub const DEFAULT_PREFIX_TOKENS: [&str; 1] = [BOS];
pub const DEFAULT_SUFFIX_TOKENS: [&str; 1] = [EOS];

#[pyclass]
pub struct SpecialTokens {}

#[pymethods]
impl SpecialTokens {
    #[classattr]
    const UNK: &str = UNK;
    #[classattr]
    const BOS: &str = BOS;
    #[classattr]
    const EOS: &str = EOS;
    #[classattr]
    const PAD: &str = PAD;
}

// language tokens
pub const LANG_UNK: &str = "(lang:unk)";

#[pyclass]
pub struct LanguageTokens {}

#[pymethods]
impl LanguageTokens {
    #[classattr]
    const UNK: &str = LANG_UNK;
}

/// Config for special tokens and options regarding special tokens
#[derive(Debug, Clone)]
pub struct SpecialConfig {
    pub pad: String,
    pub tokens: Vec<String>,
    pub prefix: Vec<String>,
    pub suffix: Vec<String>,
}

impl Default for SpecialConfig {
    fn default() -> Self {
        Self {
            pad: PAD.to_string(),
            tokens: SPECIAL_TOKENS.iter().map(|s| s.to_string()).collect(),
            prefix: DEFAULT_PREFIX_TOKENS
                .iter()
                .map(|s| s.to_string())
                .collect(),
            suffix: DEFAULT_SUFFIX_TOKENS
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }
}

impl<'a> FromPyObject<'a> for SpecialConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        Ok(Self {
            pad: if let Some(value) = d.get_item("pad") {
                value.extract()?
            } else {
                PAD.to_string()
            },
            tokens: if let Some(value) = d.get_item("tokens") {
                value.extract()?
            } else {
                SPECIAL_TOKENS.iter().map(|s| s.to_string()).collect()
            },
            prefix: if let Some(value) = d.get_item("prefix") {
                value.extract()?
            } else {
                DEFAULT_PREFIX_TOKENS
                    .iter()
                    .map(|s| s.to_string())
                    .collect()
            },
            suffix: if let Some(value) = d.get_item("suffix") {
                value.extract()?
            } else {
                DEFAULT_SUFFIX_TOKENS
                    .iter()
                    .map(|s| s.to_string())
                    .collect()
            },
        })
    }
}

/// This is a tokenizer config, containing configs for special tokens, language,
/// and the actual tokenize config inside it.
#[derive(Clone, Debug)]
pub struct TokenizerConfig {
    pub tokenize: TokenizeConfig,
    pub special: SpecialConfig,
    pub language: Option<LanguageConfig>,
}

impl<'a> FromPyObject<'a> for TokenizerConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        Ok(Self {
            tokenize: d
                .get_item("tokenize")
                .ok_or_else(|| py_required_key_error("tokenize", "tokenizer config"))?
                .extract()?,
            special: if let Some(value) = d.get_item("special") {
                value.extract()?
            } else {
                SpecialConfig::default()
            },
            language: if let Some(value) = d.get_item("language") {
                Some(value.extract()?)
            } else {
                None
            },
        })
    }
}

/// This configures the language a tokenizer can work with
#[derive(Clone, Debug)]
pub struct LanguageConfig {
    add_language_token_to_prefix: bool,
    add_language_token_to_suffix: bool,
    languages: Vec<String>,
    default_language: String,
}

impl LanguageConfig {
    pub fn new(
        add_language_token_to_prefix: bool,
        add_language_token_to_suffix: bool,
        languages: Vec<String>,
        default_language: String,
    ) -> Self {
        Self {
            add_language_token_to_prefix,
            add_language_token_to_suffix,
            languages,
            default_language,
        }
    }
}

impl<'a> FromPyObject<'a> for LanguageConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let Some(languages) = d.get_item("languages") else {
            return Err(py_required_key_error("languages", "language config"));
        };
        let languages = languages.extract()?;
        let Some(default_language) = d.get_item("default_language") else {
            return Err(py_required_key_error(
                "default_language",
                "language config",
            ));
        };
        let default_language = default_language.extract()?;
        Ok(Self {
            add_language_token_to_prefix: if let Some(value) =
                d.get_item("add_language_token_to_prefix")
            {
                value.extract()?
            } else {
                true
            },
            add_language_token_to_suffix: if let Some(value) =
                d.get_item("add_language_token_to_suffix")
            {
                value.extract()?
            } else {
                false
            },
            languages,
            default_language,
        })
    }
}

/// This enum defines all tokenizers that are supported by this crate.
#[derive(Clone, Debug)]
pub enum TokenizeConfig {
    Character(CharTokenizerConfig),
    Byte(ByteTokenizerConfig),
    ByT5(ByteTokenizerConfig),
    BPE(BPETokenizerConfig),
    Dummy(Duration),
}

impl IntoPy<PyObject> for TokenizeConfig {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let d: &PyDict = PyDict::new(py);
        let tokenizer_type = match self {
            TokenizeConfig::Character(cfg) => {
                d.set_item("use_graphemes", cfg.use_graphemes).unwrap();
                "character"
            }
            TokenizeConfig::Byte(cfg) => {
                d.set_item("use_graphemes", cfg.use_graphemes).unwrap();
                d.set_item("groups", cfg.groups.into_py(py)).unwrap();
                d.set_item("aggregation", cfg.aggregation.into_py(py))
                    .unwrap();
                if let Some(pad) = cfg.pad_to_multiple_of {
                    d.set_item("pad_to_multiple_of", pad).unwrap();
                }
                "byte"
            }
            TokenizeConfig::ByT5(cfg) => {
                d.set_item("use_graphemes", cfg.use_graphemes).unwrap();
                d.set_item("groups", cfg.groups.into_py(py)).unwrap();
                d.set_item("aggregation", cfg.aggregation.into_py(py))
                    .unwrap();
                "byt5"
            }
            TokenizeConfig::BPE(cfg) => {
                d.set_item("use_graphemes", cfg.use_graphemes).unwrap();
                d.set_item("merge_file", cfg.merge_file).unwrap();
                "bpe"
            }
            TokenizeConfig::Dummy(delay) => {
                d.set_item("delay", delay.as_millis()).unwrap();
                "dummy"
            }
        };
        d.set_item("type", tokenizer_type).unwrap();
        d.to_object(py)
    }
}

impl<'a> FromPyObject<'a> for TokenizeConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let Some(tokenizer_type) = d.get_item("type") else {
            return Err(py_required_key_error("type", "tokenizer config"));
        };
        let tokenizer_type: String = tokenizer_type.extract()?;
        let tokenizer_config = match tokenizer_type.as_str() {
            "character" => {
                let use_graphemes: bool = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };
                TokenizeConfig::Character(CharTokenizerConfig { use_graphemes })
            }
            name @ ("byte" | "byt5") => {
                let use_graphemes: bool = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };
                let Some(groups) = d.get_item("groups") else {
                    return Err(py_required_key_error("groups", format!("{name} tokenizer config")));
                };
                let agg: GroupAggregation = if let Some(value) = d.get_item("aggregation") {
                    value.extract()?
                } else {
                    GroupAggregation::Mean
                };
                let pad_to_multiple_of = if let Some(value) = d.get_item("pad_to_multiple_of") {
                    Some(value.extract()?)
                } else {
                    None
                };
                let byte_cfg = ByteTokenizerConfig {
                    use_graphemes,
                    pad_to_multiple_of,
                    groups: groups.extract()?,
                    aggregation: agg,
                };
                if name == "byt5" {
                    TokenizeConfig::ByT5(ByteTokenizerConfig {
                        pad_to_multiple_of: None,
                        ..byte_cfg
                    })
                } else {
                    TokenizeConfig::Byte(byte_cfg)
                }
            }
            "bpe" => {
                let use_graphemes: bool = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };
                let Some(merge_file) = d.get_item("merge_file") else {
                    return Err(py_required_key_error("merge_file", "bpe tokenizer config"));
                };
                let max_vocab_size: Option<usize> =
                    if let Some(value) = d.get_item("max_vocab_size") {
                        Some(value.extract()?)
                    } else {
                        None
                    };
                TokenizeConfig::BPE(BPETokenizerConfig {
                    use_graphemes,
                    merge_file: merge_file.extract()?,
                    max_vocab_size,
                })
            }
            "dummy" => {
                let millis: u64 = if let Some(value) = d.get_item("delay") {
                    value.extract()?
                } else {
                    0
                };
                TokenizeConfig::Dummy(Duration::from_millis(millis))
            }
            k => {
                return Err(py_invalid_type_error(k, "tokenizer"));
            }
        };
        Ok(tokenizer_config)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TokenGroup {
    Empty(usize),
    Full(usize),
    Nested(Vec<TokenGroup>),
}

impl IntoPy<PyObject> for TokenGroup {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let d = PyDict::new(py);
        match self {
            TokenGroup::Empty(len) => {
                d.set_item("type", "empty").unwrap();
                d.set_item("len", len).unwrap();
            }
            TokenGroup::Full(len) => {
                d.set_item("type", "full").unwrap();
                d.set_item("len", len).unwrap();
            }
            TokenGroup::Nested(groups) => {
                d.set_item("type", "nested").unwrap();
                d.set_item("groups", groups.into_py(py)).unwrap();
            }
        };
        d.into_py(py)
    }
}

impl ToPyObject for TokenGroup {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        self.clone().into_py(py)
    }
}

impl TokenGroup {
    pub fn len(&self) -> usize {
        match self {
            TokenGroup::Empty(len) => *len,
            TokenGroup::Full(len) => *len,
            TokenGroup::Nested(groups) => groups.iter().map(|g| g.len()).sum(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get_weights(&self, agg: GroupAggregation) -> Vec<f32> {
        match self {
            TokenGroup::Empty(len) => {
                vec![0.0; *len]
            }
            TokenGroup::Full(len) => {
                let weight = match agg {
                    GroupAggregation::Sum => 1.0,
                    GroupAggregation::Mean => 1.0 / *len as f32,
                };
                vec![weight; *len]
            }
            TokenGroup::Nested(groups) => {
                let weight = match agg {
                    GroupAggregation::Sum => 1.0,
                    GroupAggregation::Mean => 1.0 / groups.len() as f32,
                };
                groups
                    .iter()
                    .flat_map(|g| g.get_weights(agg))
                    .map(|w| w * weight)
                    .collect()
            }
        }
    }
}

pub type Grouping = (Vec<TokenGroup>, GroupAggregation);
/// This enum defines all possible additional infos that can be returned by
/// a tokenizers tokenize function in addition to the token ids themselves.
#[derive(Clone, Debug)]
pub enum TokenizationInfo {
    /// No additional info.
    Empty,
    /// Token groups specify which subsequent tokens belong to the same group.
    /// Useful e.g. when defining a byte tokenizer that should also return
    /// information about which byte belongs to which character.
    TokenGroups(HashMap<String, Grouping>),
    /// Multi info allows multiple additional informations to be returned
    Info(HashMap<String, TokenizationInfo>),
}

pub enum TensorizedTokenizationInfo {
    Empty,
    TokenGroups(HashMap<String, (SparseCoo, PaddingMask)>),
}

pub struct SparseCoo {
    pub(crate) indices: Array2<i32>,
    pub(crate) values: Array1<f32>,
    pub(crate) size: Vec<usize>,
    pub(crate) group_lengths: Vec<usize>,
}

impl IntoPy<PyObject> for SparseCoo {
    fn into_py(self, py: Python<'_>) -> PyObject {
        (
            self.indices.into_pyarray(py),
            self.values.into_pyarray(py),
            self.size,
            self.group_lengths,
        )
            .into_py(py)
    }
}

pub struct PaddingMask {
    inner: Array2<bool>,
}

impl IntoPy<PyObject> for PaddingMask {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.inner.into_pyarray(py).into_py(py)
    }
}

impl IntoPy<PyObject> for TensorizedTokenizationInfo {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let d = PyDict::new(py);
        if let TensorizedTokenizationInfo::TokenGroups(matrices) = self {
            for (name, (scoo, pad_mask)) in matrices {
                let t = PyTuple::new(py, &[scoo.into_py(py), pad_mask.into_py(py)]);
                d.set_item(name, t).unwrap();
            }
        };
        d.into_py(py)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupAggregation {
    Mean,
    Sum,
}

impl IntoPy<PyObject> for GroupAggregation {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            GroupAggregation::Mean => "mean",
            GroupAggregation::Sum => "sum",
        }
        .into_py(py)
    }
}

impl ToPyObject for GroupAggregation {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        self.into_py(py)
    }
}

impl<'a> FromPyObject<'a> for GroupAggregation {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let agg: String = ob.extract()?;
        match agg.as_str() {
            "mean" => Ok(GroupAggregation::Mean),
            "sum" => Ok(GroupAggregation::Sum),
            k => Err(py_invalid_type_error(k, "group aggregation")),
        }
    }
}

pub fn token_groups_to_sparse_coo_matrix(
    groupings: &[&Grouping],
    lengths: &[usize],
) -> anyhow::Result<SparseCoo> {
    assert_eq!(groupings.len(), lengths.len());
    let group_lengths: Vec<_> = groupings.iter().map(|(groups, _)| groups.len()).collect();
    let max_group_length = group_lengths.iter().max().copied().unwrap_or(0);
    let max_length = lengths.iter().max().copied().unwrap_or(0);
    let cum_lengths = accumulate(lengths);
    let stride = lengths.iter().sum();
    let mut indices = vec![0; 3 * stride];
    let mut values = vec![1.0; stride];
    let mut offset = 0;
    for (batch_index, &(groups, agg)) in groupings.iter().enumerate() {
        let mut group_offset = 0;
        for (group_idx, group) in groups.iter().enumerate() {
            let group_len = group.len();
            // batch indices
            indices[offset..offset + group_len]
                .iter_mut()
                .for_each(|v| *v = batch_index as i32);
            // target group indices
            indices[stride + offset..stride + offset + group_len]
                .iter_mut()
                .for_each(|v| *v = group_idx as i32);
            // current ungrouped indices
            indices[2 * stride + offset..2 * stride + offset + group_len]
                .iter_mut()
                .zip(group_offset..group_offset + group_len as i32)
                .for_each(|(v, w)| *v = w);
            if *agg == GroupAggregation::Mean {
                let weights = group.get_weights(*agg);
                assert_eq!(weights.len(), group_len);
                values[offset..offset + group_len]
                    .iter_mut()
                    .zip(weights)
                    .for_each(|(v, w)| *v = w);
            }
            offset += group_len;
            group_offset += group_len as i32;
        }
        assert_eq!(
            offset, cum_lengths[batch_index],
            "expected offset to be {}, but was {}",
            cum_lengths[batch_index], offset
        );
    }
    let size = vec![groupings.len(), max_group_length, max_length];
    Ok(SparseCoo {
        indices: Array2::from_shape_vec((3, stride), indices)?,
        values: Array1::from_vec(values),
        size,
        group_lengths,
    })
}

pub fn padding_mask(lengths: &[usize]) -> anyhow::Result<PaddingMask> {
    let max_length = lengths.iter().max().copied().unwrap_or(0);
    let padding_mask_vec: Vec<_> = lengths
        .iter()
        .flat_map(|l| {
            let mut pad = vec![false; *l];
            pad.append(&mut vec![true; max_length - l]);
            pad
        })
        .collect();
    Ok(PaddingMask {
        inner: Array2::from_shape_vec((lengths.len(), max_length), padding_mask_vec)?,
    })
}

impl IntoPy<PyObject> for TokenizationInfo {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let d = PyDict::new(py);
        let info_type = match self {
            TokenizationInfo::Empty => "empty",
            TokenizationInfo::TokenGroups(token_groups) => {
                for (group_name, (groups, agg)) in token_groups {
                    let l = PyList::empty(py);
                    for group in groups {
                        l.append(group).unwrap();
                    }
                    let gd = PyDict::new(py);
                    gd.set_item("groups", l).unwrap();
                    gd.set_item("aggregation", agg.into_py(py)).unwrap();
                    d.set_item(group_name, gd).unwrap();
                }
                "token_groups"
            }
            TokenizationInfo::Info(infos) => {
                for (info_name, info) in infos {
                    d.set_item(info_name, info.into_py(py)).unwrap();
                }
                "info"
            }
        };
        d.set_item("type", info_type).unwrap();
        d.into()
    }
}

// impl<'a> FromPyObject<'a> for TokenizationInfo {
//     fn extract(ob: &'a PyAny) -> PyResult<Self> {
//         let d: &PyDict = ob.extract()?;
//         let Some(info_type) = d.get_item("type") else {
//             return Err(py_required_key_error("type", "tokenization info"));
//         };
//         let info_type: String = info_type.extract()?;
//         let info = match info_type.as_str() {
//             "empty" => TokenizationInfo::Empty,
//             "token_groups" => {
//                 let mut token_groups = HashMap::new();
//                 for key in d.keys() {
//                     let key_s: String = key.extract()?;
//                     if key_s == "type" {
//                         continue;
//                     }
//                     let gd = d.get_item(key).unwrap();
//                     let groups = gd.get_item("groups")?.extract()?;
//                     let agg = gd.get_item("aggregation")?.extract()?;
//                     token_groups.insert(key_s, (groups, agg));
//                 }
//                 TokenizationInfo::TokenGroups(token_groups)
//             }
//             "info" => {
//                 let mut info = HashMap::new();
//                 for key in d.keys() {
//                     let key_s: String = key.extract()?;
//                     let value = d.get_item(key).unwrap();
//                     info.insert(key_s, value.extract()?);
//                 }
//                 TokenizationInfo::Info(info)
//             }
//             k => return Err(py_invalid_type_error(k, "tokenization info")),
//         };
//         Ok(info)
//     }
// }

/// A tokenization is defined to be a combination of token ids and some additional information.
/// This is returned by a tokenizers tokenize function.
#[derive(Debug, Clone)]
#[pyclass]
pub struct Tokenization {
    #[pyo3(get)]
    pub token_ids: Vec<u32>,
    #[pyo3(get)]
    pub info: TokenizationInfo,
}

impl Tokenization {
    pub fn new(token_ids: Vec<u32>, info: TokenizationInfo) -> Self {
        Tokenization { token_ids, info }
    }
}

/// A tokenization function in general takes in a &str and return a tokenization.
pub type TokenizationFn = Box<dyn Send + 'static + Fn(&str) -> Tokenization>;
/// A tokenizer is something that implements the tokenize trait
pub type Tokenizer = Box<dyn Send + 'static + Tokenize>;

/// The tokenize trait defines behavior that every tokenizer should support.
pub trait BaseTokenize: Send + Sync + 'static {
    fn num_prefix_tokens(&self) -> usize {
        self.prefix_token_ids().len()
            + match self.language_config().as_ref() {
                Some(cfg) => cfg.add_language_token_to_prefix as usize,
                None => 0,
            }
    }

    fn num_suffix_tokens(&self) -> usize {
        self.suffix_token_ids().len()
            + match self.language_config().as_ref() {
                Some(cfg) => cfg.add_language_token_to_suffix as usize,
                None => 0,
            }
    }

    fn num_special_tokens(&self) -> usize {
        self.num_prefix_tokens() + self.num_suffix_tokens()
    }

    fn prefix_token_ids(&self) -> &[u32];

    fn suffix_token_ids(&self) -> &[u32];

    fn pad_token_id(&self) -> u32;

    fn language_config(&self) -> Option<&LanguageConfig>;

    fn special_token_to_id(&self, token: &str) -> Option<u32>;
}

pub trait Tokenize: BaseTokenize {
    fn vocab_size(&self) -> usize;

    fn tokenize(
        &self,
        s: &str,
        lang: Option<&str>,
        prefix: Option<&[&str]>,
        suffix: Option<&[&str]>,
        ignore_special_tokens: bool,
    ) -> anyhow::Result<Tokenization>;

    fn de_tokenize(&self, token_ids: &[u32], ignore_special_tokens: bool) -> String;
}

/// A base struct for a tokenizer,
/// allows custom tokenizers to be built by setting config and state
pub struct BaseTokenizer<Config = (), State = ()> {
    prefix_token_ids: Vec<u32>,
    suffix_token_ids: Vec<u32>,
    pad_token_id: u32,
    state: State,
    config: Config,
    language_config: Option<LanguageConfig>,
    special_vocab: Vocab<String>,
    special_token_pattern: Option<Regex>,
}

impl<Config, State> BaseTokenize for BaseTokenizer<Config, State>
where
    Self: Send + Sync + 'static,
{
    fn prefix_token_ids(&self) -> &[u32] {
        &self.prefix_token_ids
    }

    fn suffix_token_ids(&self) -> &[u32] {
        &self.suffix_token_ids
    }

    fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    fn language_config(&self) -> Option<&LanguageConfig> {
        self.language_config.as_ref()
    }

    fn special_token_to_id(&self, token: &str) -> Option<u32> {
        self.special_vocab.token_to_id(token)
    }
}

enum TokenInput<'a> {
    Regular(&'a str),
    Special(&'a str),
}

impl<Config, State> BaseTokenizer<Config, State>
where
    Self: BaseTokenize,
{
    fn new_base_tokenizer(
        special_offset: u32,
        special_config: SpecialConfig,
        language_config: Option<LanguageConfig>,
        config: Config,
        state: State,
    ) -> Self {
        let languages = if let Some(lang_cfg) = language_config.as_ref() {
            let mut l = vec![lang_cfg.default_language.clone()];
            l.extend(lang_cfg.languages.iter().cloned());
            l
        } else {
            vec![]
        };
        assert!(
            special_config.tokens.contains(&special_config.pad),
            "pad token not in special tokens"
        );
        assert!(
            special_config
                .prefix
                .iter()
                .all(|tok| special_config.tokens.contains(tok)),
            "one or more prefix tokens are not in special tokens"
        );
        assert!(
            special_config
                .suffix
                .iter()
                .all(|tok| special_config.tokens.contains(tok)),
            "one or more suffix tokens are not in special tokens"
        );
        assert!(
            languages.iter().all(|l| special_config.tokens.contains(l)),
            "one or more language tokens not in special tokens:\nlanguages:{:?}\nspecial tokens:{:?}",
            &languages,
            &special_config.tokens
        );
        let special_vocab = Vocab::build(special_config.tokens, special_offset);
        let prefix_token_ids = special_config
            .prefix
            .iter()
            .map(|tok| special_vocab.token_to_id(tok).unwrap())
            .collect();
        let suffix_token_ids = special_config
            .suffix
            .iter()
            .map(|tok| special_vocab.token_to_id(tok).unwrap())
            .collect();
        let special_token_pattern = if special_vocab.size() > 0 {
            let re = Regex::new(&special_vocab.vocab.keys().map(|st| escape(st)).join(r"|"))
                .expect("invalid regex pattern, should not happen");
            Some(re)
        } else {
            None
        };
        BaseTokenizer {
            prefix_token_ids,
            suffix_token_ids,
            pad_token_id: special_vocab.token_to_id(&special_config.pad).unwrap(),
            language_config,
            special_vocab,
            special_token_pattern,
            config,
            state,
        }
    }

    fn split_input<'a>(&self, s: &'a str, ignore_special_tokens: bool) -> Vec<TokenInput<'a>> {
        if ignore_special_tokens || self.special_token_pattern.is_none() {
            vec![TokenInput::Regular(s)]
        } else {
            let mut splits = vec![];
            let mut last = 0;
            for m in self.special_token_pattern.as_ref().unwrap().find_iter(s) {
                if m.start() > last {
                    splits.push(TokenInput::Regular(&s[last..m.start()]));
                }
                splits.push(TokenInput::Special(&s[m.start()..m.end()]));
                last = m.end();
            }
            if last < s.len() {
                splits.push(TokenInput::Regular(&s[last..]));
            }
            splits
        }
    }

    fn add_prefix_and_suffix(
        &self,
        mut token_ids: Vec<u32>,
        lang: Option<&str>,
        prefix: Option<&[&str]>,
        suffix: Option<&[&str]>,
    ) -> anyhow::Result<Vec<u32>> {
        let mut pfx = self.prefix_token_ids().to_vec();
        let mut sfx = self.suffix_token_ids().to_vec();
        // add language token if needed
        if let Some(lang_cfg) = self.language_config() {
            let lang = lang.unwrap_or(&lang_cfg.default_language);
            let Some(lang_id) = self.special_token_to_id(lang) else {
                return Err(anyhow!(
                    "language {lang} is not supported by this tokenizer"
                ));
            };
            if lang_cfg.add_language_token_to_prefix {
                pfx.push(lang_id);
            }
            if lang_cfg.add_language_token_to_suffix {
                sfx.push(lang_id);
            }
        }
        // add additional prefix tokens if needed
        if let Some(add_prefix) = prefix {
            for &token in add_prefix {
                let Some(token_id) = self.special_token_to_id(token) else {
                    return Err(anyhow!(
                        "prefix token {token} is not a valid special token"
                    ));
                };
                pfx.push(token_id);
            }
        }
        // add additional suffix tokens if needed
        if let Some(add_suffix) = suffix {
            for &token in add_suffix {
                let Some(token_id) = self.special_token_to_id(token) else {
                    return Err(anyhow!(
                        "prefix token {token} is not a valid special token"
                    ));
                };
                sfx.push(token_id);
            }
        }
        pfx.reserve_exact(token_ids.len() + sfx.len());
        pfx.append(&mut token_ids);
        pfx.append(&mut sfx);
        Ok(pfx)
    }
}

pub struct Vocab<Token> {
    vocab: HashMap<Token, u32>,
    reverse_vocab: HashMap<u32, Token>,
}

impl<Token> Vocab<Token>
where
    Token: PartialEq + Eq + Hash + Clone,
{
    fn build(tokens: impl IntoIterator<Item = Token>, start_id: u32) -> Self {
        let vocab: HashMap<Token, u32> = tokens
            .into_iter()
            .unique()
            .enumerate()
            .map(|(tok_id, tok)| (tok, start_id + tok_id as u32))
            .collect();
        let reverse_vocab = vocab
            .iter()
            .map(|(token, token_id)| (*token_id, token.clone()))
            .collect();
        Self {
            vocab,
            reverse_vocab,
        }
    }

    fn size(&self) -> usize {
        self.vocab.len()
    }

    fn token_to_id<K>(&self, token: &K) -> Option<u32>
    where
        K: Hash + Eq + ?Sized,
        Token: Borrow<K>,
    {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: &u32) -> Option<&Token> {
        self.reverse_vocab.get(id)
    }
}

impl<Token> Vocab<Token>
where
    Token: PartialEq + Eq + Hash + Clone + Ord + Serialize + DeserializeOwned,
{
    #[allow(dead_code)]
    fn to_file(&self, file: impl AsRef<Path>) -> anyhow::Result<()> {
        let tokens = self
            .vocab
            .clone()
            .into_iter()
            .sorted_by_key(|&(_, id)| id)
            .collect::<Vec<_>>();
        tokens.save(file)?;
        Ok(())
    }

    fn from_file(file: impl AsRef<Path>, start_id: u32) -> anyhow::Result<Self> {
        let tokens: Vec<Token> = Vec::load(file)?;
        let vocab: HashMap<_, _> = tokens
            .into_iter()
            .enumerate()
            .map(|(token_id, token)| (token, start_id + token_id as u32))
            .collect();
        let reverse_vocab = vocab
            .iter()
            .map(|(token, token_id)| (*token_id, token.clone()))
            .collect();
        Ok(Self {
            vocab,
            reverse_vocab,
        })
    }
}

pub type VocabFreeTokenizer<Config> = BaseTokenizer<Config>;

impl<Config> VocabFreeTokenizer<Config>
where
    Config: Send + Sync + 'static,
{
    pub fn new_vocab_free_tokenizer(
        special_offset: u32,
        special_config: SpecialConfig,
        language_config: Option<LanguageConfig>,
        config: Config,
    ) -> Self {
        Self::new_base_tokenizer(special_offset, special_config, language_config, config, ())
    }
}

pub type VocabTokenizer<Token, Config> = BaseTokenizer<Config, (String, Vocab<Token>)>;

enum VocabToken<'a, Token> {
    Token(Token),
    Special(&'a str),
}
trait VocabTokenize<Token> {
    fn process_token_input<'a>(
        &'a self,
        inputs: Vec<TokenInput<'a>>,
    ) -> (Vec<VocabToken<'a, Token>>, TokenizationInfo);

    fn join_tokens(&self, tokens: &[&Token]) -> String;

    fn join_parts(&self, parts: &[impl AsRef<str>]) -> String;
}

impl<Token, Config> VocabTokenizer<Token, Config>
where
    Token: PartialEq + Eq + Hash + Send + Sync + Clone + 'static,
    Config: Send + Sync + 'static,
{
    pub fn new_vocab_tokenizer(
        tokens: Vec<Token>,
        unk_token: String,
        mut special_config: SpecialConfig,
        language_config: Option<LanguageConfig>,
        config: Config,
    ) -> Self {
        let vocab = Vocab::build(tokens, 0);
        // add unk token to special config
        special_config.tokens.push(unk_token.clone());
        Self::new_base_tokenizer(
            vocab.size() as u32,
            special_config,
            language_config,
            config,
            (unk_token, vocab),
        )
    }

    pub fn unk_token_id(&self) -> u32 {
        self.special_vocab.token_to_id(&self.state.0).unwrap()
    }
}

impl<Token, Config> VocabTokenizer<Token, Config>
where
    Token:
        PartialEq + Eq + Hash + Send + Sync + Clone + 'static + Serialize + DeserializeOwned + Ord,
    Config: Send + Sync + 'static,
{
    pub fn new_vocab_tokenizer_from_file(
        vocab_file: impl AsRef<Path>,
        unk_token: String,
        mut special_config: SpecialConfig,
        language_config: Option<LanguageConfig>,
        config: Config,
    ) -> anyhow::Result<Self> {
        let vocab = Vocab::from_file(vocab_file, 0)?;
        // add unk token to special config
        special_config.tokens.push(unk_token.clone());
        Ok(Self::new_base_tokenizer(
            vocab.size() as u32,
            special_config,
            language_config,
            config,
            (unk_token, vocab),
        ))
    }
}

impl<Token, Config> Tokenize for VocabTokenizer<Token, Config>
where
    Token: PartialEq + Eq + Hash + Send + Sync + Clone + 'static,
    Config: Send + Sync + 'static,
    Self: VocabTokenize<Token>,
{
    fn vocab_size(&self) -> usize {
        self.state.1.size() + self.special_vocab.size()
    }

    fn tokenize(
        &self,
        s: &str,
        lang: Option<&str>,
        prefix: Option<&[&str]>,
        suffix: Option<&[&str]>,
        ignore_special_tokens: bool,
    ) -> anyhow::Result<Tokenization> {
        let token_input = self.split_input(s, ignore_special_tokens);
        let (tokens, tokenization_info) = self.process_token_input(token_input);
        let token_ids = tokens
            .iter()
            .map(|token| {
                match token {
                    VocabToken::Special(token) => self.special_vocab.token_to_id(*token).unwrap(),
                    VocabToken::Token(token) => self
                        .state
                        .1
                        .token_to_id(token)
                        .unwrap_or_else(|| self.unk_token_id()),
                }
                // // }
            })
            .collect::<Vec<_>>();
        let token_ids = self.add_prefix_and_suffix(token_ids, lang, prefix, suffix)?;
        Ok(Tokenization::new(token_ids, tokenization_info))
    }

    fn de_tokenize(&self, token_ids: &[u32], ignore_special_tokens: bool) -> String {
        let mut tokens = vec![];
        let mut parts = vec![];
        for token_id in token_ids {
            if let Some(token) = self.state.1.id_to_token(token_id) {
                tokens.push(token);
            } else if !ignore_special_tokens {
                if !tokens.is_empty() {
                    parts.push(self.join_tokens(&tokens));
                    tokens.clear();
                }
                let special_token = self
                    .special_vocab
                    .id_to_token(token_id)
                    .expect("invalid token id in input");
                parts.push(special_token.to_string());
            };
        }
        // dont forget to join the remaining tokens
        if !tokens.is_empty() {
            parts.push(self.join_tokens(&tokens));
        }
        self.join_parts(&parts)
    }
}

/// Dummy tokenizer that just waits a specified time in its tokenize function.
/// Used for testing only.
pub type DummyTokenizer = VocabFreeTokenizer<Duration>;

impl DummyTokenizer {
    fn new(delay: Duration) -> Self {
        Self::new_vocab_free_tokenizer(0, SpecialConfig::default(), None, delay)
    }
}

impl Tokenize for DummyTokenizer {
    fn vocab_size(&self) -> usize {
        0
    }

    fn tokenize(
        &self,
        _: &str,
        _: Option<&str>,
        _: Option<&[&str]>,
        _: Option<&[&str]>,
        _: bool,
    ) -> anyhow::Result<Tokenization> {
        sleep(self.config);
        Ok(Tokenization::new(vec![], TokenizationInfo::Empty))
    }

    fn de_tokenize(&self, _: &[u32], _: bool) -> String {
        "".to_string()
    }
}

/// A tokenizer based on the ascii characters, digits, and punctuations marks.
/// Can e.g. be used to efficiently (meaning small vocab size) represent most
/// English texts.
#[derive(Debug, Clone)]
pub struct CharTokenizerConfig {
    pub use_graphemes: bool,
}
pub type CharTokenizer = VocabTokenizer<char, CharTokenizerConfig>;

const CHARS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\"\"!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\"\" ";

impl VocabTokenize<char> for CharTokenizer {
    #[inline]
    fn join_tokens(&self, tokens: &[&char]) -> String {
        tokens.iter().join("")
    }

    #[inline]
    fn join_parts(&self, parts: &[impl AsRef<str>]) -> String {
        parts.iter().map(|p| p.as_ref()).join("")
    }

    #[inline]
    fn process_token_input<'a>(
        &'a self,
        inputs: Vec<TokenInput<'a>>,
    ) -> (Vec<VocabToken<'a, char>>, TokenizationInfo) {
        let mut tokens = vec![];
        for input in inputs {
            match input {
                TokenInput::Regular(s) => {
                    tokens.extend(CS::new(s, self.config.use_graphemes).chars().map(|c| {
                        // Character always has at least one char so this is safe
                        let mut code_points = c.code_points();
                        let char = code_points.next().unwrap();
                        // return unk if Character has another char because
                        // our tokens in the vocab are all single char tokens
                        if code_points.next().is_some() {
                            VocabToken::Special(&self.state.0)
                        } else {
                            VocabToken::Token(char)
                        }
                    }));
                }
                TokenInput::Special(special) => {
                    tokens.push(VocabToken::Special(special));
                }
            }
        }
        (tokens, TokenizationInfo::Empty)
    }
}

impl CharTokenizer {
    pub fn new(
        config: CharTokenizerConfig,
        special_config: SpecialConfig,
        language_config: Option<LanguageConfig>,
    ) -> Self {
        Self::new_vocab_tokenizer(
            CHARS.chars().collect(),
            UNK.to_string(),
            special_config,
            language_config,
            config,
        )
    }
}

#[derive(Debug, Clone)]
pub struct BPETokenizerConfig {
    pub merge_file: PathBuf,
    pub max_vocab_size: Option<usize>,
    pub use_graphemes: bool,
}

pub type MergeOps = HashMap<Vec<u8>, u32>;

pub type BPETokenizer = BaseTokenizer<BPETokenizerConfig, (MergeOps, Vec<Vec<u8>>, Regex)>;

impl BPETokenizer {
    pub fn new(
        config: BPETokenizerConfig,
        special_config: SpecialConfig,
        language_config: Option<LanguageConfig>,
    ) -> anyhow::Result<Self> {
        let mut merge_ops = MergeOps::load(&config.merge_file)?;
        if let Some(limit) = config.max_vocab_size {
            // to limit vocab size we filter out all merges with an id higher than the limit
            let limit = limit
                .saturating_sub(special_config.tokens.len())
                .saturating_sub(256) as u32;
            merge_ops.retain(|_, &mut merge_id| merge_id < limit);
        }
        let mut reverse_merge_ops: Vec<Vec<u8>> = Vec::new();
        for b in 0..256 {
            reverse_merge_ops.push(vec![b as u8]);
        }
        for (bytes, _) in merge_ops.iter().sorted_by_key(|&(_, merge_id)| merge_id) {
            reverse_merge_ops.push(bytes.to_vec());
        }
        Ok(Self::new_base_tokenizer(
            reverse_merge_ops.len() as u32,
            special_config,
            language_config,
            config,
            (
                merge_ops,
                reverse_merge_ops,
                Regex::new(SPLIT_WORD_WHITESPACE_PATTERN)?,
            ),
        ))
    }

    fn merge_bytes(&self, s: &str) -> Vec<u32> {
        let mut joined_token_ids = vec![];

        // for word in split_words_whitespace(s, true) {
        for word in self.state.2.find_iter(s) {
            let word = word.as_str();

            let mut bytes: Vec<_> = word.as_bytes().iter().map(|b| vec![*b]).collect();
            let mut token_ids: Vec<Option<u32>> =
                word.as_bytes().iter().map(|b| Some(*b as u32)).collect();

            let mut heap: BinaryHeap<_> = bytes
                .iter()
                .enumerate()
                .zip(bytes.iter().enumerate().skip(1))
                .filter_map(|((first_idx, first), (second_idx, second))| {
                    let merged = [first.as_slice(), second.as_slice()].concat();
                    let Some(merge_id) = self.state.0.get(&merged).copied() else {
                        return None;
                    };
                    // heap in rust is a max heap by default
                    // so we reverse to get the earliest merges at earlier positions in
                    // the sequence first
                    Some((
                        Reverse(merge_id),
                        Reverse(first_idx),
                        second_idx,
                        token_ids[first_idx],
                        token_ids[second_idx],
                        merged,
                    ))
                })
                .collect();

            while let Some((
                Reverse(merge_id),
                Reverse(first_idx),
                second_idx,
                first_id,
                second_id,
                merged,
            )) = heap.pop()
            {
                // skip if some operations refer to already merged bytes
                if token_ids[first_idx] != first_id || token_ids[second_idx] != second_id {
                    continue;
                }
                // push new potential merge operation with the previous available
                // bytes onto heap
                let mut done = true;
                if let Some(Some((merge_id, first_idx, second_idx, merged))) = bytes
                    .iter()
                    .enumerate()
                    .take(first_idx)
                    .rev()
                    .find(|&(_, prev)| !prev.is_empty())
                    .map(|(prev_idx, prev)| {
                        let merged = [prev.as_slice(), merged.as_slice()].concat();
                        let merge_id = self.state.0.get(&merged).copied();
                        Some((merge_id?, prev_idx, first_idx, merged))
                    })
                {
                    heap.push((
                        Reverse(merge_id),
                        Reverse(first_idx),
                        second_idx,
                        token_ids[first_idx],
                        token_ids[second_idx],
                        merged,
                    ));
                    done = false;
                };
                // push new potential merge operation with the next available
                // bytes onto heap
                if let Some(Some((merge_id, first_idx, second_idx, merged))) = bytes
                    .iter()
                    .enumerate()
                    .skip(second_idx + 1)
                    .find(|&(_, next)| !next.is_empty())
                    .map(|(next_idx, next)| {
                        let merged = [merged.as_slice(), next.as_slice()].concat();
                        let merge_id = self.state.0.get(&merged).copied();
                        Some((merge_id?, first_idx, next_idx, merged))
                    })
                {
                    heap.push((
                        Reverse(merge_id),
                        Reverse(first_idx),
                        second_idx,
                        token_ids[first_idx],
                        token_ids[second_idx],
                        merged,
                    ));
                    done = false;
                };
                bytes[first_idx] = merged;
                bytes[second_idx].clear();
                token_ids[first_idx] = Some(256 + merge_id);
                token_ids[second_idx] = None;
                if done {
                    break;
                }
            }

            joined_token_ids.extend(token_ids.into_iter().flatten());
        }

        joined_token_ids
    }
}

impl Tokenize for BPETokenizer {
    fn vocab_size(&self) -> usize {
        self.state.1.len() + self.special_vocab.size()
    }

    fn tokenize(
        &self,
        s: &str,
        lang: Option<&str>,
        prefix: Option<&[&str]>,
        suffix: Option<&[&str]>,
        ignore_special_tokens: bool,
    ) -> anyhow::Result<Tokenization> {
        let inputs = self.split_input(s, ignore_special_tokens);
        let mut token_ids = vec![];
        for input in inputs {
            match input {
                TokenInput::Regular(s) => {
                    token_ids.extend(self.merge_bytes(s));
                }
                TokenInput::Special(token) => {
                    token_ids.push(self.special_vocab.token_to_id(token).unwrap());
                }
            }
        }
        let token_ids = self.add_prefix_and_suffix(token_ids, lang, prefix, suffix)?;
        Ok(Tokenization::new(token_ids, TokenizationInfo::Empty))
    }

    fn de_tokenize(&self, token_ids: &[u32], ignore_special_tokens: bool) -> String {
        let mut bytes = Vec::new();
        let num_merge_ops = self.state.1.len() as u32;
        for token_id in token_ids {
            if *token_id < num_merge_ops {
                bytes.extend(&self.state.1[*token_id as usize]);
            } else if !ignore_special_tokens {
                bytes.extend(
                    self.special_vocab
                        .id_to_token(token_id)
                        .expect("invalid token id in input")
                        .as_bytes(),
                );
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }
}

#[inline]
fn max_byte_pair(stats: &BytePairStats) -> Option<BytePair> {
    stats
        .iter()
        .max_by_key(|&(_, info)| info.freq)
        .map(|(pair, _)| pair.clone())
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
struct BytePair {
    first: Vec<u8>,
    second: Vec<u8>,
}

impl BytePair {
    fn merge(&self) -> Vec<u8> {
        [&self.first[..], &self.second[..]].concat()
    }
}

#[derive(Clone, Debug)]
struct BytePairInfo {
    freq: usize,
    words: HashMap<usize, usize>,
}
type BytePairVocab = Vec<(Vec<Vec<u8>>, usize)>;
type BytePairStats = HashMap<BytePair, BytePairInfo>;
#[inline]
fn byte_pair_stats(vocab: &BytePairVocab) -> BytePairStats {
    let mut stats = BytePairStats::new();
    for (idx, (word, freq)) in vocab.iter().enumerate() {
        for i in 1..word.len() {
            let pair = BytePair {
                first: word[i - 1].to_vec(),
                second: word[i].to_vec(),
            };
            stats
                .entry(pair)
                .and_modify(|info| {
                    info.freq += freq;
                    info.words
                        .entry(idx)
                        .and_modify(|count| *count += 1)
                        .or_insert(1);
                })
                .or_insert(BytePairInfo {
                    freq: *freq,
                    words: [(idx, 1)].into(),
                });
        }
    }
    stats
}

#[inline]
fn replace_pair_in_word(word: &Vec<Vec<u8>>, pair: &BytePair) -> Vec<Vec<u8>> {
    assert!(!word.is_empty());
    let mut new_word = vec![word[0].to_vec()];
    for subword in &word[1..] {
        if new_word.last().unwrap() == &pair.first && subword == &pair.second {
            new_word.last_mut().unwrap().extend(subword);
        } else {
            new_word.push(subword.to_vec());
        }
    }
    new_word
}

type BytePairChanges = Vec<(usize, Vec<Vec<u8>>, Vec<Vec<u8>>, usize)>;
#[inline]
fn replace_pair(
    vocab: &mut BytePairVocab,
    pair: &BytePair,
    stats: &BytePairStats,
) -> BytePairChanges {
    let mut changes = Vec::new();
    for (idx, occ) in &stats[pair].words {
        if *occ < 1 {
            continue;
        }
        let (word, freq) = &vocab[*idx];
        let new_word = replace_pair_in_word(word, pair);
        changes.push((*idx, word.to_vec(), new_word.clone(), *freq));
        vocab[*idx] = (new_word, *freq);
    }
    changes
}

#[inline]
fn update_stats(stats: &mut BytePairStats, pair: &BytePair, changes: &BytePairChanges) {
    let stat = stats.get_mut(pair).unwrap();
    stat.freq = 0;
    stat.words.iter_mut().for_each(|(_, occ)| *occ = 0);
    let merged = pair.merge();
    for (idx, old_word, new_word, freq) in changes {
        let mut i = 0;
        while i < old_word.len() {
            let Some((start, _)) = old_word[i..]
            .iter()
            .find_position(|&subword| subword == &pair.first) else {
                break;
            };
            i += start;
            if i == old_word.len() - 1 || old_word[i + 1] != pair.second {
                i += 1;
                continue;
            }
            if i > 0 {
                let prev_pair = BytePair {
                    first: old_word[i - 1].to_vec(),
                    second: old_word[i].to_vec(),
                };
                let mut stat = stats.get_mut(&prev_pair).unwrap();
                stat.freq = stat.freq.saturating_sub(*freq);
                let occ = stat.words.get_mut(idx).unwrap();
                *occ = occ.saturating_sub(1);
            }
            if i < old_word.len() - 2
                && (old_word[i + 2] != pair.first
                    || i >= old_word.len() - 3
                    || old_word[i + 3] != pair.second)
            {
                let next_pair = BytePair {
                    first: old_word[i + 1].to_vec(),
                    second: old_word[i + 2].to_vec(),
                };
                let mut stat = stats.get_mut(&next_pair).unwrap();
                stat.freq = stat.freq.saturating_sub(*freq);
                let occ = stat.words.get_mut(idx).unwrap();
                *occ = occ.saturating_sub(1);
            }
            i += 2;
        }

        i = 0;
        while i < new_word.len() {
            let Some((start, _)) = new_word[i..]
            .iter()
            .find_position(|&subword| subword == &merged) else {
                break;
            };
            i += start;
            if i > 0 {
                let prev_pair = BytePair {
                    first: new_word[i - 1].to_vec(),
                    second: new_word[i].to_vec(),
                };
                stats
                    .entry(prev_pair)
                    .and_modify(|info| {
                        info.freq += *freq;
                        *info.words.entry(*idx).or_insert(0) += 1;
                    })
                    .or_insert(BytePairInfo {
                        freq: *freq,
                        words: [(*idx, 1)].into(),
                    });
            }
            if i < new_word.len() - 1 && new_word[i + 1] != merged {
                let next_pair = BytePair {
                    first: new_word[i].to_vec(),
                    second: new_word[i + 1].to_vec(),
                };
                stats
                    .entry(next_pair)
                    .and_modify(|info| {
                        info.freq += *freq;
                        *info.words.entry(*idx).or_insert(0) += 1;
                    })
                    .or_insert(BytePairInfo {
                        freq: *freq,
                        words: [(*idx, 1)].into(),
                    });
            }
            i += 1;
        }
    }
}

#[pyfunction(
    name = "train_bpe", 
    signature = (
        files,
        vocab_size,
        num_special_tokens,
        out_file,
        max_lines_per_file = None,
        normalization = Normalization::NFKC,
        num_threads = num_cpus::get() as u8,
        progress = true,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn train_bpe_py(
    files: Vec<&str>,
    vocab_size: usize,
    num_special_tokens: usize,
    out_file: &str,
    max_lines_per_file: Option<usize>,
    normalization: Option<Normalization>,
    num_threads: u8,
    progress: bool,
) -> anyhow::Result<()> {
    train_bpe(
        &files,
        vocab_size,
        num_special_tokens,
        out_file,
        max_lines_per_file,
        normalization,
        num_threads,
        progress,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn train_bpe(
    files: &[impl AsRef<Path>],
    vocab_size: usize,
    num_special_tokens: usize,
    out_file: impl AsRef<Path>,
    max_lines_per_file: Option<usize>,
    normalization: Option<Normalization>,
    num_threads: u8,
    progress: bool,
) -> anyhow::Result<()> {
    assert!(
        (vocab_size / 64) * 64 == vocab_size,
        "vocab size must be a multiple of 64 for better performance"
    );
    assert!(vocab_size <= u32::MAX as usize, "vocab size is too large");
    let max_lines_per_file = max_lines_per_file.unwrap_or(usize::MAX);
    let num_threads = num_threads.max(1);
    let num_total_lines: usize = files
        .iter()
        .map(|path| {
            let (num_lines, _) = file_size(path).expect("failed to get file size");
            num_lines.min(max_lines_per_file)
        })
        .sum();
    let path_bufs = files
        .iter()
        .map(|path| path.as_ref().to_path_buf())
        .collect::<Vec<_>>();
    let line_iter = path_bufs.into_iter().flat_map(move |path| {
        let file = File::open(path).expect("failed to open file");
        let reader = BufReader::new(file);
        reader
            .lines()
            .take(max_lines_per_file)
            .filter_map(|line| line.ok())
    });
    let line_iter = Arc::new(Mutex::new(line_iter));
    let (count_tx, count_rx) = mpsc::sync_channel(num_threads as usize);
    panic::set_hook(Box::new(move |info| {
        println!("thread panicked: {info}");
    }));
    for idx in 0..num_threads.max(1) {
        let count_tx_clone = count_tx.clone();
        let line_iter_clone = line_iter.clone();
        let _: JoinHandle<()> = Builder::new()
            .name(format!("bpe worker thread {idx}"))
            .spawn(move || {
                loop {
                    let Some(mut line) =
                        line_iter_clone.lock().expect("failed to lock line iter").next() else  {
                        return;
                    };
                    line = clean(&line, true);
                    if let Some(normalization) = normalization {
                        line = normalize(&line, normalization, true);
                    }
                    let counts: HashMap<_, _> = count_words_whitespace(&line, true)
                        .into_iter()
                        .map(|(w, c)| (w.to_string(), c))
                        .collect();
                    if count_tx_clone.send(counts).is_err() {
                        // receiver is closed, so we can return this thread
                        return;
                    };
                }
            })
            .unwrap_or_else(|_| panic!("failed building bpe worker thread {idx}"));
    }
    let pbar = progress_bar("counting words in files", num_total_lines as u64, !progress);
    drop(count_tx); // close the channel so the threads can exit
    let mut vocab: Vec<(Vec<Vec<u8>>, usize)> = count_rx
        .into_iter()
        .fold(HashMap::new(), |mut acc, counts| {
            for (word, count) in counts {
                *acc.entry(word).or_insert(0) += count;
            }
            pbar.inc(1);
            acc
        })
        .into_iter()
        .map(|(word, count)| (word.as_bytes().iter().map(|b| vec![*b]).collect(), count))
        .collect();
    pbar.finish_and_clear();
    let num_merges = vocab_size
        .saturating_sub(256)
        .saturating_sub(num_special_tokens);
    let pbar = progress_bar("performing byte merges", num_merges as u64, !progress);
    let mut merge_ops = MergeOps::new();
    if progress {
        info!(
            "found {} unique words in files, going to perform {num_merges} byte merges",
            vocab.len()
        );
    }
    let mut stats = byte_pair_stats(&vocab);
    for merge_idx in 0..num_merges {
        let Some(pair) = max_byte_pair(&stats) else {
            break;
        };
        let changes = replace_pair(&mut vocab, &pair, &stats);
        update_stats(&mut stats, &pair, &changes);
        merge_ops.insert(pair.merge(), merge_idx as u32);
        pbar.inc(1);
    }
    pbar.finish_and_clear();
    merge_ops.save(out_file)?;
    Ok(())
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum ByteGroups {
    Bytes,
    CodePoints,
}

impl IntoPy<PyObject> for ByteGroups {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            ByteGroups::Bytes => "bytes",
            ByteGroups::CodePoints => "code_points",
        }
        .into_py(py)
    }
}

impl<'a> FromPyObject<'a> for ByteGroups {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let s: String = ob.extract()?;
        let groups = match s.as_str() {
            "bytes" => ByteGroups::Bytes,
            "code_points" => ByteGroups::CodePoints,
            k => return Err(py_invalid_type_error(k, "byte groups")),
        };
        Ok(groups)
    }
}

#[derive(Clone, Debug)]
pub struct ByteTokenizerConfig {
    pub use_graphemes: bool,
    pub pad_to_multiple_of: Option<usize>,
    pub groups: ByteGroups,
    pub aggregation: GroupAggregation,
}

pub type ByteTokenizer = VocabFreeTokenizer<ByteTokenizerConfig>;

impl ByteTokenizer {
    pub fn new(
        config: ByteTokenizerConfig,
        special_config: SpecialConfig,
        language_config: Option<LanguageConfig>,
    ) -> Self {
        Self::new_with(config, special_config, language_config)
    }

    fn new_with(
        config: ByteTokenizerConfig,
        mut special_config: SpecialConfig,
        language_config: Option<LanguageConfig>,
    ) -> Self {
        if let Some(pad_to) = config.pad_to_multiple_of {
            assert!(
                pad_to.is_power_of_two(),
                "pad_to_multiple_of must be a power of two"
            );
            let mut special_tokens: HashSet<&String> =
                HashSet::from_iter(special_config.tokens.iter());
            if let Some(lang_cfg) = &language_config {
                special_tokens.insert(&lang_cfg.default_language);
                for token in lang_cfg.languages.iter() {
                    special_tokens.insert(token);
                }
            }
            let num_special_tokens = special_tokens.len();
            let num_tokens = 256 + num_special_tokens;
            let num_pad_tokens =
                (num_tokens as f64 / pad_to as f64).ceil() as usize * pad_to - num_tokens;
            special_config
                .tokens
                .extend((0..num_pad_tokens).map(|idx| format!("<extra_token_{}>", idx)));
        }
        Self::new_vocab_free_tokenizer(256, special_config, language_config, config)
    }

    fn process_input(
        &self,
        s: &str,
        prefix: Option<&[&str]>,
        suffix: Option<&[&str]>,
        ignore_special_tokens: bool,
    ) -> (Vec<u32>, TokenizationInfo) {
        let additional_prefix_tokens = prefix.map(|pfx| pfx.len()).unwrap_or(0);
        let additional_suffix_tokens = suffix.map(|sfx| sfx.len()).unwrap_or(0);

        let mut tokens = vec![];
        let group_name = match self.config.groups {
            ByteGroups::Bytes => "byte_groups",
            ByteGroups::CodePoints => "code_point_groups",
        }
        .to_string();

        // initialize groups with 1 for each prefix token
        let mut groups =
            vec![TokenGroup::Full(1); self.num_prefix_tokens() + additional_prefix_tokens];

        for input in self.split_input(s, ignore_special_tokens) {
            match input {
                TokenInput::Special(token) => {
                    let token_id = self.special_vocab.token_to_id(token).unwrap();
                    tokens.push(token_id);
                    groups.push(TokenGroup::Full(1));
                }
                TokenInput::Regular(s) => {
                    tokens.extend(s.as_bytes().iter().map(|b| *b as u32));
                    let cs = CS::new(s, self.config.use_graphemes);
                    match self.config.groups {
                        ByteGroups::Bytes => {
                            groups.extend(
                                cs.get_char_byte_lengths().into_iter().map(TokenGroup::Full),
                            );
                        }
                        ByteGroups::CodePoints => {
                            for char in cs.chars() {
                                let code_point_groups = char
                                    .code_points()
                                    .map(|code_point| TokenGroup::Full(code_point.len_utf8()))
                                    .collect();
                                groups.push(TokenGroup::Nested(code_point_groups))
                            }
                        }
                    }
                }
            }
        }

        // append group of length 1 for each suffix token
        groups.append(&mut vec![
            TokenGroup::Full(1);
            self.num_suffix_tokens() + additional_suffix_tokens
        ]);
        (
            tokens,
            TokenizationInfo::TokenGroups(HashMap::from([(
                group_name,
                (groups, self.config.aggregation),
            )])),
        )
    }
}

impl Tokenize for ByteTokenizer {
    fn vocab_size(&self) -> usize {
        256 + self.special_vocab.size()
    }

    fn tokenize(
        &self,
        s: &str,
        lang: Option<&str>,
        prefix: Option<&[&str]>,
        suffix: Option<&[&str]>,
        ignore_special_tokens: bool,
    ) -> anyhow::Result<Tokenization> {
        let (bytes, info) = self.process_input(s, prefix, suffix, ignore_special_tokens);

        Ok(Tokenization::new(
            self.add_prefix_and_suffix(bytes, lang, prefix, suffix)?,
            info,
        ))
    }

    fn de_tokenize(&self, token_ids: &[u32], ignore_special_tokens: bool) -> String {
        let mut bytes = vec![];
        for token_id in token_ids {
            if *token_id < 256 {
                bytes.push(*token_id as u8);
            } else if !ignore_special_tokens {
                bytes.extend(
                    self.special_vocab
                        .id_to_token(token_id)
                        .expect("invalid token id in input")
                        .as_bytes(),
                );
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }
}

pub struct ByT5Tokenizer {
    inner: ByteTokenizer,
}

impl ByT5Tokenizer {
    pub fn new(config: ByteTokenizerConfig) -> Self {
        // disable vocab padding for byt5 tokenizer
        let inner = ByteTokenizer::new_with(
            ByteTokenizerConfig {
                pad_to_multiple_of: None,
                ..config
            },
            SpecialConfig {
                pad: "<pad>".into(),
                tokens: vec!["<pad>".into(), "</s>".into(), "<unk>".into()],
                prefix: vec![],
                suffix: vec!["</s>".into()],
            },
            None,
        );
        Self { inner }
    }
}

impl BaseTokenize for ByT5Tokenizer {
    fn prefix_token_ids(&self) -> &[u32] {
        self.inner.prefix_token_ids()
    }

    fn suffix_token_ids(&self) -> &[u32] {
        self.inner.suffix_token_ids()
    }

    fn language_config(&self) -> Option<&LanguageConfig> {
        self.inner.language_config()
    }

    fn special_token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.special_token_to_id(token)
    }

    fn pad_token_id(&self) -> u32 {
        self.inner.pad_token_id()
    }
}

impl Tokenize for ByT5Tokenizer {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn tokenize(
        &self,
        s: &str,
        lang: Option<&str>,
        prefix: Option<&[&str]>,
        suffix: Option<&[&str]>,
        ignore_special_tokens: bool,
    ) -> anyhow::Result<Tokenization> {
        let Tokenization { token_ids, info } =
            self.inner
                .tokenize(s, lang, prefix, suffix, ignore_special_tokens)?;
        // adapt token ids to byt5 format
        // --> shift byte token_ids from 0..255 to 3..258
        // --> shift eos token
        let num_token_ids = token_ids.len();
        let mut token_ids: Vec<_> = token_ids
            .into_iter()
            .take(num_token_ids - 1)
            .map(|t| {
                assert!(t < 256);
                t + 3
            })
            .collect();
        token_ids.push(1);
        Ok(Tokenization::new(token_ids, info))
    }

    fn de_tokenize(&self, token_ids: &[u32], ignore_special_tokens: bool) -> String {
        self.inner.de_tokenize(token_ids, ignore_special_tokens)
    }
}

pub fn tokenizer(cfg: TokenizerConfig) -> anyhow::Result<Tokenizer> {
    Ok(match cfg.tokenize {
        TokenizeConfig::Character(char_cfg) => {
            Box::new(CharTokenizer::new(char_cfg, cfg.special, cfg.language))
        }
        TokenizeConfig::Byte(byte_cfg) => {
            Box::new(ByteTokenizer::new(byte_cfg, cfg.special, cfg.language))
        }
        TokenizeConfig::ByT5(byte_cfg) => Box::new(ByT5Tokenizer::new(byte_cfg)),
        TokenizeConfig::BPE(bpe_cfg) => {
            Box::new(BPETokenizer::new(bpe_cfg, cfg.special, cfg.language)?)
        }
        TokenizeConfig::Dummy(d) => Box::new(DummyTokenizer::new(d)),
    })
}

#[pyclass]
#[pyo3(name = "Tokenizer")]
struct PyTokenizer {
    tokenizer: Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    #[staticmethod]
    fn from_config(config: TokenizerConfig) -> anyhow::Result<Self> {
        Ok(PyTokenizer {
            tokenizer: tokenizer(config)?,
        })
    }

    #[pyo3(signature = (s, lang = None, prefix = None, suffix = None, ignore_special_tokens = true))]
    fn tokenize(
        &self,
        s: &str,
        lang: Option<&str>,
        prefix: Option<Vec<&str>>,
        suffix: Option<Vec<&str>>,
        ignore_special_tokens: bool,
    ) -> anyhow::Result<Tokenization> {
        self.tokenizer.tokenize(
            s,
            lang,
            prefix.as_deref(),
            suffix.as_deref(),
            ignore_special_tokens,
        )
    }

    fn special_token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.special_token_to_id(token)
    }

    #[pyo3(signature = (token_ids, ignore_special_tokens = true))]
    fn de_tokenize(&self, token_ids: Vec<u32>, ignore_special_tokens: bool) -> String {
        self.tokenizer
            .de_tokenize(&token_ids, ignore_special_tokens)
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    fn num_prefix_tokens(&self) -> usize {
        self.tokenizer.num_prefix_tokens()
    }

    fn num_suffix_tokens(&self) -> usize {
        self.tokenizer.num_suffix_tokens()
    }

    fn pad_token_id(&self) -> u32 {
        self.tokenizer.pad_token_id()
    }
}

/// A submodule containing functionality to tokenize text into tokens.
/// Currently supported tokenization schemes are:
/// - character level tokenization
/// - byte level tokenization
pub(super) fn add_submodule(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "tokenization")?;
    m.add_class::<PyTokenizer>()?;
    m.add_class::<Tokenization>()?;
    m.add_class::<SpecialTokens>()?;
    m.add_class::<LanguageTokens>()?;
    m.add_function(wrap_pyfunction!(train_bpe_py, m)?)?;
    parent_module.add_submodule(m)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::PathBuf};

    use log::info;
    use numpy::ndarray::{Array1, Array2};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use crate::{
        tokenization::{
            token_groups_to_sparse_coo_matrix, train_bpe, BPETokenizer, ByT5Tokenizer, ByteGroups,
            ByteTokenizer, ByteTokenizerConfig, CharTokenizer, CharTokenizerConfig,
            GroupAggregation, SparseCoo, SpecialConfig, TokenGroup, Tokenization, TokenizationInfo,
            Tokenize,
        },
        unicode::Normalization,
    };

    use super::BPETokenizerConfig;

    #[test]
    fn test_char_tokenizer() {
        let tok = CharTokenizer::new(
            CharTokenizerConfig {
                use_graphemes: true,
            },
            SpecialConfig::default(),
            None,
        );
        let text = "a täst";
        let Tokenization { token_ids, .. } = tok.tokenize(text, None, None, None, true).unwrap();
        assert_eq!(token_ids.len(), 6 + 2);
        assert_eq!(token_ids[4], tok.unk_token_id());
        assert_eq!(tok.de_tokenize(&token_ids, true), "a tst".to_string());
        assert_eq!(
            tok.de_tokenize(&token_ids, false),
            "(bos)a t(unk)st(eos)".to_string()
        );
        let text = "a (pad)täst";
        let Tokenization { token_ids, .. } = tok.tokenize(text, None, None, None, false).unwrap();
        assert_eq!(token_ids.len(), 7 + 2);
        assert_eq!(token_ids[3], tok.pad_token_id);
        assert_eq!(token_ids[5], tok.unk_token_id());
        assert_eq!(tok.de_tokenize(&token_ids, true), "a tst".to_string());
        assert_eq!(
            tok.de_tokenize(&token_ids, false),
            "(bos)a (pad)t(unk)st(eos)".to_string()
        );
        let text = "a (pad)täst";
        let Tokenization { token_ids, .. } = tok.tokenize(text, None, None, None, true).unwrap();
        assert_eq!(token_ids.len(), 11 + 2);
        assert_eq!(tok.de_tokenize(&token_ids, true), "a (pad)tst".to_string());
        assert_eq!(
            tok.de_tokenize(&token_ids, false),
            "(bos)a (pad)t(unk)st(eos)".to_string()
        );
    }

    #[test]
    fn test_byte_tokenizer() {
        let tokenize_cfg = ByteTokenizerConfig {
            use_graphemes: true,
            pad_to_multiple_of: Some(128),
            groups: ByteGroups::Bytes,
            aggregation: GroupAggregation::Mean,
        };
        let tok = ByteTokenizer::new(tokenize_cfg.clone(), SpecialConfig::default(), None);
        assert_eq!(tok.vocab_size(), 384);
        let text = "a täst";
        let Tokenization { token_ids, info } = tok.tokenize(text, None, None, None, true).unwrap();
        assert_eq!(
            token_ids[1..token_ids.len() - 1]
                .iter()
                .map(|tok| *tok as u8)
                .collect::<Vec<u8>>(),
            text.as_bytes().clone()
        );
        match info {
            TokenizationInfo::TokenGroups(groups) => {
                assert_eq!(
                    groups,
                    HashMap::from([(
                        "byte_groups".to_string(),
                        (
                            [1, 1, 1, 1, 2, 1, 1, 1]
                                .into_iter()
                                .map(|l| TokenGroup::Full(l))
                                .collect(),
                            GroupAggregation::Mean
                        )
                    )])
                )
            }
            _ => panic!("wrong info"),
        };
        assert_eq!(token_ids.len(), 7 + 2);
        assert_eq!(tok.de_tokenize(&token_ids, true), text.to_string());
        let tokenize_cfg = ByteTokenizerConfig {
            groups: ByteGroups::CodePoints,
            ..tokenize_cfg
        };
        let tok = ByteTokenizer::new(tokenize_cfg, SpecialConfig::default(), None);
        let text = "a täst";
        let Tokenization { token_ids, info } = tok.tokenize(text, None, None, None, true).unwrap();
        assert_eq!(
            token_ids[1..token_ids.len() - 1]
                .iter()
                .map(|tok| *tok as u8)
                .collect::<Vec<u8>>(),
            text.as_bytes().clone()
        );
        match info {
            TokenizationInfo::TokenGroups(groups) => {
                assert_eq!(
                    groups,
                    HashMap::from([(
                        "code_point_groups".to_string(),
                        (
                            vec![TokenGroup::Full(1)]
                                .into_iter()
                                .chain(
                                    [[1].as_slice(), &[1], &[1], &[2], &[1], &[1]]
                                        .into_iter()
                                        .map(|vs| {
                                            TokenGroup::Nested(
                                                vs.into_iter()
                                                    .map(|v| TokenGroup::Full(*v))
                                                    .collect(),
                                            )
                                        })
                                )
                                .chain(vec![TokenGroup::Full(1)])
                                .collect(),
                            GroupAggregation::Mean
                        )
                    )])
                )
            }
            _ => panic!("wrong info"),
        };
    }

    #[test]
    fn test_bpe_tokenizer() {
        let _ = env_logger::try_init();

        let special_config = SpecialConfig::default();
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let file = base.clone().join("resources/test/multi30k.txt");
        let out_file = base.clone().join("resources/test/multi30k.bpe.merges");
        train_bpe(
            &[file],
            1024,
            special_config.tokens.len(),
            &out_file,
            None,
            Some(Normalization::NFKC),
            num_cpus::get() as u8,
            true,
        )
        .unwrap();

        let bpe_config = BPETokenizerConfig {
            merge_file: out_file,
            max_vocab_size: None,
            use_graphemes: true,
        };
        let bpe = BPETokenizer::new(bpe_config, special_config, None).unwrap();
        info!("loaded bpe tokenizer");
        let s = "this is a long reading couple restaurant";
        let token_ids = bpe.tokenize(s, None, None, None, true).unwrap().token_ids;
        info!("token ids: {token_ids:?}");
        let tokens: Vec<_> = token_ids
            .iter()
            .filter_map(|t| bpe.state.1.get(*t as usize))
            .map(|b| String::from_utf8_lossy(b).to_string())
            .collect();
        info!("tokens: {tokens:?}");
        let ds = bpe.de_tokenize(&token_ids, true);
        assert_eq!(s, ds);
        info!("de-tokenized: \"{ds}\"");

        let mut rng = ChaCha8Rng::seed_from_u64(22);
        for _ in 0..200 {
            let s: String = (&mut rng)
                .sample_iter::<char, _>(rand::distributions::Standard)
                .take(256)
                .collect();

            let token_ids = bpe.tokenize(&s, None, None, None, true).unwrap().token_ids;
            let ds = bpe.de_tokenize(&token_ids, true);
            assert_eq!(s, ds);
        }
    }

    #[test]
    fn test_byt5_tokenizer() {
        let tokenize_cfg = ByteTokenizerConfig {
            use_graphemes: true,
            pad_to_multiple_of: Some(128),
            groups: ByteGroups::Bytes,
            aggregation: GroupAggregation::Mean,
        };
        let tok = ByT5Tokenizer::new(tokenize_cfg);
        assert_eq!(tok.vocab_size(), 259);
        let Tokenization { token_ids, info: _ } =
            tok.tokenize("a täst", None, None, None, true).unwrap();
        assert_eq!(token_ids, vec![100, 35, 119, 198, 167, 118, 119, 1]);
    }

    #[test]
    fn test_token_groups_to_sparse_coo_matrix() {
        // one stage grouping
        let grouping = (
            [1, 1, 1, 1, 2, 1, 1, 1]
                .into_iter()
                .map(|l| TokenGroup::Full(l))
                .collect(),
            GroupAggregation::Mean,
        );
        let SparseCoo {
            indices,
            values,
            size,
            group_lengths,
        } = token_groups_to_sparse_coo_matrix(&[&grouping], &[9]).unwrap();
        assert_eq!(
            indices,
            Array2::from_shape_vec(
                (3, 9),
                vec![
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8
                ]
            )
            .unwrap()
        );
        assert_eq!(size, vec![1, 8, 9]);
        assert_eq!(
            values,
            Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0])
        );
        assert_eq!(group_lengths, vec![8]);

        // two stage grouping
        let grouping = (
            [[1, 1, 1, 1].as_slice(), &[2, 1, 1, 1]]
                .into_iter()
                .map(|vs| {
                    TokenGroup::Nested(vs.into_iter().map(|v| TokenGroup::Full(*v)).collect())
                })
                .collect(),
            GroupAggregation::Mean,
        );
        let SparseCoo {
            indices,
            values,
            size,
            group_lengths,
        } = token_groups_to_sparse_coo_matrix(&[&grouping], &[9]).unwrap();
        assert_eq!(
            indices,
            Array2::from_shape_vec(
                (3, 9),
                vec![
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8
                ]
            )
            .unwrap()
        );
        assert_eq!(size, vec![1, 2, 9]);
        assert_eq!(
            values,
            Array1::from_vec(vec![0.25, 0.25, 0.25, 0.25, 0.125, 0.125, 0.25, 0.25, 0.25])
        );
        assert_eq!(group_lengths, vec![2]);
    }
}
