use crate::text::{clean, count_words_whitespace, file_size, SPLIT_WORD_WHITESPACE_PATTERN};
use crate::unicode::{self, normalize, Normalization, CS};
use crate::utils::{
    accumulate, progress_bar, py_invalid_type_error, py_required_key_error, SerializeMsgPack,
};
use anyhow::anyhow;
use hft::{NormalizedString, Normalizer};
use itertools::Itertools;
use log::info;
use numpy::ndarray::{Array1, Array2};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyDict, PyList, PyTuple};
use regex::{escape, Regex};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::borrow::Borrow;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::hash::Hash;
use std::io::{BufRead, BufReader};
use std::iter::repeat;
use std::panic;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{sleep, Builder, JoinHandle};
use std::time::Duration;
use text_utils_grammar::LR1GrammarParser;
use tokenizers::{self as hft, Decoder};

pub const UNK: &str = "<unk>";
pub const BOS: &str = "<bos>";
pub const EOS: &str = "<eos>";
pub const PAD: &str = "<pad>";
pub const SPECIAL_TOKENS: [&str; 4] = [UNK, BOS, EOS, PAD];
pub const DEFAULT_PREFIX_TOKENS: [&str; 1] = [BOS];
pub const DEFAULT_SUFFIX_TOKENS: [&str; 1] = [EOS];

#[pyclass]
pub struct SpecialTokens {}

#[pymethods]
impl SpecialTokens {
    #[classattr]
    const UNK: &'static str = UNK;
    #[classattr]
    const BOS: &'static str = BOS;
    #[classattr]
    const EOS: &'static str = EOS;
    #[classattr]
    const PAD: &'static str = PAD;
}

// language tokens
pub const LANG_UNK: &str = "<lang:unk>";

#[pyclass]
pub struct LanguageTokens {}

#[pymethods]
impl LanguageTokens {
    #[classattr]
    const UNK: &'static str = LANG_UNK;
}

/// Config for special tokens and options regarding special tokens
#[derive(Debug, Clone)]
pub struct SpecialConfig {
    pub pad: String,
    pub tokens: Vec<String>,
    pub prefix: Vec<String>,
    pub suffix: Vec<String>,
}

fn validate_special_config(special_config: &SpecialConfig) -> anyhow::Result<()> {
    if !special_config.tokens.contains(&special_config.pad) {
        return Err(anyhow!("pad token not in special tokens",));
    }
    if special_config
        .prefix
        .iter()
        .any(|tok| !special_config.tokens.contains(tok))
    {
        return Err(anyhow!(
            "one or more prefix tokens are not in special tokens",
        ));
    }
    if special_config
        .suffix
        .iter()
        .any(|tok| !special_config.tokens.contains(tok))
    {
        return Err(anyhow!(
            "one or more suffix tokens are not in special tokens",
        ));
    }
    Ok(())
}

impl Default for SpecialConfig {
    fn default() -> Self {
        Self {
            pad: PAD.to_string(),
            tokens: SPECIAL_TOKENS.iter().map(|s| s.to_string()).collect(),
            prefix: vec![],
            suffix: vec![],
        }
    }
}

impl<'a> FromPyObject<'a> for SpecialConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let Some(pad) = d.get_item("pad")? else {
            return Err(py_required_key_error("pad", "special config"));
        };
        let Some(tokens) = d.get_item("tokens")? else {
            return Err(py_required_key_error("tokens", "special config"));
        };
        Ok(Self {
            pad: pad.extract()?,
            tokens: tokens.extract()?,
            prefix: if let Some(value) = d.get_item("prefix")? {
                value.extract()?
            } else {
                vec![]
            },
            suffix: if let Some(value) = d.get_item("suffix")? {
                value.extract()?
            } else {
                vec![]
            },
        })
    }
}

#[derive(Clone, Debug)]
pub enum TokenizationConstraintConfig {
    LR1Grammar {
        lexer: String,
        grammar: String,
        skip_ignore_tokens: bool,
    },
}

pub enum TokenizationConstraint {
    LR1Grammar {
        parser: LR1GrammarParser,
        skip_ignore_tokens: bool,
    },
}

impl TokenizationConstraint {
    pub fn from_config(config: TokenizationConstraintConfig) -> anyhow::Result<Self> {
        match config {
            TokenizationConstraintConfig::LR1Grammar {
                lexer,
                grammar,
                skip_ignore_tokens,
            } => {
                let parser = LR1GrammarParser::from_files(&grammar, &lexer).map_err(|e| {
                    anyhow!(
                        "failed to create LR1 grammar parser from lexer {lexer} and grammar {grammar}: {e}"
                    )
                })?;
                Ok(Self::LR1Grammar {
                    parser,
                    skip_ignore_tokens,
                })
            }
        }
    }
}

impl<'a> FromPyObject<'a> for TokenizationConstraintConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let Some(constraint_type) = d.get_item("type")? else {
            return Err(py_required_key_error("type", "generation config"));
        };
        let constraint_type: String = constraint_type.extract()?;
        let constraint = match constraint_type.as_str() {
            "lr1_grammar" => {
                let Some(lexer) = d.get_item("lexer")? else {
                    return Err(py_required_key_error(
                        "lexer",
                        "tokenization constraint config",
                    ));
                };
                let Some(grammar) = d.get_item("grammar")? else {
                    return Err(py_required_key_error(
                        "grammar",
                        "tokenization constraint config",
                    ));
                };
                let skip_ignore_tokens = d
                    .get_item("skip_ignore_tokens")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(false);
                TokenizationConstraintConfig::LR1Grammar {
                    lexer: lexer.extract()?,
                    grammar: grammar.extract()?,
                    skip_ignore_tokens,
                }
            }
            k => {
                return Err(py_invalid_type_error(k, "tokenization constraint config"));
            }
        };
        Ok(constraint)
    }
}

/// This is a tokenizer config, containing configs for special tokens, language,
/// and the actual tokenize config inside it.
#[derive(Clone, Debug)]
pub struct TokenizerConfig {
    pub tokenize: TokenizeConfig,
    pub special: SpecialConfig,
}

impl<'a> FromPyObject<'a> for TokenizerConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        Ok(Self {
            tokenize: d
                .get_item("tokenize")?
                .ok_or_else(|| py_required_key_error("tokenize", "tokenizer config"))?
                .extract()?,
            special: d
                .get_item("special")?
                .ok_or_else(|| py_required_key_error("special", "tokenizer config"))?
                .extract()?,
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
    Huggingface(String),
}

impl IntoPy<PyObject> for TokenizeConfig {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let d = PyDict::new_bound(py);
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
            TokenizeConfig::Huggingface(path) => {
                d.set_item("path", path).unwrap();
                "huggingface"
            }
        };
        d.set_item("type", tokenizer_type).unwrap();
        d.to_object(py)
    }
}

impl<'a> FromPyObject<'a> for TokenizeConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let Some(tokenizer_type) = d.get_item("type")? else {
            return Err(py_required_key_error("type", "tokenizer config"));
        };
        let tokenizer_type: String = tokenizer_type.extract()?;
        let tokenizer_config = match tokenizer_type.as_str() {
            "character" => {
                let use_graphemes: bool = if let Some(value) = d.get_item("use_graphemes")? {
                    value.extract()?
                } else {
                    true
                };
                TokenizeConfig::Character(CharTokenizerConfig { use_graphemes })
            }
            name @ ("byte" | "byt5") => {
                let use_graphemes: bool = if let Some(value) = d.get_item("use_graphemes")? {
                    value.extract()?
                } else {
                    true
                };
                let Some(groups) = d.get_item("groups")? else {
                    return Err(py_required_key_error(
                        "groups",
                        format!("{name} tokenizer config"),
                    ));
                };
                let agg: GroupAggregation = if let Some(value) = d.get_item("aggregation")? {
                    value.extract()?
                } else {
                    GroupAggregation::Mean
                };
                let pad_to_multiple_of = if let Some(value) = d.get_item("pad_to_multiple_of")? {
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
                let use_graphemes: bool = if let Some(value) = d.get_item("use_graphemes")? {
                    value.extract()?
                } else {
                    true
                };
                let Some(merge_file) = d.get_item("merge_file")? else {
                    return Err(py_required_key_error("merge_file", "bpe tokenizer config"));
                };
                let max_vocab_size: Option<usize> =
                    if let Some(value) = d.get_item("max_vocab_size")? {
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
                let millis: u64 = if let Some(value) = d.get_item("delay")? {
                    value.extract()?
                } else {
                    0
                };
                TokenizeConfig::Dummy(Duration::from_millis(millis))
            }
            "huggingface" => {
                let Some(path) = d.get_item("path")? else {
                    return Err(py_required_key_error(
                        "path",
                        "huggingface tokenizer config",
                    ));
                };
                TokenizeConfig::Huggingface(path.extract()?)
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
        let d = PyDict::new_bound(py);
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
            self.indices.into_pyarray_bound(py),
            self.values.into_pyarray_bound(py),
            self.size,
            self.group_lengths,
        )
            .into_py(py)
    }
}

pub type PaddingMask = Array2<bool>;

impl IntoPy<PyObject> for TensorizedTokenizationInfo {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let d = PyDict::new_bound(py);
        if let TensorizedTokenizationInfo::TokenGroups(matrices) = self {
            for (name, (scoo, pad_mask)) in matrices {
                let t = PyTuple::new_bound(
                    py,
                    &[
                        scoo.into_py(py),
                        pad_mask.into_pyarray_bound(py).into_py(py),
                    ],
                );
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

pub fn padding_mask(lengths: &[usize]) -> PaddingMask {
    let batch_size = lengths.len();
    let max_length = lengths.iter().max().copied().unwrap_or(0);
    let mut mask = Vec::with_capacity(batch_size * max_length);
    for &len in lengths {
        mask.extend(repeat(true).take(len));
        mask.extend(repeat(false).take(max_length - len));
    }
    PaddingMask::from_shape_vec((batch_size, max_length), mask).expect("should not fail")
}

impl IntoPy<PyObject> for TokenizationInfo {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let d = PyDict::new_bound(py);
        let info_type = match self {
            TokenizationInfo::Empty => "empty",
            TokenizationInfo::TokenGroups(token_groups) => {
                for (group_name, (groups, agg)) in token_groups {
                    let l = PyList::empty_bound(py);
                    for group in groups {
                        l.append(group).unwrap();
                    }
                    let gd = PyDict::new_bound(py);
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
    }

    fn num_suffix_tokens(&self) -> usize {
        self.suffix_token_ids().len()
    }

    fn num_special_tokens(&self) -> usize {
        self.num_prefix_tokens() + self.num_suffix_tokens()
    }

    fn normalize(&self, s: &str) -> String {
        unicode::normalize(s, Normalization::NFKC, true)
    }

    fn prefix_token_ids(&self) -> &[u32];

    fn suffix_token_ids(&self) -> &[u32];

    fn pad_token_id(&self) -> u32;
}

pub trait Tokenize: BaseTokenize {
    fn vocab_size(&self) -> usize;

    fn get_vocab(&self) -> anyhow::Result<Vec<Vec<u8>>>;

    fn get_continuations(&self, initial: bool) -> anyhow::Result<Vec<Vec<u8>>>;

    fn token_to_id(&self, token: &str) -> Option<u32>;

    fn id_to_token(&self, id: u32) -> Option<Vec<u8>>;

    fn tokenize(&self, s: &str, ignore_special_tokens: bool) -> anyhow::Result<Tokenization>;

    fn tokenize_with_constraint(
        &self,
        s: &str,
        ignore_special_tokens: bool,
        constraint: &TokenizationConstraint,
    ) -> anyhow::Result<Tokenization> {
        match constraint {
            TokenizationConstraint::LR1Grammar {
                parser,
                skip_ignore_tokens,
            } => {
                let lexemes = parser.lex(s).map_err(|e| {
                    anyhow!("tokenizing with grammar constraint failed with a lexer error: {e}")
                })?;
                let mut all_token_ids = vec![];
                let num_lexemes = lexemes.len();
                for (i, (lexeme, (start, len))) in lexemes.into_iter().enumerate() {
                    if *skip_ignore_tokens && lexeme.is_none() {
                        continue;
                    }
                    let tokenization =
                        self.tokenize(&s[start..start + len], ignore_special_tokens)?;
                    if !matches!(tokenization.info, TokenizationInfo::Empty) {
                        return Err(anyhow!(
                            "default implementation does not support tokenization info with grammar constraint"
                        ));
                    }
                    let pfx = if i == 0 { 0 } else { self.num_prefix_tokens() };
                    let sfx = if i == num_lexemes - 1 {
                        0
                    } else {
                        self.num_suffix_tokens()
                    };
                    let num_tokens = tokenization.token_ids.len() - pfx - sfx;
                    all_token_ids.extend(
                        tokenization
                            .token_ids
                            .into_iter()
                            .skip(pfx)
                            .take(num_tokens),
                    );
                }
                Ok(Tokenization::new(all_token_ids, TokenizationInfo::Empty))
            }
        }
    }

    fn de_tokenize(&self, token_ids: &[u32], ignore_special_tokens: bool)
        -> anyhow::Result<String>;
}

/// A base struct for a tokenizer,
/// allows custom tokenizers to be built by setting config and state
pub struct BaseTokenizer<Config = (), State = ()> {
    prefix_token_ids: Vec<u32>,
    suffix_token_ids: Vec<u32>,
    pad_token_id: u32,
    state: State,
    config: Config,
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
        config: Config,
        state: State,
    ) -> Self {
        if let Err(e) = validate_special_config(&special_config) {
            panic!("invalid special config: {e}");
        }
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

    fn add_prefix_and_suffix(&self, token_ids: Vec<u32>) -> Vec<u32> {
        self.prefix_token_ids()
            .iter()
            .cloned()
            .chain(token_ids)
            .chain(self.suffix_token_ids().iter().cloned())
            .collect()
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
        config: Config,
    ) -> Self {
        Self::new_base_tokenizer(special_offset, special_config, config, ())
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
        config: Config,
    ) -> Self {
        let vocab = Vocab::build(tokens, 0);
        // add unk token to special config
        special_config.tokens.push(unk_token.clone());
        Self::new_base_tokenizer(
            vocab.size() as u32,
            special_config,
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
        config: Config,
    ) -> anyhow::Result<Self> {
        let vocab = Vocab::from_file(vocab_file, 0)?;
        // add unk token to special config
        special_config.tokens.push(unk_token.clone());
        Ok(Self::new_base_tokenizer(
            vocab.size() as u32,
            special_config,
            config,
            (unk_token, vocab),
        ))
    }
}

pub trait ToBytes {
    fn to_bytes(&self) -> Vec<u8>;
}

pub trait FromBytes: Sized {
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self>;
}

impl ToBytes for char {
    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = [0; 4];
        self.encode_utf8(&mut buf).as_bytes().to_vec()
    }
}

impl FromBytes for char {
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let s = String::from_utf8(bytes.to_vec())?;
        let chars: Vec<_> = s.chars().collect();
        if chars.len() != 1 {
            return Err(anyhow!("expected a single char, but got {}", chars.len()));
        }
        Ok(chars[0])
    }
}

impl ToBytes for String {
    fn to_bytes(&self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }
}

impl FromBytes for String {
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(String::from_utf8(bytes.to_vec())?)
    }
}

impl ToBytes for Vec<u8> {
    fn to_bytes(&self) -> Vec<u8> {
        self.clone()
    }
}

impl FromBytes for Vec<u8> {
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(bytes.to_vec())
    }
}

impl<Token, Config> Tokenize for VocabTokenizer<Token, Config>
where
    Token: PartialEq + Eq + Hash + Send + Sync + Clone + FromBytes + ToBytes + 'static,
    Config: Send + Sync + 'static,
    Self: VocabTokenize<Token>,
{
    fn vocab_size(&self) -> usize {
        self.state.1.size() + self.special_vocab.size()
    }

    fn get_vocab(&self) -> anyhow::Result<Vec<Vec<u8>>> {
        let mut vocab = BTreeMap::new();
        for (token, &id) in &self.state.1.vocab {
            assert!(vocab.insert(id, token.to_bytes()).is_none());
        }
        for (special, &id) in &self.special_vocab.vocab {
            assert!(vocab.insert(id, special.to_bytes()).is_none());
        }
        Ok(vocab.into_values().collect())
    }

    fn get_continuations(&self, _: bool) -> anyhow::Result<Vec<Vec<u8>>> {
        self.get_vocab()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        if let Some(id) = self.special_vocab.token_to_id(token) {
            Some(id)
        } else {
            let token = Token::from_bytes(token.as_bytes()).ok()?;
            self.state.1.token_to_id(&token)
        }
    }

    fn id_to_token(&self, id: u32) -> Option<Vec<u8>> {
        if let Some(token) = self.special_vocab.id_to_token(&id) {
            Some(token.to_bytes())
        } else {
            let token = self.state.1.id_to_token(&id)?;
            Some(token.to_bytes())
        }
    }

    fn tokenize(&self, s: &str, ignore_special_tokens: bool) -> anyhow::Result<Tokenization> {
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
        Ok(Tokenization::new(
            self.add_prefix_and_suffix(token_ids),
            tokenization_info,
        ))
    }

    fn de_tokenize(
        &self,
        token_ids: &[u32],
        ignore_special_tokens: bool,
    ) -> anyhow::Result<String> {
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
                    .ok_or_else(|| anyhow!("unknown special token id {token_id}"))?;
                parts.push(special_token.to_string());
            };
        }
        // dont forget to join the remaining tokens
        if !tokens.is_empty() {
            parts.push(self.join_tokens(&tokens));
        }
        Ok(self.join_parts(&parts))
    }
}

/// Dummy tokenizer that just waits a specified time in its tokenize function.
/// Used for testing only.
pub type DummyTokenizer = VocabFreeTokenizer<Duration>;

impl DummyTokenizer {
    fn new(delay: Duration) -> Self {
        Self::new_vocab_free_tokenizer(0, SpecialConfig::default(), delay)
    }
}

impl Tokenize for DummyTokenizer {
    fn vocab_size(&self) -> usize {
        0
    }

    fn get_vocab(&self) -> anyhow::Result<Vec<Vec<u8>>> {
        Err(anyhow!("dummy tokenizer does not have a vocab"))
    }

    fn get_continuations(&self, _: bool) -> anyhow::Result<Vec<Vec<u8>>> {
        Err(anyhow!("dummy tokenizer does not have continuations"))
    }

    fn token_to_id(&self, _: &str) -> Option<u32> {
        None
    }

    fn id_to_token(&self, _: u32) -> Option<Vec<u8>> {
        None
    }

    fn tokenize(&self, _: &str, _: bool) -> anyhow::Result<Tokenization> {
        sleep(self.config);
        Ok(Tokenization::new(vec![], TokenizationInfo::Empty))
    }

    fn de_tokenize(&self, _: &[u32], _: bool) -> anyhow::Result<String> {
        Ok(String::default())
    }
}

pub struct HuggingfaceTokenizer {
    inner: hft::Tokenizer,
    pad_token_id: u32,
    prefix_token_ids: Vec<u32>,
    suffix_token_ids: Vec<u32>,
}

impl HuggingfaceTokenizer {
    pub fn new(path: impl AsRef<str>, special_config: SpecialConfig) -> anyhow::Result<Self> {
        let mut tok = hft::Tokenizer::from_file(path.as_ref()).map_err(|err| {
            anyhow!(
                "error loading huggingface tokenizer from path {}: {err}",
                path.as_ref()
            )
        })?;
        let enc = tok.encode("this is a test", true).map_err(|err| {
            anyhow!("error encoding test string with huggingface tokenizer: {err}")
        })?;
        let mask = enc.get_special_tokens_mask();
        if mask.iter().all(|m| *m > 0) {
            return Err(anyhow!(
                "all tokens in test string are special tokens, this is not supported"
            ));
        }
        let ids = enc.get_ids();
        let mut prefix_token_ids: Vec<_> = ids
            .iter()
            .zip(mask)
            .take_while(|&(_, m)| *m > 0)
            .map(|(id, _)| *id)
            .collect();
        let mut suffix_token_ids: Vec<_> = ids
            .iter()
            .zip(mask)
            .rev()
            .take_while(|&(_, m)| *m > 0)
            .map(|(id, _)| *id)
            .collect();
        for token in &special_config.tokens {
            if tok.token_to_id(token).is_some() {
                continue;
            }
            tok.add_tokens(&[hft::AddedToken {
                content: token.clone(),
                single_word: false,
                lstrip: false,
                rstrip: false,
                normalized: false,
                special: false,
            }]);
        }
        if !special_config.prefix.is_empty() {
            prefix_token_ids.extend(special_config.prefix.iter().map(|token| {
                tok.token_to_id(token).unwrap_or_else(|| {
                    panic!("token '{token}' not found in huggingface tokenizer",)
                })
            }));
        }
        if !special_config.suffix.is_empty() {
            suffix_token_ids.extend(special_config.suffix.iter().map(|token| {
                tok.token_to_id(token).unwrap_or_else(|| {
                    panic!("token '{token}' not found in huggingface tokenizer",)
                })
            }));
        }
        let Some(pad_token_id) = tok.token_to_id(&special_config.pad) else {
            return Err(anyhow!(
                "pad token {} not found in huggingface tokenizer",
                special_config.pad
            ));
        };
        Ok(Self {
            inner: tok,
            pad_token_id,
            prefix_token_ids,
            suffix_token_ids,
        })
    }
}

impl BaseTokenize for HuggingfaceTokenizer {
    fn prefix_token_ids(&self) -> &[u32] {
        &self.prefix_token_ids
    }

    fn suffix_token_ids(&self) -> &[u32] {
        &self.suffix_token_ids
    }

    fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    fn normalize(&self, s: &str) -> String {
        if let Some(normalizer) = self.inner.get_normalizer() {
            let mut normalized = NormalizedString::from(s);
            normalizer
                .normalize(&mut normalized)
                .expect("huggingface tokenizer failed during normalization");
            normalized.get().to_string()
        } else {
            s.to_string()
        }
    }
}

impl Tokenize for HuggingfaceTokenizer
where
    Self: BaseTokenize,
{
    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    fn get_vocab(&self) -> anyhow::Result<Vec<Vec<u8>>> {
        Ok(self
            .inner
            .get_vocab(true)
            .into_iter()
            .sorted_by_key(|(_, id)| *id)
            .map(|(k, _)| k.into_bytes())
            .collect())
    }

    fn get_continuations(&self, initial: bool) -> anyhow::Result<Vec<Vec<u8>>> {
        let pad_token = self
            .id_to_token(self.pad_token_id())
            .ok_or_else(|| {
                anyhow!(
                    "pad token id {} not found in huggingface tokenizer",
                    self.pad_token_id()
                )
            })
            .and_then(|t| Ok(String::from_utf8(t)?))?;
        let is_byte_fallback = |token: &str| -> Option<u8> {
            if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
                u8::from_str_radix(&token[3..5], 16).ok()
            } else {
                None
            }
        };
        let decode_fn = |token: String| -> anyhow::Result<Vec<u8>> {
            if let Some(dec) = self.inner.get_decoder() {
                if initial {
                    dec.decode(vec![token])
                        .map(|s| s.into_bytes())
                        .map_err(|e| anyhow!("{e}"))
                } else {
                    dec.decode(vec![pad_token.clone(), token, pad_token.clone()])
                        .map(|s| {
                            let mut bytes = s.into_bytes();
                            bytes.truncate(bytes.len() - pad_token.len());
                            // split off padding bytes
                            bytes.split_off(pad_token.len())
                        })
                        .map_err(|e| anyhow!("{e}"))
                }
            } else {
                Ok(token.into_bytes())
            }
        };
        self.inner
            .get_vocab(true)
            .into_iter()
            .sorted_by_key(|(_, id)| *id)
            .map(|(k, _)| match self.inner.get_model() {
                hft::ModelWrapper::BPE(bpe) => {
                    if !bpe.byte_fallback {
                        decode_fn(k)
                    } else if let Some(b) = is_byte_fallback(&k) {
                        Ok(vec![b])
                    } else {
                        decode_fn(k)
                    }
                }
                _ => decode_fn(k),
            })
            .collect()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<Vec<u8>> {
        self.inner.id_to_token(id).map(|s| s.into_bytes())
    }

    fn tokenize(&self, s: &str, ignore_special_tokens: bool) -> anyhow::Result<Tokenization> {
        let enc = self.inner.encode(s, !ignore_special_tokens).map_err(|e| {
            anyhow!("error encoding input {s:?} with huggingface tokenizer: {e:?}",)
        })?;
        Ok(Tokenization::new(
            enc.get_ids().to_vec(),
            TokenizationInfo::Empty,
        ))
    }

    fn de_tokenize(
        &self,
        token_ids: &[u32],
        ignore_special_tokens: bool,
    ) -> anyhow::Result<String> {
        self.inner
            .decode(token_ids, ignore_special_tokens)
            .map_err(|e| {
                anyhow!("error decoding token ids {token_ids:?} with huggingface tokenizer: {e}",)
            })
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
    pub fn new(config: CharTokenizerConfig, special_config: SpecialConfig) -> Self {
        Self::new_vocab_tokenizer(
            CHARS.chars().collect(),
            UNK.to_string(),
            special_config,
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
    pub fn new(config: BPETokenizerConfig, special_config: SpecialConfig) -> anyhow::Result<Self> {
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
                    let merge_id = self.state.0.get(&merged).copied()?;
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

    fn get_vocab(&self) -> anyhow::Result<Vec<Vec<u8>>> {
        let mut vocab: BTreeMap<_, _> = (0..256).map(|b| (b as u32, vec![b as u8])).collect();
        for (merge, id) in &self.state.0 {
            assert!(vocab.insert(256 + id, merge.clone()).is_none());
        }
        for (special, &id) in &self.special_vocab.vocab {
            assert!(vocab.insert(id, special.to_bytes()).is_none());
        }
        Ok(vocab.into_values().collect())
    }

    fn get_continuations(&self, _: bool) -> anyhow::Result<Vec<Vec<u8>>> {
        self.get_vocab()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        if let Some(id) = self.special_vocab.token_to_id(token) {
            Some(id)
        } else {
            let bytes = token.as_bytes();
            if bytes.len() == 1 {
                Some(bytes[0] as u32)
            } else {
                let merge_id = self.state.0.get(bytes)?;
                Some(256 + *merge_id)
            }
        }
    }

    fn id_to_token(&self, id: u32) -> Option<Vec<u8>> {
        if id < 256 {
            Some(vec![id as u8])
        } else if id < 256 + u32::try_from(self.state.1.len()).ok()? {
            Some(self.state.1[usize::try_from(id - 256).ok()?].clone())
        } else {
            self.special_vocab.id_to_token(&id).map(|s| s.to_bytes())
        }
    }

    fn tokenize(&self, s: &str, ignore_special_tokens: bool) -> anyhow::Result<Tokenization> {
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
        Ok(Tokenization::new(
            self.add_prefix_and_suffix(token_ids),
            TokenizationInfo::Empty,
        ))
    }

    fn de_tokenize(
        &self,
        token_ids: &[u32],
        ignore_special_tokens: bool,
    ) -> anyhow::Result<String> {
        let mut bytes = Vec::new();
        let num_merge_ops = self.state.1.len() as u32;
        for token_id in token_ids {
            if *token_id < num_merge_ops {
                bytes.extend(&self.state.1[*token_id as usize]);
            } else if !ignore_special_tokens {
                bytes.extend(
                    self.special_vocab
                        .id_to_token(token_id)
                        .ok_or_else(|| anyhow!("unknown special token id {token_id}"))?
                        .as_bytes(),
                );
            }
        }
        Ok(String::from_utf8(bytes)?)
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
fn replace_pair_in_word(word: &[Vec<u8>], pair: &BytePair) -> Vec<Vec<u8>> {
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
                .find_position(|&subword| subword == &pair.first)
            else {
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
                let stat = stats.get_mut(&prev_pair).unwrap();
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
                let stat = stats.get_mut(&next_pair).unwrap();
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
                .find_position(|&subword| subword == &merged)
            else {
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
    files: Vec<PyBackedStr>,
    vocab_size: usize,
    num_special_tokens: usize,
    out_file: &str,
    max_lines_per_file: Option<usize>,
    normalization: Option<Normalization>,
    num_threads: u8,
    progress: bool,
) -> anyhow::Result<()> {
    let files: Vec<&str> = files.iter().map(|s| s.as_ref()).collect();
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
                    let Some(mut line) = line_iter_clone
                        .lock()
                        .expect("failed to lock line iter")
                        .next()
                    else {
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
    pub fn new(config: ByteTokenizerConfig, special_config: SpecialConfig) -> Self {
        Self::new_with(config, special_config)
    }

    fn new_with(config: ByteTokenizerConfig, mut special_config: SpecialConfig) -> Self {
        if let Some(pad_to) = config.pad_to_multiple_of {
            assert!(
                pad_to.is_power_of_two(),
                "pad_to_multiple_of must be a power of two"
            );
            let special_tokens: HashSet<&String> = special_config.tokens.iter().collect();
            let num_special_tokens = special_tokens.len();
            let num_tokens = 256 + num_special_tokens;
            let num_pad_tokens =
                (num_tokens as f64 / pad_to as f64).ceil() as usize * pad_to - num_tokens;
            special_config
                .tokens
                .extend((0..num_pad_tokens).map(|idx| format!("<extra_token_{}>", idx)));
        }
        Self::new_vocab_free_tokenizer(256, special_config, config)
    }

    fn process_input(&self, s: &str, ignore_special_tokens: bool) -> (Vec<u32>, TokenizationInfo) {
        let mut tokens = vec![];
        let group_name = match self.config.groups {
            ByteGroups::Bytes => "byte_groups",
            ByteGroups::CodePoints => "code_point_groups",
        }
        .to_string();

        // initialize groups with 1 for each prefix token
        let mut groups = vec![TokenGroup::Full(1); self.num_prefix_tokens()];

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
        groups.append(&mut vec![TokenGroup::Full(1); self.num_suffix_tokens()]);
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

    fn get_vocab(&self) -> anyhow::Result<Vec<Vec<u8>>> {
        let mut vocab: BTreeMap<_, _> = (0..=u8::MAX).map(|b| (u32::from(b), vec![b])).collect();
        for (special, &id) in &self.special_vocab.vocab {
            vocab.insert(id, special.to_bytes());
        }
        Ok(vocab.into_values().collect())
    }

    fn get_continuations(&self, _: bool) -> anyhow::Result<Vec<Vec<u8>>> {
        self.get_vocab()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match token.as_bytes() {
            [b] => Some(u32::from(*b)),
            _ => self.special_vocab.token_to_id(token),
        }
    }

    fn id_to_token(&self, id: u32) -> Option<Vec<u8>> {
        if id < 256 {
            Some(vec![id as u8])
        } else {
            self.special_vocab
                .id_to_token(&id)
                .map(|s| s.as_bytes().to_vec())
        }
    }

    fn tokenize(&self, s: &str, ignore_special_tokens: bool) -> anyhow::Result<Tokenization> {
        let (bytes, info) = self.process_input(s, ignore_special_tokens);
        Ok(Tokenization::new(self.add_prefix_and_suffix(bytes), info))
    }

    fn de_tokenize(
        &self,
        token_ids: &[u32],
        ignore_special_tokens: bool,
    ) -> anyhow::Result<String> {
        let mut bytes = vec![];
        for token_id in token_ids {
            if *token_id < 256 {
                bytes.push(u8::try_from(*token_id)?);
            } else if !ignore_special_tokens {
                bytes.extend(
                    self.special_vocab
                        .id_to_token(token_id)
                        .ok_or_else(|| anyhow!("unknown special token id {token_id}"))?
                        .as_bytes(),
                );
            }
        }
        Ok(String::from_utf8(bytes)?)
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

    fn pad_token_id(&self) -> u32 {
        self.inner.pad_token_id()
    }
}

impl Tokenize for ByT5Tokenizer {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn get_vocab(&self) -> anyhow::Result<Vec<Vec<u8>>> {
        self.inner.get_vocab()
    }

    fn get_continuations(&self, initial: bool) -> anyhow::Result<Vec<Vec<u8>>> {
        self.inner.get_continuations(initial)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<Vec<u8>> {
        self.inner.id_to_token(id)
    }

    fn tokenize(&self, s: &str, ignore_special_tokens: bool) -> anyhow::Result<Tokenization> {
        let Tokenization { token_ids, info } = self.inner.tokenize(s, ignore_special_tokens)?;
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

    fn de_tokenize(
        &self,
        token_ids: &[u32],
        ignore_special_tokens: bool,
    ) -> anyhow::Result<String> {
        self.inner.de_tokenize(token_ids, ignore_special_tokens)
    }
}

pub fn tokenizer(cfg: TokenizerConfig) -> anyhow::Result<Tokenizer> {
    Ok(match cfg.tokenize {
        TokenizeConfig::Character(char_cfg) => Box::new(CharTokenizer::new(char_cfg, cfg.special)),
        TokenizeConfig::Byte(byte_cfg) => Box::new(ByteTokenizer::new(byte_cfg, cfg.special)),
        TokenizeConfig::ByT5(byte_cfg) => Box::new(ByT5Tokenizer::new(byte_cfg)),
        TokenizeConfig::BPE(bpe_cfg) => Box::new(BPETokenizer::new(bpe_cfg, cfg.special)?),
        TokenizeConfig::Dummy(d) => Box::new(DummyTokenizer::new(d)),
        TokenizeConfig::Huggingface(path) => {
            Box::new(HuggingfaceTokenizer::new(path, cfg.special)?)
        }
    })
}

#[pyclass]
#[pyo3(name = "Tokenizer")]
struct PyTokenizer {
    tokenizer: Tokenizer,
    constraint: Option<TokenizationConstraint>,
}

#[pymethods]
impl PyTokenizer {
    #[staticmethod]
    #[pyo3(signature = (config, constraint = None))]
    fn from_config(
        config: TokenizerConfig,
        constraint: Option<TokenizationConstraintConfig>,
    ) -> anyhow::Result<Self> {
        Ok(PyTokenizer {
            tokenizer: tokenizer(config)?,
            constraint: constraint
                .map(TokenizationConstraint::from_config)
                .transpose()?,
        })
    }

    #[pyo3(signature = (s, ignore_special_tokens = false))]
    fn tokenize(&self, s: &str, ignore_special_tokens: bool) -> anyhow::Result<Tokenization> {
        if let Some(constraint) = &self.constraint {
            self.tokenizer
                .tokenize_with_constraint(s, ignore_special_tokens, constraint)
        } else {
            self.tokenizer.tokenize(s, ignore_special_tokens)
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<Vec<u8>> {
        self.tokenizer.id_to_token(id)
    }

    fn normalize(&self, s: &str) -> String {
        self.tokenizer.normalize(s)
    }

    #[pyo3(signature = (token_ids, ignore_special_tokens = true))]
    fn de_tokenize(
        &self,
        token_ids: Vec<u32>,
        ignore_special_tokens: bool,
    ) -> anyhow::Result<String> {
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

    fn get_vocab(&self) -> anyhow::Result<Vec<Vec<u8>>> {
        self.tokenizer.get_vocab()
    }

    fn get_continuations(&self, initial: bool) -> anyhow::Result<Vec<Vec<u8>>> {
        self.tokenizer.get_continuations(initial)
    }
}

/// A submodule containing functionality to tokenize text into tokens.
/// Currently supported tokenization schemes are:
/// - character level tokenization
/// - byte level tokenization
pub(super) fn add_submodule(py: Python<'_>, parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(py, "tokenization")?;
    m.add_class::<PyTokenizer>()?;
    m.add_class::<Tokenization>()?;
    m.add_class::<SpecialTokens>()?;
    m.add_class::<LanguageTokens>()?;
    m.add_function(wrap_pyfunction!(train_bpe_py, m.clone())?)?;
    parent_module.add_submodule(&m)?;

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
        );
        let text = "a täst";
        let Tokenization { token_ids, .. } = tok.tokenize(text, true).unwrap();
        assert_eq!(token_ids.len(), 6);
        assert_eq!(token_ids[3], tok.unk_token_id());
        assert_eq!(
            tok.de_tokenize(&token_ids, true).unwrap(),
            "a tst".to_string()
        );
        assert_eq!(
            tok.de_tokenize(&token_ids, false).unwrap(),
            "a t<unk>st".to_string()
        );
        let text = "a <pad>täst";
        let Tokenization { token_ids, .. } = tok.tokenize(text, false).unwrap();
        assert_eq!(token_ids.len(), 7);
        assert_eq!(token_ids[2], tok.pad_token_id);
        assert_eq!(token_ids[4], tok.unk_token_id());
        assert_eq!(
            tok.de_tokenize(&token_ids, true).unwrap(),
            "a tst".to_string()
        );
        assert_eq!(
            tok.de_tokenize(&token_ids, false).unwrap(),
            "a <pad>t<unk>st".to_string()
        );
        let text = "a <pad>täst";
        let Tokenization { token_ids, .. } = tok.tokenize(text, true).unwrap();
        assert_eq!(token_ids.len(), 11);
        assert_eq!(
            tok.de_tokenize(&token_ids, true).unwrap(),
            "a <pad>tst".to_string()
        );
        assert_eq!(
            tok.de_tokenize(&token_ids, false).unwrap(),
            "a <pad>t<unk>st".to_string()
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
        let tok = ByteTokenizer::new(tokenize_cfg.clone(), SpecialConfig::default());
        assert_eq!(tok.vocab_size(), 384);
        let text = "a täst";
        let Tokenization { token_ids, info } = tok.tokenize(text, true).unwrap();
        assert_eq!(
            token_ids.iter().map(|tok| *tok as u8).collect::<Vec<u8>>(),
            text.as_bytes()
        );
        match info {
            TokenizationInfo::TokenGroups(groups) => {
                assert_eq!(
                    groups,
                    HashMap::from([(
                        "byte_groups".to_string(),
                        (
                            [1, 1, 1, 2, 1, 1]
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
        assert_eq!(token_ids.len(), 7);
        assert_eq!(tok.de_tokenize(&token_ids, true).unwrap(), text.to_string());
        let tokenize_cfg = ByteTokenizerConfig {
            groups: ByteGroups::CodePoints,
            ..tokenize_cfg
        };
        let tok = ByteTokenizer::new(tokenize_cfg, SpecialConfig::default());
        let text = "a täst";
        let Tokenization { token_ids, info } = tok.tokenize(text, true).unwrap();
        assert_eq!(
            token_ids.iter().map(|tok| *tok as u8).collect::<Vec<u8>>(),
            text.as_bytes()
        );
        match info {
            TokenizationInfo::TokenGroups(groups) => {
                assert_eq!(
                    groups,
                    HashMap::from([(
                        "code_point_groups".to_string(),
                        (
                            [[1].as_slice(), &[1], &[1], &[2], &[1], &[1]]
                                .into_iter()
                                .map(|vs| {
                                    TokenGroup::Nested(
                                        vs.into_iter().map(|v| TokenGroup::Full(*v)).collect(),
                                    )
                                })
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
        let bpe = BPETokenizer::new(bpe_config, special_config).unwrap();
        info!("loaded bpe tokenizer");
        let s = "this is a long reading couple restaurant";
        let token_ids = bpe.tokenize(s, true).unwrap().token_ids;
        info!("token ids: {token_ids:?}");
        let tokens: Vec<_> = token_ids
            .iter()
            .filter_map(|t| bpe.state.1.get(*t as usize))
            .map(|b| String::from_utf8_lossy(b).to_string())
            .collect();
        info!("tokens: {tokens:?}");
        let ds = bpe.de_tokenize(&token_ids, true).unwrap();
        assert_eq!(s, ds);
        info!("de-tokenized: \"{ds}\"");

        let mut rng = ChaCha8Rng::seed_from_u64(22);
        for _ in 0..200 {
            let s: String = (&mut rng)
                .sample_iter::<char, _>(rand::distributions::Standard)
                .take(256)
                .collect();

            let token_ids = bpe.tokenize(&s, true).unwrap().token_ids;
            let ds = bpe.de_tokenize(&token_ids, true).unwrap();
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
        let Tokenization { token_ids, info: _ } = tok.tokenize("a täst", true).unwrap();
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
