use crate::unicode::CS;
use crate::utils::{py_invalid_type_error, py_required_key_error, run_length_decode};
use anyhow::anyhow;
use itertools::Itertools;
use numpy::ndarray::{Array1, Array2};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::HashMap;
use std::thread::sleep;
use std::time::Duration;

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
    const UNK: &str = UNK;
    #[classattr]
    const BOS: &str = BOS;
    #[classattr]
    const EOS: &str = EOS;
    #[classattr]
    const PAD: &str = PAD;
}

// language tokens
pub const LANG_UNK: &str = "[unk]";

#[pyclass]
pub struct LanguageTokens {}

#[pymethods]
impl LanguageTokens {
    #[classattr]
    const UNK: &str = LANG_UNK;
}

/// This is a tokenizer config, containing language options
/// and the actual tokenize config inside it.
#[derive(Clone, Debug)]
pub struct TokenizerConfig {
    tokenize: TokenizeConfig,
    prefix: Vec<String>,
    suffix: Vec<String>,
    language: Option<LanguageConfig>,
}

impl TokenizerConfig {
    pub fn new(
        tokenize: TokenizeConfig,
        prefix: Vec<String>,
        suffix: Vec<String>,
        language: Option<LanguageConfig>,
    ) -> Self {
        Self {
            tokenize,
            prefix,
            suffix,
            language,
        }
    }
}

impl<'a> FromPyObject<'a> for TokenizerConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        Ok(Self {
            tokenize: d
                .get_item("tokenize")
                .ok_or_else(|| py_required_key_error("tokenize", "tokenizer config"))?
                .extract()?,
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
            languages: if let Some(value) = d.get_item("languages") {
                value.extract()?
            } else {
                vec![]
            },
            default_language: if let Some(value) = d.get_item("default_language") {
                value.extract()?
            } else {
                LANG_UNK.to_string()
            },
        })
    }
}

/// This enum defines all tokenizers that are supported by this crate.
#[derive(Clone, Debug)]
pub enum TokenizeConfig {
    Character(bool),
    Byte(bool, ByteGroups, GroupAggregation),
    ByT5(bool, ByteGroups, GroupAggregation),
    Dummy(Duration),
}

impl IntoPy<PyObject> for TokenizeConfig {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let d: &PyDict = PyDict::new(py);
        let tokenizer_type = match self {
            TokenizeConfig::Character(use_g) => {
                d.set_item("use_graphemes", use_g).unwrap();
                "character"
            }
            TokenizeConfig::Byte(use_g, groups, agg) => {
                d.set_item("use_graphemes", use_g).unwrap();
                d.set_item("groups", groups.into_py(py)).unwrap();
                d.set_item("aggregation", agg.into_py(py)).unwrap();
                "byte"
            }
            TokenizeConfig::ByT5(use_g, groups, agg) => {
                d.set_item("use_graphemes", use_g).unwrap();
                d.set_item("groups", groups.into_py(py)).unwrap();
                d.set_item("aggregation", agg.into_py(py)).unwrap();
                "byt5"
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
                TokenizeConfig::Character(use_graphemes)
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
                if name == "byt5" {
                    TokenizeConfig::ByT5(use_graphemes, groups.extract()?, agg)
                } else {
                    TokenizeConfig::Byte(use_graphemes, groups.extract()?, agg)
                }
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

pub type Grouping = (Vec<Vec<usize>>, GroupAggregation);
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
}

pub enum TensorizedTokenizationInfo {
    Empty,
    TokenGroups(HashMap<String, (SparseCoo, PaddingMask)>),
}

pub struct SparseCoo {
    indices: Array2<i32>,
    values: Array1<f32>,
    size: Vec<usize>,
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
        match self {
            TensorizedTokenizationInfo::Empty => PyDict::new(py),
            TensorizedTokenizationInfo::TokenGroups(matrices) => {
                let d = PyDict::new(py);
                for (name, (scoo, pad_mask)) in matrices {
                    let t = PyTuple::new(py, &[scoo.into_py(py), pad_mask.into_py(py)]);
                    d.set_item(name, t).unwrap();
                }
                d
            }
        }
        .into_py(py)
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

type Values = Vec<Vec<f32>>;
type Indices = Vec<Vec<i32>>;
#[inline]
fn expand_grouping(s_idx: usize, groups: &Vec<Vec<usize>>, pow: i32) -> (Values, Indices, Indices) {
    let num_groups = groups[s_idx].len();
    if s_idx > 0 {
        let mut new_weights = vec![vec![]; num_groups];
        let mut new_group_indices = vec![];
        let mut new_seq_indices = vec![vec![]; num_groups];
        let (prev_weights, _, mut prev_seq_indices) = expand_grouping(s_idx - 1, groups, pow);
        assert_eq!(prev_weights.len(), groups[s_idx].iter().sum::<usize>());
        let mut cum_g = 0;
        for (i, &g) in groups[s_idx].iter().enumerate() {
            let fac = (g as f32).powi(pow);
            for j in cum_g..cum_g + g {
                new_weights[i].extend(prev_weights[j].iter().map(|w| w * fac));
                new_seq_indices[i].append(&mut prev_seq_indices[j]);
            }
            new_group_indices.push(vec![i as i32; new_weights[i].len()]);
            cum_g += g;
        }
        (new_weights, new_group_indices, new_seq_indices)
    } else {
        let mut weights = vec![];
        let mut group_indices = vec![];
        let mut seq_indices = vec![];
        let mut cum_g = 0;
        for (i, &g) in groups[s_idx].iter().enumerate() {
            let fac = (g as f32).powi(pow);
            weights.push(vec![fac; g]);
            group_indices.push(vec![i as i32; g]);
            let cum_g_i = cum_g as i32;
            seq_indices.push((cum_g_i..cum_g_i + g as i32).collect());
            cum_g += g;
        }
        (weights, group_indices, seq_indices)
    }
}

#[inline]
fn group_values(grouping: &Grouping) -> (Vec<f32>, Vec<i32>, Vec<i32>, usize) {
    let (groups, agg) = grouping;
    assert!(!groups.is_empty());
    let pow = match agg {
        GroupAggregation::Mean => -1,
        GroupAggregation::Sum => 0,
    };
    let s_idx = groups.len() - 1;
    let num_groups = groups[s_idx].len();
    let (values, group_indices, seq_indices) = expand_grouping(s_idx, groups, pow);
    (
        values.into_iter().flatten().collect(),
        group_indices.into_iter().flatten().collect(),
        seq_indices.into_iter().flatten().collect(),
        num_groups,
    )
}

pub fn token_groups_to_sparse_coo_matrix(
    groupings: &[&Grouping],
    lengths: &[usize],
) -> anyhow::Result<SparseCoo> {
    let mut indices = vec![vec![]; 3];
    let mut values = vec![];
    let mut group_lengths = vec![];

    for (i, grouping) in groupings.iter().enumerate() {
        let (mut v, mut g, mut s, l) = group_values(grouping);
        values.append(&mut v);
        indices[0].append(&mut vec![i as i32; g.len()]);
        indices[1].append(&mut g);
        indices[2].append(&mut s);
        group_lengths.push(l);
    }

    let max_group_length = group_lengths.iter().max().copied().unwrap_or(0);
    let max_length = lengths.iter().max().copied().unwrap_or(0);
    let size = vec![groupings.len(), max_group_length, max_length];
    Ok(SparseCoo {
        indices: Array2::from_shape_vec(
            (3, indices[0].len()),
            indices.into_iter().flatten().collect(),
        )?,
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
                for (group_name, (stages, agg)) in token_groups.iter() {
                    let l = PyList::empty(py);
                    for groups in stages.iter() {
                        l.append(groups).unwrap();
                    }
                    let gd = PyDict::new(py);
                    gd.set_item("groups", l).unwrap();
                    gd.set_item("aggregation", agg.into_py(py)).unwrap();
                    d.set_item(group_name, gd).unwrap();
                }
                "token_groups"
            }
        };
        d.set_item("type", info_type).unwrap();
        d.into()
    }
}

impl<'a> FromPyObject<'a> for TokenizationInfo {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let Some(info_type) = d.get_item("type") else {
            return Err(py_required_key_error("type", "tokenization info"));
        };
        let info_type: String = info_type.extract()?;
        let info = match info_type.as_str() {
            "empty" => TokenizationInfo::Empty,
            "token_groups" => {
                let mut token_groups = HashMap::new();
                for key in d.keys() {
                    let key_s: String = key.extract()?;
                    if key_s == "type" {
                        continue;
                    }
                    let gd = d.get_item(key).unwrap();
                    let groups = gd.get_item("groups")?.extract()?;
                    let agg = gd.get_item("aggregation")?.extract()?;
                    token_groups.insert(key_s, (groups, agg));
                }
                TokenizationInfo::TokenGroups(token_groups)
            }
            k => return Err(py_invalid_type_error(k, "tokenization info")),
        };
        Ok(info)
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
pub trait Tokenize: Send + Sync + 'static {
    fn vocab_size(&self) -> usize;

    fn unk_token_id(&self) -> u32;

    fn bos_token_id(&self) -> u32;

    fn eos_token_id(&self) -> u32;

    fn pad_token_id(&self) -> u32;

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

    fn prefix_token_ids(&self) -> &[u32];

    fn suffix_token_ids(&self) -> &[u32];

    fn language_config(&self) -> Option<&LanguageConfig>;

    fn add_prefix_and_suffix(
        &self,
        mut token_ids: Vec<u32>,
        lang: Option<&str>,
    ) -> anyhow::Result<Vec<u32>> {
        let mut prefix = self.prefix_token_ids().to_vec();
        let mut suffix = self.suffix_token_ids().to_vec();
        if let Some(lang_cfg) = self.language_config() {
            let lang = lang.unwrap_or(&lang_cfg.default_language);
            let lang_id = self.special_token_to_id(lang);
            if lang_id == self.unk_token_id() {
                return Err(anyhow!(
                    "language {} is not supported by this tokenizer",
                    lang
                ));
            }
            if lang_cfg.add_language_token_to_prefix {
                prefix.push(lang_id);
            }
            if lang_cfg.add_language_token_to_suffix {
                suffix.push(lang_id);
            }
        }
        prefix.reserve_exact(token_ids.len() + suffix.len());
        prefix.append(&mut token_ids);
        prefix.append(&mut suffix);
        Ok(prefix)
    }

    fn add_special_token(&mut self, special_token: &str);

    fn add_special_tokens(&mut self, special_tokens: &[&str]) {
        for special_token in special_tokens {
            self.add_special_token(special_token);
        }
    }

    fn tokenize(&self, s: &str, lang: Option<&str>) -> anyhow::Result<Tokenization>;

    fn de_tokenize(&self, token_ids: &[u32]) -> String;

    fn special_token_to_id(&self, token: &str) -> u32;

    fn id_to_special_token(&self, token_id: &u32) -> &str;
}

/// Dummy tokenizer that just waits a specified time in its tokenize function.
/// Used for testing only.
#[derive(Clone, Debug)]
struct DummyTokenizer {
    delay: Duration,
    dummy_token: String,
}

impl DummyTokenizer {
    fn new(delay: Duration) -> Self {
        DummyTokenizer {
            delay,
            dummy_token: "".to_string(),
        }
    }
}

impl Tokenize for DummyTokenizer {
    fn vocab_size(&self) -> usize {
        0
    }

    fn unk_token_id(&self) -> u32 {
        0
    }

    fn bos_token_id(&self) -> u32 {
        0
    }

    fn eos_token_id(&self) -> u32 {
        0
    }

    fn pad_token_id(&self) -> u32 {
        0
    }

    fn num_prefix_tokens(&self) -> usize {
        0
    }

    fn num_suffix_tokens(&self) -> usize {
        0
    }

    fn prefix_token_ids(&self) -> &[u32] {
        &[]
    }

    fn suffix_token_ids(&self) -> &[u32] {
        &[]
    }

    fn language_config(&self) -> Option<&LanguageConfig> {
        None
    }

    fn add_special_token(&mut self, _: &str) {}

    fn tokenize(&self, _: &str, _: Option<&str>) -> anyhow::Result<Tokenization> {
        sleep(self.delay);
        Ok(Tokenization::new(vec![], TokenizationInfo::Empty))
    }

    fn de_tokenize(&self, _: &[u32]) -> String {
        "".to_string()
    }

    fn special_token_to_id(&self, _: &str) -> u32 {
        0
    }

    fn id_to_special_token(&self, _: &u32) -> &str {
        &self.dummy_token
    }
}

/// A tokenizer based on the ascii characters, digits, and punctuations marks.
/// Can e.g. be used to efficiently (meaning small vocab size) represent most
/// English texts.
#[derive(Clone, Debug)]
pub struct CharTokenizer {
    _prefix_token_ids: Vec<u32>,
    _suffix_token_ids: Vec<u32>,
    language_config: Option<LanguageConfig>,
    vocab: HashMap<char, u32>,
    reverse_vocab: HashMap<u32, char>,
    special_vocab: HashMap<String, u32>,
    reverse_special_vocab: HashMap<u32, String>,
    unk_token: String,
    use_graphemes: bool,
}

const CHARS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\"\"!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\"\" ";

impl CharTokenizer {
    pub fn new(
        use_graphemes: bool,
        prefix_tokens: &[&str],
        suffix_tokens: &[&str],
        language_config: Option<LanguageConfig>,
    ) -> Self {
        let vocab: HashMap<char, u32> = CHARS
            .chars()
            .unique()
            .enumerate()
            .map(|(tok_id, c)| (c, tok_id as u32))
            .collect();
        let reverse_vocab = vocab
            .iter()
            .map(|(token, token_id)| (*token_id, *token))
            .collect();
        let special_vocab: HashMap<String, u32> = SPECIAL_TOKENS
            .into_iter()
            .map(str::to_string)
            .zip(vocab.len() as u32..(vocab.len() + SPECIAL_TOKENS.len()) as u32)
            .collect();
        let reverse_special_vocab = special_vocab
            .iter()
            .map(|(st, tok_id)| (*tok_id, st.clone()))
            .collect();
        let unk_token = UNK.to_string();
        let mut tokenizer = CharTokenizer {
            _prefix_token_ids: vec![],
            _suffix_token_ids: vec![],
            language_config,
            vocab,
            reverse_vocab,
            special_vocab,
            reverse_special_vocab,
            unk_token,
            use_graphemes,
        };
        tokenizer.add_special_tokens(prefix_tokens);
        tokenizer.add_special_tokens(suffix_tokens);
        let mut token_ids = prefix_tokens
            .iter()
            .map(|s| tokenizer.special_token_to_id(s))
            .collect();
        tokenizer._prefix_token_ids.append(&mut token_ids);
        let mut token_ids = suffix_tokens
            .iter()
            .map(|s| tokenizer.special_token_to_id(s))
            .collect();
        tokenizer._suffix_token_ids.append(&mut token_ids);
        let languages = if let Some(lang_cfg) = tokenizer.language_config.as_ref() {
            let mut l = vec![lang_cfg.default_language.clone()];
            l.extend(lang_cfg.languages.iter().cloned());
            l
        } else {
            vec![]
        };
        for lang in languages.iter() {
            tokenizer.add_special_token(lang);
        }
        tokenizer
    }
}

impl Default for CharTokenizer {
    fn default() -> Self {
        Self::new(true, &DEFAULT_PREFIX_TOKENS, &DEFAULT_SUFFIX_TOKENS, None)
    }
}

impl Tokenize for CharTokenizer {
    fn vocab_size(&self) -> usize {
        self.vocab.len() + self.special_vocab.len()
    }

    fn unk_token_id(&self) -> u32 {
        *self.special_vocab.get(UNK).unwrap()
    }

    fn bos_token_id(&self) -> u32 {
        *self.special_vocab.get(BOS).unwrap()
    }

    fn eos_token_id(&self) -> u32 {
        *self.special_vocab.get(EOS).unwrap()
    }

    fn pad_token_id(&self) -> u32 {
        *self.special_vocab.get(PAD).unwrap()
    }

    fn prefix_token_ids(&self) -> &[u32] {
        &self._prefix_token_ids
    }

    fn suffix_token_ids(&self) -> &[u32] {
        &self._suffix_token_ids
    }

    fn language_config(&self) -> Option<&LanguageConfig> {
        self.language_config.as_ref()
    }

    fn add_special_token(&mut self, special_token: &str) {
        let token_id = self.vocab_size() as u32;
        if self.reverse_special_vocab.contains_key(&token_id) {
            panic!("cannot add any more tokens to the character tokenizer");
        } else if self.special_vocab.contains_key(special_token) {
            return;
        }
        self.special_vocab
            .insert(special_token.to_string(), token_id);
        self.reverse_special_vocab
            .insert(token_id, special_token.to_string());
    }

    fn tokenize(&self, s: &str, lang: Option<&str>) -> anyhow::Result<Tokenization> {
        let token_ids = CS::new(s, self.use_graphemes)
            .chars()
            .map(|c| {
                // Character always has at least one char so this is safe
                let mut c_iter = c.code_points();
                let char = c_iter.next().unwrap();
                // return unk if Character has another char because
                // our tokens in the vocab are all single char tokens
                if c_iter.next().is_some() {
                    self.unk_token_id()
                } else {
                    self.vocab
                        .get(&char)
                        .copied()
                        .unwrap_or_else(|| self.unk_token_id())
                }
            })
            .collect();
        Ok(Tokenization::new(
            self.add_prefix_and_suffix(token_ids, lang)?,
            TokenizationInfo::Empty,
        ))
    }

    fn de_tokenize(&self, token_ids: &[u32]) -> String {
        token_ids
            .iter()
            .filter_map(|i| self.reverse_vocab.get(i))
            .join("")
    }

    fn special_token_to_id(&self, token: &str) -> u32 {
        self.special_vocab
            .get(token)
            .copied()
            .unwrap_or_else(|| self.unk_token_id())
    }

    fn id_to_special_token(&self, token_id: &u32) -> &str {
        if let Some(token) = self.reverse_special_vocab.get(token_id) {
            token
        } else {
            &self.unk_token
        }
    }
}

#[derive(Clone, Debug)]
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
pub struct ByteTokenizer {
    _prefix_token_ids: Vec<u32>,
    _suffix_token_ids: Vec<u32>,
    language_config: Option<LanguageConfig>,
    special_vocab: HashMap<String, u32>,
    reverse_special_vocab: HashMap<u32, String>,
    unk_token: String,
    use_graphemes: bool,
    groups: ByteGroups,
    aggregation: GroupAggregation,
}

impl Default for ByteTokenizer {
    fn default() -> Self {
        ByteTokenizer::new(
            true,
            ByteGroups::CodePoints,
            GroupAggregation::Mean,
            &[],
            &[],
            None,
        )
    }
}

impl ByteTokenizer {
    pub fn new(
        use_graphemes: bool,
        groups: ByteGroups,
        aggregation: GroupAggregation,
        prefix_tokens: &[&str],
        suffix_tokens: &[&str],
        language_config: Option<LanguageConfig>,
    ) -> Self {
        Self::new_inner(
            use_graphemes,
            groups,
            aggregation,
            &SPECIAL_TOKENS,
            UNK,
            prefix_tokens,
            suffix_tokens,
            language_config,
        )
    }

    pub(self) fn new_byt5(
        use_graphemes: bool,
        groups: ByteGroups,
        aggregation: GroupAggregation,
    ) -> Self {
        // byt5 has 3 special tokens: <pad>, </s>, <unk>
        Self::new_inner(
            use_graphemes,
            groups,
            aggregation,
            &["<pad>", "</s>", "<unk>"],
            "<unk>",
            &[],
            &["</s>"],
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_inner(
        use_graphemes: bool,
        groups: ByteGroups,
        aggregation: GroupAggregation,
        special_tokens: &[&str],
        unk_token: &str,
        prefix_tokens: &[&str],
        suffix_tokens: &[&str],
        language_config: Option<LanguageConfig>,
    ) -> Self {
        let special_vocab: HashMap<String, u32> = special_tokens
            .iter()
            .zip(u8::MAX as u32..u8::MAX as u32 + special_tokens.len() as u32)
            .map(|(&st, tok_id)| (st.into(), tok_id))
            .collect();
        let reverse_special_vocab = special_vocab
            .iter()
            .map(|(token, token_id)| (*token_id, token.clone()))
            .collect();
        let unk_token = unk_token.to_string();
        let mut tokenizer = ByteTokenizer {
            _prefix_token_ids: vec![],
            _suffix_token_ids: vec![],
            language_config,
            special_vocab,
            reverse_special_vocab,
            unk_token,
            use_graphemes,
            groups,
            aggregation,
        };
        tokenizer.add_special_tokens(prefix_tokens);
        tokenizer.add_special_tokens(suffix_tokens);
        let mut token_ids = prefix_tokens
            .iter()
            .map(|s| tokenizer.special_token_to_id(s))
            .collect();
        tokenizer._prefix_token_ids.append(&mut token_ids);
        let mut token_ids = suffix_tokens
            .iter()
            .map(|s| tokenizer.special_token_to_id(s))
            .collect();
        tokenizer._suffix_token_ids.append(&mut token_ids);
        let languages = if let Some(lang_cfg) = tokenizer.language_config.as_ref() {
            let mut l = vec![lang_cfg.default_language.clone()];
            l.extend(lang_cfg.languages.iter().cloned());
            l
        } else {
            vec![]
        };
        for lang in languages.iter() {
            tokenizer.add_special_token(lang);
        }
        tokenizer
    }

    fn split(&self, s: &str) -> (Vec<u32>, HashMap<String, Grouping>) {
        let tokens = s.as_bytes().iter().map(|b| *b as u32).collect();
        let groups = match self.groups {
            ByteGroups::Bytes => {
                let cs = CS::new(s, self.use_graphemes);
                let mut groups = vec![1; self.num_prefix_tokens()];
                groups.extend(run_length_decode(&cs.rle_cluster_lengths));
                groups.extend(vec![1; self.num_suffix_tokens()]);
                HashMap::from([("byte_groups".to_string(), (vec![groups], self.aggregation))])
            }
            ByteGroups::CodePoints => {
                let cs = CS::new(s, self.use_graphemes);
                let mut byte_groups = vec![1; self.num_prefix_tokens()];
                let mut code_point_groups = vec![1; self.num_prefix_tokens()];
                for char in cs.chars() {
                    let mut num_chars = 0;
                    for code_point in char.code_points() {
                        byte_groups.push(code_point.len_utf8());
                        num_chars += 1;
                    }
                    code_point_groups.push(num_chars);
                }
                byte_groups.extend(vec![1; self.num_suffix_tokens()]);
                code_point_groups.extend(vec![1; self.num_suffix_tokens()]);
                HashMap::from([(
                    "code_point_groups".to_string(),
                    (vec![byte_groups, code_point_groups], self.aggregation),
                )])
            }
        };
        (tokens, groups)
    }
}

impl Tokenize for ByteTokenizer {
    fn vocab_size(&self) -> usize {
        u8::MAX as usize + self.special_vocab.len()
    }

    fn unk_token_id(&self) -> u32 {
        *self.special_vocab.get(UNK).unwrap()
    }

    fn bos_token_id(&self) -> u32 {
        *self.special_vocab.get(BOS).unwrap()
    }

    fn eos_token_id(&self) -> u32 {
        *self.special_vocab.get(EOS).unwrap()
    }

    fn pad_token_id(&self) -> u32 {
        *self.special_vocab.get(PAD).unwrap()
    }

    fn prefix_token_ids(&self) -> &[u32] {
        &self._prefix_token_ids
    }

    fn suffix_token_ids(&self) -> &[u32] {
        &self._suffix_token_ids
    }

    fn language_config(&self) -> Option<&LanguageConfig> {
        self.language_config.as_ref()
    }

    fn add_special_token(&mut self, special_token: &str) {
        let token_id = self.vocab_size() as u32;
        if self.reverse_special_vocab.contains_key(&token_id) {
            panic!("cannot add any more tokens to the character tokenizer");
        } else if self.special_vocab.contains_key(special_token) {
            return;
        }
        self.special_vocab
            .insert(special_token.to_string(), token_id);
        self.reverse_special_vocab
            .insert(token_id, special_token.to_string());
    }

    fn tokenize(&self, s: &str, lang: Option<&str>) -> anyhow::Result<Tokenization> {
        let (bytes, token_groups) = self.split(s);

        Ok(Tokenization::new(
            self.add_prefix_and_suffix(bytes, lang)?,
            TokenizationInfo::TokenGroups(token_groups),
        ))
    }

    fn de_tokenize(&self, token_ids: &[u32]) -> String {
        let bytes: Vec<u8> = token_ids
            .iter()
            .filter_map(|t| if *t < 256u32 { Some(*t as u8) } else { None })
            .collect();
        String::from_utf8_lossy(&bytes).to_string()
    }

    fn special_token_to_id(&self, token: &str) -> u32 {
        self.special_vocab
            .get(token)
            .copied()
            .unwrap_or_else(|| self.unk_token_id())
    }

    fn id_to_special_token(&self, token_id: &u32) -> &str {
        if let Some(token) = self.reverse_special_vocab.get(token_id) {
            token
        } else {
            &self.unk_token
        }
    }
}

pub struct ByT5Tokenizer {
    inner: ByteTokenizer,
}

impl ByT5Tokenizer {
    pub fn new(
        use_graphemes: bool,
        groups: ByteGroups,
        group_aggregation: GroupAggregation,
    ) -> Self {
        let inner = ByteTokenizer::new_byt5(use_graphemes, groups, group_aggregation);
        Self { inner }
    }
}

impl Tokenize for ByT5Tokenizer {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn unk_token_id(&self) -> u32 {
        2
    }

    fn bos_token_id(&self) -> u32 {
        panic!("ByT5 does not have a bos token")
    }

    fn eos_token_id(&self) -> u32 {
        1
    }

    fn pad_token_id(&self) -> u32 {
        0
    }

    fn prefix_token_ids(&self) -> &[u32] {
        &[]
    }

    fn suffix_token_ids(&self) -> &[u32] {
        &[1]
    }

    fn language_config(&self) -> Option<&LanguageConfig> {
        None
    }

    fn add_special_token(&mut self, _: &str) {
        panic!("ByT5 does not support adding special tokens")
    }

    fn tokenize(&self, s: &str, lang: Option<&str>) -> anyhow::Result<Tokenization> {
        let Tokenization { token_ids, info } = self.inner.tokenize(s, lang)?;
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

    fn de_tokenize(&self, token_ids: &[u32]) -> String {
        self.inner.de_tokenize(token_ids)
    }

    fn special_token_to_id(&self, token: &str) -> u32 {
        self.inner.special_token_to_id(token)
    }

    fn id_to_special_token(&self, token_id: &u32) -> &str {
        self.inner.id_to_special_token(token_id)
    }
}

pub fn tokenizer(cfg: TokenizerConfig) -> Tokenizer {
    let prefix: Vec<&str> = cfg.prefix.iter().map(String::as_str).collect();
    let suffix: Vec<&str> = cfg.suffix.iter().map(String::as_str).collect();
    match cfg.tokenize {
        TokenizeConfig::Character(use_g) => {
            Box::new(CharTokenizer::new(use_g, &prefix, &suffix, cfg.language))
        }
        TokenizeConfig::Byte(use_g, groups, agg) => Box::new(ByteTokenizer::new(
            use_g,
            groups,
            agg,
            &prefix,
            &suffix,
            cfg.language,
        )),
        TokenizeConfig::ByT5(use_g, groups, agg) => {
            Box::new(ByT5Tokenizer::new(use_g, groups, agg))
        }
        TokenizeConfig::Dummy(d) => Box::new(DummyTokenizer::new(d)),
    }
}

#[pyclass]
#[pyo3(name = "Tokenizer")]
struct PyTokenizer {
    tokenizer: Tokenizer,
    #[pyo3(get)]
    name: String,
}

#[pymethods]
impl PyTokenizer {
    #[staticmethod]
    fn from_config(config: TokenizerConfig) -> Self {
        PyTokenizer {
            name: match config.tokenize {
                TokenizeConfig::Character(_) => "character",
                TokenizeConfig::Byte(_, _, _) => "byte",
                TokenizeConfig::ByT5(_, _, _) => "byt5",
                TokenizeConfig::Dummy(_) => "dummy",
            }
            .to_string(),
            tokenizer: tokenizer(config),
        }
    }

    #[pyo3(signature = (s, lang = None))]
    fn tokenize(&self, s: &str, lang: Option<&str>) -> anyhow::Result<Tokenization> {
        self.tokenizer.tokenize(s, lang)
    }

    fn special_token_to_id(&self, token: &str) -> u32 {
        self.tokenizer.special_token_to_id(token)
    }

    fn id_to_special_token(&self, token_id: u32) -> &str {
        self.tokenizer.id_to_special_token(&token_id)
    }

    fn de_tokenize(&self, token_ids: Vec<u32>) -> String {
        self.tokenizer.de_tokenize(&token_ids)
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

    fn add_special_tokens(&mut self, tokens: Vec<&str>) {
        self.tokenizer.add_special_tokens(&tokens);
    }

    fn unk_token_id(&self) -> u32 {
        self.tokenizer.unk_token_id()
    }

    fn pad_token_id(&self) -> u32 {
        self.tokenizer.pad_token_id()
    }

    fn bos_token_id(&self) -> u32 {
        self.tokenizer.bos_token_id()
    }

    fn eos_token_id(&self) -> u32 {
        self.tokenizer.eos_token_id()
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
    parent_module.add_submodule(m)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use numpy::ndarray::{Array1, Array2};

    use crate::tokenization::{
        ByteGroups, ByteTokenizer, CharTokenizer, SparseCoo, Tokenization, TokenizationInfo,
        Tokenize, BOS, EOS,
    };

    use super::{token_groups_to_sparse_coo_matrix, ByT5Tokenizer, GroupAggregation};

    #[test]
    fn test_char_tokenizer() {
        let pfx = vec![BOS];
        let sfx = vec![EOS];
        let tok = CharTokenizer::new(true, &pfx, &sfx, None);
        let text = "a täst";
        let Tokenization { token_ids, .. } = tok.tokenize(text, None).unwrap();
        assert_eq!(token_ids.len(), 6 + 2);
        assert_eq!(token_ids[4], tok.unk_token_id());
        assert_eq!(tok.de_tokenize(&token_ids), "a tst".to_string());
    }

    #[test]
    fn test_byte_tokenizer() {
        let pfx = vec![BOS];
        let sfx = vec![EOS];
        let tok = ByteTokenizer::new(
            true,
            ByteGroups::Bytes,
            GroupAggregation::Mean,
            &pfx,
            &sfx,
            None,
        );
        let text = "a täst";
        let Tokenization { token_ids, info } = tok.tokenize(text, None).unwrap();
        assert_eq!(
            token_ids[1..token_ids.len() - 1]
                .iter()
                .map(|tok| *tok as u8)
                .collect::<Vec<u8>>(),
            text.as_bytes().clone()
        );
        match info {
            TokenizationInfo::Empty => panic!("wrong info"),
            TokenizationInfo::TokenGroups(groups) => {
                assert_eq!(
                    groups,
                    HashMap::from([(
                        "byte_groups".to_string(),
                        (vec![vec![1, 1, 1, 1, 2, 1, 1, 1]], GroupAggregation::Mean)
                    )])
                )
            }
        };
        assert_eq!(token_ids.len(), 7 + 2);
        assert_eq!(tok.de_tokenize(&token_ids), text.to_string());
        let tok = ByteTokenizer::new(
            true,
            ByteGroups::CodePoints,
            GroupAggregation::Mean,
            &pfx,
            &sfx,
            None,
        );
        let text = "a täst";
        let Tokenization { token_ids, info } = tok.tokenize(text, None).unwrap();
        assert_eq!(
            token_ids[1..token_ids.len() - 1]
                .iter()
                .map(|tok| *tok as u8)
                .collect::<Vec<u8>>(),
            text.as_bytes().clone()
        );
        match info {
            TokenizationInfo::Empty => panic!("wrong info"),
            TokenizationInfo::TokenGroups(groups) => {
                assert_eq!(
                    groups,
                    HashMap::from([(
                        "code_point_groups".to_string(),
                        (
                            vec![vec![1, 1, 1, 1, 2, 1, 1, 1], vec![1, 1, 1, 1, 1, 1, 1, 1]],
                            GroupAggregation::Mean
                        )
                    )])
                )
            }
        };
    }

    #[test]
    fn test_byt5_tokenizer() {
        let tok = ByT5Tokenizer::new(true, ByteGroups::Bytes, GroupAggregation::Mean).unwrap();
        let Tokenization { token_ids, info: _ } = tok.tokenize("a täst", None).unwrap();
        assert_eq!(token_ids, vec![100, 35, 119, 198, 167, 118, 119, 1]);
    }

    #[test]
    fn test_token_groups_to_sparse_coo_matrix() {
        // one stage grouping
        let grouping = (vec![vec![1, 1, 1, 1, 2, 1, 1, 1]], GroupAggregation::Mean);
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
            vec![vec![1, 1, 1, 1, 2, 1, 1, 1], vec![4, 4]],
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
