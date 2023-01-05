use crate::unicode::CS;
use crate::utils::{py_invalid_type_error, py_required_key_error, run_length_decode};
use anyhow::anyhow;
use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::PyDict;
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
#[pyclass]
pub struct TokenizerConfig {
    tokenize: TokenizeConfig,
    prefix: Vec<String>,
    suffix: Vec<String>,
    language: Option<LanguageConfig>,
}

#[pymethods]
impl TokenizerConfig {
    #[new]
    #[args(prefix = "vec![]", suffix = "vec![]", language = "None")]
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

/// This configures the language a tokenizer can work with
#[derive(Clone, Debug)]
#[pyclass]
pub struct LanguageConfig {
    add_language_token_to_prefix: bool,
    add_language_token_to_suffix: bool,
    languages: Vec<String>,
    default_language: String,
}

#[pymethods]
impl LanguageConfig {
    #[new]
    #[args(
        add_language_token_to_prefix = "true",
        add_language_token_to_suffix = "false",
        languages = "vec![]",
        default_language = "LANG_UNK.to_string()"
    )]
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

/// This enum defines all tokenizers that are supported by this crate.
#[derive(Clone, Debug)]
pub enum TokenizeConfig {
    Character(bool),
    Byte(bool),
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
            TokenizeConfig::Byte(use_g) => {
                d.set_item("use_graphemes", use_g).unwrap();
                "byte"
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
            "character" | "byte" => {
                let use_graphemes: bool = if let Some(value) = d.get_item("use_graphemes") {
                    value.extract()?
                } else {
                    true
                };
                if tokenizer_type.as_str() == "character" {
                    TokenizeConfig::Character(use_graphemes)
                } else {
                    TokenizeConfig::Byte(use_graphemes)
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

/// This enum defines all possible additional infos that can be returned by
/// a tokenizers tokenize function in addition to the token ids themselves.
#[derive(Clone, Debug)]
pub enum TokenizationInfo {
    /// No additional info.
    Empty,
    /// Token groups specify which subsequent tokens belong to the same group.
    /// Useful e.g. when defining a byte tokenizer that should also return
    /// information about which byte belongs to which character.
    TokenGroups(HashMap<String, Vec<usize>>),
}

impl IntoPy<PyObject> for TokenizationInfo {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let d = PyDict::new(py);
        let info_type = match self {
            TokenizationInfo::Empty => "empty",
            TokenizationInfo::TokenGroups(token_groups) => {
                for (key, groups) in token_groups.iter() {
                    d.set_item(key, groups).unwrap();
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
                    let groups = d
                        .get_item(key)
                        .ok_or_else(|| anyhow!("should not happen"))?;
                    token_groups.insert(key_s, groups.extract()?);
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

    fn add_prefix_and_suffix(&self, mut token_ids: Vec<u32>, lang: Option<&str>) -> Vec<u32> {
        let mut prefix = self.prefix_token_ids().to_vec();
        let mut suffix = self.suffix_token_ids().to_vec();
        if let Some(lang_cfg) = self.language_config() {
            let lang_id = self.special_token_to_id(lang.unwrap_or(&lang_cfg.default_language));
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
        prefix
    }

    fn add_special_token(&mut self, special_token: &str);

    fn add_special_tokens(&mut self, special_tokens: &[&str]) {
        for special_token in special_tokens {
            self.add_special_token(special_token);
        }
    }

    fn tokenize(&self, s: &str, lang: Option<&str>) -> Tokenization;

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

    fn tokenize(&self, _: &str, _: Option<&str>) -> Tokenization {
        sleep(self.delay);
        Tokenization::new(vec![], TokenizationInfo::Empty)
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

    pub fn default() -> Self {
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

    fn tokenize(&self, s: &str, lang: Option<&str>) -> Tokenization {
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
        Tokenization::new(
            self.add_prefix_and_suffix(token_ids, lang),
            TokenizationInfo::Empty,
        )
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
pub struct ByteTokenizer {
    _prefix_token_ids: Vec<u32>,
    _suffix_token_ids: Vec<u32>,
    language_config: Option<LanguageConfig>,
    special_vocab: HashMap<String, u32>,
    reverse_special_vocab: HashMap<u32, String>,
    unk_token: String,
    use_graphemes: bool,
}

impl ByteTokenizer {
    pub fn new(
        use_graphemes: bool,
        prefix_tokens: &[&str],
        suffix_tokens: &[&str],
        language_config: Option<LanguageConfig>,
    ) -> Self {
        let special_vocab: HashMap<String, u32> = SPECIAL_TOKENS
            .iter()
            .zip(u8::MAX as u32..u8::MAX as u32 + SPECIAL_TOKENS.len() as u32)
            .map(|(&st, tok_id)| (st.into(), tok_id))
            .collect();
        let reverse_special_vocab = special_vocab
            .iter()
            .map(|(token, token_id)| (*token_id, token.clone()))
            .collect();
        let unk_token = UNK.to_string();
        let mut tokenizer = ByteTokenizer {
            _prefix_token_ids: vec![],
            _suffix_token_ids: vec![],
            language_config,
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

    pub fn default() -> Self {
        Self::new(true, &DEFAULT_PREFIX_TOKENS, &DEFAULT_SUFFIX_TOKENS, None)
    }

    fn split(&self, s: &str) -> (Vec<u32>, Vec<usize>) {
        let tokens = s.as_bytes().iter().map(|b| *b as u32).collect();
        (
            tokens,
            run_length_decode(&CS::new(s, self.use_graphemes).rle_cluster_lengths),
        )
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

    fn tokenize(&self, s: &str, lang: Option<&str>) -> Tokenization {
        let (bytes, mut token_groups) = self.split(s);
        let mut groups: Vec<usize> = vec![1; self.num_prefix_tokens()];
        groups.append(&mut token_groups);
        groups.append(&mut vec![1; self.num_suffix_tokens()]);

        Tokenization::new(
            self.add_prefix_and_suffix(bytes, lang),
            TokenizationInfo::TokenGroups(HashMap::from([("groups".to_string(), groups)])),
        )
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

pub fn tokenizer(cfg: TokenizerConfig) -> Tokenizer {
    let prefix: Vec<&str> = cfg.prefix.iter().map(String::as_str).collect();
    let suffix: Vec<&str> = cfg.suffix.iter().map(String::as_str).collect();
    match cfg.tokenize {
        TokenizeConfig::Character(use_g) => {
            Box::new(CharTokenizer::new(use_g, &prefix, &suffix, cfg.language))
        }
        TokenizeConfig::Byte(use_g) => {
            Box::new(ByteTokenizer::new(use_g, &prefix, &suffix, cfg.language))
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
                TokenizeConfig::Byte(_) => "byte",
                TokenizeConfig::Dummy(_) => "dummy",
            }
            .to_string(),
            tokenizer: tokenizer(config),
        }
    }

    #[args(lang = "None")]
    fn tokenize(&self, s: &str, lang: Option<&str>) -> Tokenization {
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

    fn add_special_tokens(&mut self, tokens: Vec<&str>) -> () {
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
    let m = PyModule::new(py, "tokenization_rs")?;
    m.add_class::<PyTokenizer>()?;
    m.add_class::<Tokenization>()?;
    m.add_class::<TokenizerConfig>()?;
    m.add_class::<LanguageConfig>()?;
    m.add_class::<SpecialTokens>()?;
    m.add_class::<LanguageTokens>()?;
    parent_module.add_submodule(m)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::tokenization::{ByteTokenizer, CharTokenizer, Tokenization, Tokenize, BOS, EOS};

    #[test]
    fn test_char_tokenizer() {
        let pfx = vec![BOS];
        let sfx = vec![EOS];
        let tok = CharTokenizer::new(true, &pfx, &sfx, None);
        let text = "a täst";
        let Tokenization { token_ids, .. } = tok.tokenize(text, None);
        assert_eq!(token_ids.len(), 6 + 2);
        assert_eq!(token_ids[4], tok.unk_token_id());
        assert_eq!(tok.de_tokenize(&token_ids), "a tst".to_string());
    }

    #[test]
    fn test_byte_tokenizer() {
        let pfx = vec![BOS];
        let sfx = vec![EOS];
        let tok = ByteTokenizer::new(true, &pfx, &sfx, None);
        let text = "a täst";
        let Tokenization { token_ids, .. } = tok.tokenize(text, None);
        assert_eq!(
            token_ids[1..token_ids.len() - 1]
                .iter()
                .map(|tok| *tok as u8)
                .collect::<Vec<u8>>(),
            text.as_bytes().clone()
        );
        assert_eq!(token_ids.len(), 7 + 2);
        assert_eq!(tok.de_tokenize(&token_ids), text.to_string());
    }
}
