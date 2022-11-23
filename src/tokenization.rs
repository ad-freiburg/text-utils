use std::collections::HashMap;
use std::thread::sleep;
use std::time::Duration;
use itertools::Itertools;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use crate::unicode::{CS};

pub const UNK: &str = "<unk>";
pub const BOS: &str = "<bos>";
pub const EOS: &str = "<eos>";
pub const PAD: &str = "<pad>";
pub const SPECIAL_TOKENS: [&str; 4] = [UNK, BOS, EOS, PAD];
pub const DEFAULT_PREFIX_TOKENS: [&str; 1] = [BOS];
pub const DEFAULT_SUFFIX_TOKENS: [&str; 1] = [EOS];
// language tokens
pub const LANG_UNK: &str = "[unk]";

/// This enum defines all tokenizers that are supported by this crate.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum TokenizerConfig {
    Character(bool, Vec<String>, Vec<String>),
    Byte(bool, Vec<String>, Vec<String>),
    Dummy(Duration),
}

impl<'a> FromPyObject<'a> for TokenizerConfig {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let d: &PyDict = ob.extract()?;
        let tokenizer_config = if d.contains("character")?
            || d.contains("byte")? {
            let character = d.contains("character")?;
            let config: &PyDict = d.get_item(if character { "character" } else { "byte" })
                .unwrap()
                .extract()?;
            let use_graphemes: bool = if let Some(value) = config.get_item("use_graphemes") {
                value.extract()?
            } else {
                true
            };
            let prefix: Vec<String> = if let Some(value) = config.get_item("default_prefix_tokens") {
                value.extract()?
            } else {
                vec![]
            };
            let suffix: Vec<String> = if let Some(value) = config.get_item("default_suffix_tokens") {
                value.extract()?
            } else {
                vec![]
            };
            if character {
                TokenizerConfig::Character(use_graphemes, prefix, suffix)
            } else {
                TokenizerConfig::Byte(use_graphemes, prefix, suffix)
            }
        } else if d.contains("dummy")? {
            let config: &PyDict = d.get_item("dummy")
                .unwrap()
                .extract()?;
            let millis: u64 = if let Some(value) = config.get_item("delay") {
                value.extract()?
            } else {
                0
            };
            TokenizerConfig::Dummy(Duration::from_millis(millis))
        } else {
            return Err(PyTypeError::new_err(format!("could not find a valid tokenizer name in \
            config")));
        };
        Ok(tokenizer_config)
    }
}

/// This enum defines all possible additional infos that can be returned by
/// a tokenizers tokenize function in addition to the token ids themselves.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenizationInfo {
    /// No additional info.
    Empty,
    /// Token groups specify which subsequent tokens belong to the same group.
    /// Useful e.g. when defining a byte tokenizer that should also return
    /// information about which byte belongs to which character.
    TokenGroups(Vec<usize>),
}

impl IntoPy<PyObject> for TokenizationInfo {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let d = PyDict::new(py);
        match self {
            TokenizationInfo::Empty => {}
            TokenizationInfo::TokenGroups(groups) => {
                d.set_item("token_groups", groups).unwrap();
            }
        }
        d.into()
    }
}

/// A tokenization is defined to be a tuple of token ids and some additional information.
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
        Tokenization {
            token_ids,
            info,
        }
    }
}

/// A tokenization function in general takes in a &str and return a tokenization.
pub type TokenizationFn = Box<dyn FnMut(&str) -> Tokenization>;
/// A tokenizer is something that implements the tokenize trait with the
/// appropriate bounds on tokens and token ids.
pub type Tokenizer = Box<dyn Tokenize>;

/// The tokenize trait defines behavior that every tokenizer should support.
pub trait Tokenize: Send {
    fn vocab_size(&self) -> usize;

    fn unk_token_id(&self) -> u32;

    fn num_prefix_tokens(&self) -> usize;

    fn num_suffix_tokens(&self) -> usize;

    fn add_special_tokens(&mut self, special_tokens: &[String]);

    fn tokenize(
        &self,
        s: &str,
        prefix: Option<&[String]>,
        suffix: Option<&[String]>,
    ) -> Tokenization;

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
        DummyTokenizer { delay, dummy_token: "".to_string() }
    }
}

impl Tokenize for DummyTokenizer {
    fn vocab_size(&self) -> usize {
        0
    }

    fn unk_token_id(&self) -> u32 {
        0
    }

    fn num_prefix_tokens(&self) -> usize {
        0
    }

    fn num_suffix_tokens(&self) -> usize {
        0
    }

    fn add_special_tokens(&mut self, _: &[String]) {}

    fn tokenize(
        &self,
        _: &str,
        _: Option<&[String]>,
        _: Option<&[String]>,
    ) -> Tokenization {
        sleep(self.delay.clone());
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
    default_prefix_tokens: Vec<String>,
    default_suffix_tokens: Vec<String>,
    vocab: HashMap<char, u32>,
    reverse_vocab: HashMap<u32, char>,
    special_vocab: HashMap<String, u32>,
    reverse_special_vocab: HashMap<u32, String>,
    unk_token_id: u32,
    unk_token: String,
    use_graphemes: bool,
}

const CHARS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\"\"!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\"\" ";

impl CharTokenizer {
    pub fn new(
        use_graphemes: bool,
        default_prefix_tokens: &[String],
        default_suffix_tokens: &[String],
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
            .iter()
            .map(|&s| s.to_string())
            .zip(vocab.len() as u32..(vocab.len() + SPECIAL_TOKENS.len()) as u32)
            .collect();
        let reverse_special_vocab = special_vocab
            .iter()
            .map(|(st, tok_id)| (*tok_id, st.clone()))
            .collect();
        let unk_token = UNK.to_string();
        let unk_token_id = *special_vocab
            .get(&unk_token)
            .expect("should not fail");
        CharTokenizer {
            default_prefix_tokens: default_prefix_tokens.into(),
            default_suffix_tokens: default_suffix_tokens.into(),
            vocab,
            reverse_vocab,
            special_vocab,
            reverse_special_vocab,
            unk_token_id,
            unk_token,
            use_graphemes,
        }
    }

    pub fn default() -> Self {
        let pfx: Vec<String> = DEFAULT_PREFIX_TOKENS
            .iter()
            .map(|&p| p.into())
            .collect();
        let sfx: Vec<String> = DEFAULT_SUFFIX_TOKENS
            .iter()
            .map(|&p| p.into())
            .collect();
        Self::new(true, &pfx, &sfx)
    }
}

impl Tokenize for CharTokenizer {
    fn vocab_size(&self) -> usize {
        self.vocab.len() + self.special_vocab.len()
    }

    fn unk_token_id(&self) -> u32 {
        self.unk_token_id
    }

    fn num_prefix_tokens(&self) -> usize {
        self.default_prefix_tokens.len()
    }

    fn num_suffix_tokens(&self) -> usize {
        self.default_suffix_tokens.len()
    }

    fn add_special_tokens(&mut self, special_tokens: &[String]) {
        for st in special_tokens {
            let token_id = self.vocab_size() as u32;
            if self.reverse_special_vocab.contains_key(&token_id) {
                panic!("cannot add any more tokens to the character tokenizer");
            }
            self.special_vocab.insert(st.clone(), token_id);
            self.reverse_special_vocab.insert(token_id, st.clone());
        }
    }

    fn tokenize(
        &self,
        s: &str,
        prefix: Option<&[String]>,
        suffix: Option<&[String]>,
    ) -> Tokenization {
        Tokenization::new(
            prefix
                .unwrap_or(&self.default_prefix_tokens)
                .iter()
                .map(|t| self.special_token_to_id(t))
                .chain(
                    CS::new(s, self.use_graphemes)
                        .chars()
                        .filter_map(|c| {
                            // Character always has at least one char so this is safe
                            let mut c_iter = c.code_points();
                            let char = c_iter.next().unwrap();
                            // return unk if Character has another char because
                            // our tokens in the vocab are all single char tokens
                            if c_iter.next().is_some() {
                                Some(self.unk_token_id)
                            } else {
                                Some(self.vocab
                                    .get(&char)
                                    .copied()
                                    .unwrap_or(self.unk_token_id))
                            }
                        })
                )
                .chain(
                    suffix
                        .unwrap_or(&self.default_suffix_tokens)
                        .iter()
                        .map(|t| self.special_token_to_id(t))
                )
                .collect(),
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
            .get(token.into())
            .copied()
            .unwrap_or(self.unk_token_id)
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
    default_prefix_tokens: Vec<String>,
    default_suffix_tokens: Vec<String>,
    special_vocab: HashMap<String, u32>,
    reverse_special_vocab: HashMap<u32, String>,
    unk_token_id: u32,
    unk_token: String,
    use_graphemes: bool,
}

impl ByteTokenizer {
    pub fn new(
        use_graphemes: bool,
        default_prefix_tokens: &[String],
        default_suffix_tokens: &[String],
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
        let unk_token_id = *special_vocab
            .get(&unk_token)
            .expect("should not fail");
        ByteTokenizer {
            default_prefix_tokens: default_prefix_tokens.into(),
            default_suffix_tokens: default_suffix_tokens.into(),
            special_vocab,
            reverse_special_vocab,
            unk_token_id,
            unk_token,
            use_graphemes,
        }
    }

    pub fn default() -> Self {
        let pfx: Vec<String> = DEFAULT_PREFIX_TOKENS
            .iter()
            .map(|&p| p.into())
            .collect();
        let sfx: Vec<String> = DEFAULT_SUFFIX_TOKENS
            .iter()
            .map(|&p| p.into())
            .collect();
        Self::new(true, &pfx, &sfx)
    }

    fn split(&self, s: &str) -> (Vec<u32>, Vec<usize>) {
        let tokens = s
            .as_bytes()
            .iter()
            .map(|b| *b as u32)
            .collect();
        (tokens, CS::new(s, self.use_graphemes).cluster_lengths)
    }
}

impl Tokenize for ByteTokenizer {
    fn vocab_size(&self) -> usize {
        u8::MAX as usize + self.special_vocab.len()
    }

    fn unk_token_id(&self) -> u32 {
        self.unk_token_id
    }

    fn num_prefix_tokens(&self) -> usize {
        self.default_prefix_tokens.len()
    }

    fn num_suffix_tokens(&self) -> usize {
        self.default_suffix_tokens.len()
    }

    fn add_special_tokens(&mut self, special_tokens: &[String]) {
        for st in special_tokens {
            let token_id = self.vocab_size() as u32;
            if self.reverse_special_vocab.contains_key(&token_id) {
                panic!("cannot add any more tokens to the character tokenizer");
            }
            self.special_vocab.insert(st.clone(), token_id);
            self.reverse_special_vocab.insert(token_id, st.clone());
        }
    }

    fn tokenize(
        &self,
        s: &str,
        prefix: Option<&[String]>,
        suffix: Option<&[String]>,
    ) -> Tokenization {
        let (bytes, info) = self.split(s);
        Tokenization::new(
            prefix
                .unwrap_or(&self.default_prefix_tokens)
                .iter()
                .map(|t| self.special_token_to_id(t))
                .chain(bytes.into_iter())
                .chain(
                    suffix
                        .unwrap_or(&self.default_suffix_tokens)
                        .iter()
                        .map(|t| self.special_token_to_id(t))
                )
                .collect(),
            TokenizationInfo::TokenGroups(info),
        )
    }

    fn de_tokenize(&self, token_ids: &[u32]) -> String {
        let bytes: Vec<u8> = token_ids
            .iter()
            .filter_map(|t| {
                if *t < 256u32 {
                    Some(*t as u8)
                } else {
                    None
                }
            })
            .collect();
        String::from_utf8(bytes).expect("invalid utf8")
    }

    fn special_token_to_id(&self, token: &str) -> u32 {
        self.special_vocab
            .get(token)
            .copied()
            .unwrap_or(self.unk_token_id)
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
    match cfg {
        TokenizerConfig::Character(use_g, pfx, sfx) => {
            Box::new(CharTokenizer::new(use_g, &pfx, &sfx))
        }
        TokenizerConfig::Byte(use_g, pfx, sfx) => {
            Box::new(ByteTokenizer::new(use_g, &pfx, &sfx))
        }
        TokenizerConfig::Dummy(d) => {
            Box::new(DummyTokenizer::new(d))
        }
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
    fn from_config(config: TokenizerConfig) -> PyResult<Self> {
        Ok(PyTokenizer {
            name: match config {
                TokenizerConfig::Character(_, _, _) => "character",
                TokenizerConfig::Byte(_, _, _) => "byte",
                TokenizerConfig::Dummy(_) => "dummy"
            }.to_string(),
            tokenizer: tokenizer(config),
        })
    }

    #[args(
    prefix = "None",
    suffix = "None"
    )]
    fn tokenize(
        &self,
        s: &str,
        prefix: Option<Vec<String>>,
        suffix: Option<Vec<String>>,
    ) -> PyResult<Tokenization> {
        Ok(self.tokenizer.tokenize(
            s,
            if prefix.is_some() {
                Some(prefix.as_ref().unwrap())
            } else {
                None
            },
            if suffix.is_some() {
                Some(suffix.as_ref().unwrap())
            } else {
                None
            }))
    }

    fn special_token_to_id(&self, token: &str) -> PyResult<u32> {
        Ok(self.tokenizer.special_token_to_id(token))
    }

    fn id_to_special_token(&self, token_id: u32) -> PyResult<&str> {
        Ok(self.tokenizer.id_to_special_token(&token_id))
    }

    fn de_tokenize(&self, token_ids: Vec<u32>) -> PyResult<String> {
        Ok(self.tokenizer.de_tokenize(&token_ids))
    }

    fn vocab_size(&self) -> PyResult<usize> {
        Ok(self.tokenizer.vocab_size())
    }

    fn num_prefix_tokens(&self) -> PyResult<usize> {
        Ok(self.tokenizer.num_prefix_tokens())
    }

    fn num_suffix_tokens(&self) -> PyResult<usize> {
        Ok(self.tokenizer.num_suffix_tokens())
    }

    fn add_special_tokens(&mut self, tokens: Vec<String>) -> PyResult<()> {
        self.tokenizer.add_special_tokens(&tokens);
        Ok(())
    }

    fn unk_token_id(&self) -> PyResult<u32> {
        Ok(self.tokenizer.unk_token_id())
    }
}

pub(super) fn add_submodule(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "tokenization")?;
    m.add_class::<PyTokenizer>()?;
    parent_module.add_submodule(m)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::tokenization::{BOS, EOS, CharTokenizer, Tokenize, ByteTokenizer, Tokenization};

    #[test]
    fn test_char_tokenizer() {
        let pfx = vec![BOS.to_string()];
        let sfx = vec![EOS.to_string()];
        let tok = CharTokenizer::new(
            true, &pfx, &sfx,
        );
        let text = "a täst";
        let Tokenization { token_ids, .. } = tok.tokenize(text, None, None);
        assert_eq!(token_ids.len(), 6 + 2);
        assert_eq!(token_ids[4], tok.unk_token_id());
        assert_eq!(tok.de_tokenize(&token_ids), "a tst".to_string());
    }

    #[test]
    fn test_byte_tokenizer() {
        let pfx = vec![BOS.to_string()];
        let sfx = vec![EOS.to_string()];
        let tok = ByteTokenizer::new(
            true, &pfx, &sfx,
        );
        let text = "a täst";
        let Tokenization { token_ids, .. } = tok.tokenize(text, None, None);
        assert_eq!(
            token_ids[1..token_ids.len() - 1]
                .iter()
                .map(|tok| *tok as u8).collect::<Vec<u8>>(),
            text.as_bytes().clone()
        );
        assert_eq!(token_ids.len(), 7 + 2);
        assert_eq!(tok.de_tokenize(&token_ids), text.to_string());
    }
}
