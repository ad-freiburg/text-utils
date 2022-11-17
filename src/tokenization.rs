use std::collections::HashMap;
use itertools::Itertools;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use crate::unicode::CS;

pub const UNK: &str = "<unk>";
pub const BOS: &str = "<bos>";
pub const EOS: &str = "<eos>";
pub const PAD: &str = "<pad>";
pub const SPECIAL_TOKENS: [&str; 4] = [UNK, BOS, EOS, PAD];
pub const DEFAULT_PREFIX_TOKENS: [&str; 1] = [BOS];
pub const DEFAULT_SUFFIX_TOKENS: [&str; 1] = [EOS];

/// This enum defines all tokenizers that are supported by this crate.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum TokenizerConfig {
    Character(bool, Vec<String>, Vec<String>),
    Byte(bool, Vec<String>, Vec<String>),
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
    TokenGroups(Vec<usize>),
}

/// A tokenization is defined to be a tuple of token ids and some additional information.
/// This is returned by a tokenizers tokenize function.
pub type Tokenization = (Vec<u32>, TokenizationInfo);
/// A tokenization function in general takes in a &str and return a tokenization.
pub type TokenizationFn = Box<dyn FnMut(&str) -> Tokenization>;
/// A tokenizer is something that implements the tokenize trait with the
/// appropriate bounds on tokens and token ids.
pub type Tokenizer = Box<dyn Tokenize>;

/// The tokenize trait defines behavior that every tokenizer should support.
pub trait Tokenize: Send + Sync {
    fn vocab_size(&self) -> usize;

    fn unk_token_id(&self) -> u32;

    fn num_prefix_tokens(&self) -> usize;

    fn num_suffix_tokens(&self) -> usize;

    fn add_special_tokens(&mut self, special_tokens: &[String]);

    fn tokenize(&self, s: &str) -> Tokenization;

    fn tokenize_with(&self, s: &str, prefix: &[String], suffix: &[String]) -> Tokenization;

    fn de_tokenize(&self, token_ids: &[u32]) -> String;

    fn special_token_to_id(&self, token: &String) -> u32;

    fn id_to_special_token(&self, token_id: &u32) -> &String;
}

pub trait BatchTokenize: Tokenize {
    fn batch_tokenize(&self, s: &[String]) -> Vec<Tokenization>
        where Self: Sync {
        s
            .par_iter()
            .map(|s| self.tokenize(s))
            .collect()
    }

    fn batch_tokenize_with(
        &self,
        s: &[String],
        prefixes: &[Vec<String>],
        suffixes: &[Vec<String>],
    ) -> Vec<Tokenization>
        where Self: Sync {
        s
            .par_iter()
            .enumerate()
            .map(|(idx, s)| self.tokenize_with(s, &prefixes[idx], &suffixes[idx]))
            .collect()
    }
}

/// A tokenizer based on the ascii characters, digits, and punctuations marks.
/// Can e.g. be used to efficiently (meaning small vocab size) represent most
/// English texts.
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
        let vocab = HashMap::from_iter(
            CHARS
                .chars()
                .unique()
                .enumerate()
                .map(|(tok_id, c)| (c, tok_id as u32))
        );
        let reverse_vocab = HashMap::from_iter(
            vocab
                .iter()
                .map(|(token, token_id)| (*token_id, *token))
        );
        let special_vocab = HashMap::from_iter(
            SPECIAL_TOKENS
                .iter()
                .map(|&s| s.to_string())
                .zip(vocab.len() as u32..(vocab.len() + SPECIAL_TOKENS.len()) as u32)
        );
        let reverse_special_vocab = HashMap::from_iter(
            special_vocab
                .iter()
                .map(|(st, tok_id)| (*tok_id, st.clone()))
        );
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

    fn tokenize(&self, s: &str) -> Tokenization {
        self.tokenize_with(
            s,
            &self.default_prefix_tokens,
            &self.default_suffix_tokens,
        )
    }

    fn tokenize_with(
        &self,
        s: &str,
        prefix: &[String],
        suffix: &[String],
    ) -> Tokenization {
        (
            prefix
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
                        .iter()
                        .map(|t| self.special_token_to_id(t))
                )
                .collect(),
            TokenizationInfo::Empty
        )
    }

    fn de_tokenize(&self, token_ids: &[u32]) -> String {
        token_ids
            .iter()
            .filter_map(|i| self.reverse_vocab.get(i))
            .join("")
    }

    fn special_token_to_id(&self, token: &String) -> u32 {
        self.special_vocab
            .get(token.into())
            .copied()
            .unwrap_or(self.unk_token_id)
    }

    fn id_to_special_token(&self, token_id: &u32) -> &String {
        if let Some(token) = self.reverse_special_vocab.get(token_id) {
            token
        } else {
            &self.unk_token
        }
    }
}

impl BatchTokenize for CharTokenizer {}

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
        let special_vocab: HashMap<String, u32> = HashMap::from_iter(
            SPECIAL_TOKENS
                .iter()
                .zip(u8::MAX as u32..u8::MAX as u32 + SPECIAL_TOKENS.len() as u32)
                .map(|(&st, tok_id)| (st.into(), tok_id))
        );
        let reverse_special_vocab = HashMap::from_iter(
            special_vocab
                .iter()
                .map(|(token, token_id)| (*token_id, token.clone()))
        );
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

    fn tokenize(&self, s: &str) -> Tokenization {
        self.tokenize_with(
            s,
            &self.default_prefix_tokens,
            &self.default_suffix_tokens,
        )
    }

    fn tokenize_with(
        &self,
        s: &str,
        prefix: &[String],
        suffix: &[String],
    ) -> Tokenization {
        let (bytes, info) = self.split(s);
        (
            prefix
                .iter()
                .map(|t| self.special_token_to_id(t))
                .chain(bytes.into_iter())
                .chain(
                    suffix
                        .iter()
                        .map(|t| self.special_token_to_id(t))
                )
                .collect(),
            TokenizationInfo::TokenGroups(info)
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

    fn special_token_to_id(&self, token: &String) -> u32 {
        self.special_vocab
            .get(token)
            .copied()
            .unwrap_or(self.unk_token_id)
    }

    fn id_to_special_token(&self, token_id: &u32) -> &String {
        if let Some(token) = self.reverse_special_vocab.get(token_id) {
            token
        } else {
            &self.unk_token
        }
    }
}

impl BatchTokenize for ByteTokenizer {}

pub fn tokenizer(cfg: TokenizerConfig) -> Tokenizer {
    match cfg {
        TokenizerConfig::Character(use_g, pfx, sfx) => {
            Box::new(CharTokenizer::new(use_g, &pfx, &sfx))
        }
        TokenizerConfig::Byte(use_g, pfx, sfx) => {
            Box::new(ByteTokenizer::new(use_g, &pfx, &sfx))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tokenization::{BOS, EOS, CharTokenizer, Tokenize, ByteTokenizer};

    #[test]
    fn test_char_tokenizer() {
        let pfx = vec![BOS.to_string()];
        let sfx = vec![EOS.to_string()];
        let tok = CharTokenizer::new(
            true, &pfx, &sfx,
        );
        let text = "a täst";
        let (tokens, _) = tok.tokenize(text);
        assert_eq!(tokens.len(), 6 + 2);
        assert_eq!(tokens[4], tok.unk_token_id());
        assert_eq!(tok.de_tokenize(&tokens), String::from("a tst"));
    }

    #[test]
    fn test_byte_tokenizer() {
        let pfx = vec![BOS.to_string()];
        let sfx = vec![EOS.to_string()];
        let tok = ByteTokenizer::new(
            true, &pfx, &sfx,
        );
        let text = "a täst";
        let (tokens, _) = tok.tokenize(text);
        assert_eq!(
            tokens[1..tokens.len() - 1].iter().map(|tok| *tok as u8).collect::<Vec<u8>>(),
            text.as_bytes().clone()
        );
        assert_eq!(tokens.len(), 7 + 2);
        assert_eq!(tok.de_tokenize(&tokens), text.to_string());
    }
}
