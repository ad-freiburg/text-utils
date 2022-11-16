use std::collections::HashMap;
use std::hash::Hash;
use itertools::Itertools;
use num::Unsigned as TokenID;
use crate::unicode::CS;

pub const UNK: &str = "<unk>";
pub const BOS: &str = "<bos>";
pub const EOS: &str = "<eos>";
pub const PAD: &str = "<pad>";
pub const SPECIAL_TOKENS: [&str; 4] = [UNK, BOS, EOS, PAD];
pub const DEFAULT_PREFIX_TOKENS: [&str; 1] = [BOS];
pub const DEFAULT_SUFFIX_TOKENS: [&str; 1] = [EOS];

/// This trait makes sure that a special token can be converted into a String,
/// dereferenced into &str, constructed from &str, and hashed (for vocabulary lookups).
/// One could e.g. just use a regular String, but we define our own special token struct below.
/// Important: We do not enforce how a regular token should look like since
/// they should be handled entirely by the tokenizer itself. E.g. a ByteTokenizer
/// does not need the notion of a regular token at all because it directly uses utf8
/// encoded bytes.
pub trait SpecialToken: Hash + Into<String> + AsRef<str> + for<'a> From<&'a str> + Clone {}

/// Our own special token struct (a wrapper around String).
/// We use this just for the better name.
#[derive(Hash, Debug, Clone, Eq, PartialEq)]
pub struct SpecialTok {
    t: String,
}

impl AsRef<str> for SpecialTok {
    fn as_ref(&self) -> &str {
        &self.t
    }
}

impl Into<String> for SpecialTok {
    fn into(self) -> String {
        self.t
    }
}

impl From<&str> for SpecialTok {
    fn from(s: &str) -> Self {
        SpecialTok { t: s.to_string() }
    }
}

impl SpecialToken for SpecialTok {}

/// This enum defines all tokenizers that are supported by this crate.
#[derive(Clone, Debug)]
pub enum TokenizerType<T: SpecialToken> {
    Character(Vec<T>, Vec<T>),
    Byte(Vec<T>, Vec<T>),
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
pub type Tokenization<ID> = (Vec<ID>, TokenizationInfo);
/// A tokenization function in general takes in a &str and return a tokenization.
pub type TokenizationFn<ID> = Box<dyn FnMut(&str) -> Tokenization<ID>>;
/// A tokenizer is something that implements the tokenize trait with the
/// appropriate bounds on tokens and token ids.
pub type Tokenizer<T, ID> = Box<dyn Tokenize<T, ID>>;

/// The tokenize trait defines behavior that every tokenizer should support.
pub trait Tokenize<T: SpecialToken, ID: TokenID> {
    fn vocab_size(&self) -> usize;

    fn unk_token_id(&self) -> ID;

    fn num_prefix_tokens(&self) -> usize;

    fn num_suffix_tokens(&self) -> usize;

    fn add_special_tokens(&mut self, special_tokens: &[T]);

    fn tokenize(&self, s: &str) -> Tokenization<ID>;

    fn tokenize_with(&self, s: &str, prefix: &[T], suffix: &[T]) -> Tokenization<ID>;

    fn de_tokenize(&self, token_ids: &[ID]) -> String;

    fn special_token_to_id(&self, token: &T) -> ID;

    fn id_to_special_token(&self, token_id: &ID) -> &T;
}

/// A tokenizer based on the ascii characters, digits, and punctuations marks.
/// Can e.g. be used to efficiently (meaning small vocab size) represent most
/// English texts.
pub struct CharTokenizer {
    default_prefix_tokens: Vec<SpecialTok>,
    default_suffix_tokens: Vec<SpecialTok>,
    vocab: HashMap<char, u16>,
    reverse_vocab: HashMap<u16, char>,
    special_vocab: HashMap<SpecialTok, u16>,
    reverse_special_vocab: HashMap<u16, SpecialTok>,
    unk_token_id: u16,
    unk_token: SpecialTok
}

const CHARS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\"\"!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\"\" ";

impl CharTokenizer {
    pub fn new(default_prefix_tokens: &[SpecialTok], default_suffix_tokens: &[SpecialTok]) -> Self
        where Self: Tokenize<SpecialTok, u16> {
        let vocab = HashMap::from_iter(
            CHARS
                .chars()
                .unique()
                .enumerate()
                .map(|(tok_id, c)| (c, tok_id as u16))
        );
        let reverse_vocab = HashMap::from_iter(
            vocab
                .iter()
                .map(|(token, token_id)| (*token_id, *token))
        );
        let special_vocab = HashMap::from_iter(
            SPECIAL_TOKENS
                .iter()
                .map(|&s| SpecialTok::from(s))
                .zip(vocab.len() as u16..(vocab.len() + SPECIAL_TOKENS.len()) as u16)
        );
        let reverse_special_vocab = HashMap::from_iter(
            special_vocab
                .iter()
                .map(|(st, tok_id)| (*tok_id, st.clone()))
        );
        let unk_token = UNK.into();
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
            unk_token
        }
    }

    pub fn default() -> Self {
        let pfx: Vec<SpecialTok> = DEFAULT_PREFIX_TOKENS
            .iter()
            .map(|&p| p.into())
            .collect();
        let sfx: Vec<SpecialTok> = DEFAULT_SUFFIX_TOKENS
            .iter()
            .map(|&p| p.into())
            .collect();
        Self::new(&pfx, &sfx)
    }
}

impl Tokenize<SpecialTok, u16> for CharTokenizer {
    fn vocab_size(&self) -> usize {
        self.vocab.len() + self.special_vocab.len()
    }

    fn unk_token_id(&self) -> u16 {
        self.unk_token_id
    }

    fn num_prefix_tokens(&self) -> usize {
        self.default_prefix_tokens.len()
    }

    fn num_suffix_tokens(&self) -> usize {
        self.default_suffix_tokens.len()
    }

    fn add_special_tokens(&mut self, special_tokens: &[SpecialTok]) {
        for st in special_tokens {
            let token_id = self.vocab_size();
            if token_id > u16::MAX as usize {
                panic!("cannot add more than {} tokens to a character tokenizer", u16::MAX);
            }
            let token_id = token_id as u16;
            self.special_vocab.insert(st.clone(), token_id);
            self.reverse_special_vocab.insert(token_id, st.clone());
        }
    }

    fn tokenize(&self, s: &str) -> Tokenization<u16> {
        self.tokenize_with(
            s,
            &self.default_prefix_tokens,
            &self.default_suffix_tokens,
        )
    }

    fn tokenize_with(
        &self,
        s: &str,
        prefix: &[SpecialTok],
        suffix: &[SpecialTok],
    ) -> Tokenization<u16> {
        (
            prefix
                .iter()
                .map(|t| self.special_token_to_id(t))
                .chain(
                    s
                        .chars()
                        .map(|c| self.vocab
                            .get(&c)
                            .copied()
                            .unwrap_or(self.unk_token_id))
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

    fn de_tokenize(&self, token_ids: &[u16]) -> String {
        token_ids
            .iter()
            .filter(|&&i| i < self.vocab.len() as u16)
            .map(|i| self.reverse_vocab[i])
            .join("")
    }

    fn special_token_to_id(&self, token: &SpecialTok) -> u16 {
        self.special_vocab
            .get(token.into())
            .copied()
            .unwrap_or(self.unk_token_id)
    }

    fn id_to_special_token(&self, token_id: &u16) -> &SpecialTok {
        if let Some(token) = self.reverse_special_vocab.get(token_id) {
            token
        } else {
            &self.unk_token
        }
    }
}

pub struct ByteTokenizer {
    default_prefix_tokens: Vec<SpecialTok>,
    default_suffix_tokens: Vec<SpecialTok>,
    special_vocab: HashMap<SpecialTok, u16>,
    reverse_special_vocab: HashMap<u16, SpecialTok>,
    unk_token_id: u16,
    unk_token: SpecialTok
}

impl ByteTokenizer {
    pub fn new(default_prefix_tokens: &[SpecialTok], default_suffix_tokens: &[SpecialTok]) -> Self
        where Self: Tokenize<SpecialTok, u16> {
        let special_vocab: HashMap<SpecialTok, u16> = HashMap::from_iter(
            SPECIAL_TOKENS
                .iter()
                .zip(u8::MAX as u16..u8::MAX as u16 + SPECIAL_TOKENS.len() as u16)
                .map(|(&st, tok_id)| (st.into(), tok_id))
        );
        let reverse_special_vocab = HashMap::from_iter(
            special_vocab
                .iter()
                .map(|(token, token_id)| (*token_id, token.clone()))
        );
        let unk_token = UNK.into();
        let unk_token_id = *special_vocab
            .get(&unk_token)
            .expect("should not fail");
        ByteTokenizer {
            default_prefix_tokens: default_prefix_tokens.into(),
            default_suffix_tokens: default_suffix_tokens.into(),
            special_vocab,
            reverse_special_vocab,
            unk_token_id,
            unk_token
        }
    }

    pub fn default() -> Self {
        let pfx: Vec<SpecialTok> = DEFAULT_PREFIX_TOKENS
            .iter()
            .map(|&p| p.into())
            .collect();
        let sfx: Vec<SpecialTok> = DEFAULT_SUFFIX_TOKENS
            .iter()
            .map(|&p| p.into())
            .collect();
        Self::new(&pfx, &sfx)
    }

    fn split(&self, s: &str) -> (Vec<u16>, Vec<usize>) {
        let tokens = s
            .as_bytes()
            .iter()
            .map(|b| *b as u16)
            .collect();
        (tokens, CS::new(s, true).cluster_lengths)
    }
}

impl Tokenize<SpecialTok, u16> for ByteTokenizer {
    fn vocab_size(&self) -> usize {
        u8::MAX as usize + self.special_vocab.len()
    }

    fn unk_token_id(&self) -> u16 {
        self.unk_token_id
    }

    fn num_prefix_tokens(&self) -> usize {
        self.default_prefix_tokens.len()
    }

    fn num_suffix_tokens(&self) -> usize {
        self.default_suffix_tokens.len()
    }

    fn add_special_tokens(&mut self, special_tokens: &[SpecialTok]) {
        for st in special_tokens {
            let token_id = self.vocab_size();
            if token_id > u16::MAX as usize {
                panic!("cannot add more than {} tokens to a byte tokenizer", u16::MAX);
            }
            let token_id = token_id as u16;
            self.special_vocab.insert(st.clone(), token_id);
            self.reverse_special_vocab.insert(token_id, st.clone());
        }
    }

    fn tokenize(&self, s: &str) -> Tokenization<u16> {
        self.tokenize_with(
            s,
            &self.default_prefix_tokens,
            &self.default_suffix_tokens
        )
    }

    fn tokenize_with(
        &self,
        s: &str,
        prefix: &[SpecialTok],
        suffix: &[SpecialTok],
    ) -> Tokenization<u16> {
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

    fn de_tokenize(&self, token_ids: &[u16]) -> String {
        let bytes: Vec<u8> = token_ids
            .iter()
            .filter(|&&t| t < u8::MAX as u16)
            .map(|&t| t as u8)
            .collect();
        String::from_utf8(bytes).expect("invalid utf8")
    }

    fn special_token_to_id(&self, token: &SpecialTok) -> u16 {
        self.special_vocab
            .get(token)
            .copied()
            .unwrap_or(self.unk_token_id)
    }

    fn id_to_special_token(&self, token_id: &u16) -> &SpecialTok {
        if let Some(token) = self.reverse_special_vocab.get(token_id) {
            token
        } else {
            &self.unk_token
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tokenization::{get_tokenizer, TokenizerType, BOS, EOS};

    #[test]
    fn test_char_tokenizer() {
        let pfx = vec![BOS.to_string()];
        let sfx = vec![EOS.to_string()];
        let tok = get_tokenizer(
            TokenizerType::Character(pfx, sfx)
        );
        let text = "a täst";
        let (tokens, _) = tok.tokenize(text, None, None);
        assert_eq!(tokens.len(), 6 + 2);
        assert_eq!(tokens[4], tok.unk_token_id());
        assert_eq!(tok.de_tokenize(&tokens), String::from("a tst"));
    }

    #[test]
    fn test_byte_tokenizer() {
        let pfx = vec![BOS.to_string()];
        let sfx = vec![EOS.to_string()];
        let tok = get_tokenizer(
            TokenizerType::Byte(pfx, sfx)
        );
        let text = "a täst";
        let (tokens, _) = tok.tokenize(text, None, None);
        assert_eq!(
            tokens[1..tokens.len() - 1].iter().map(|tok| *tok as u8).collect::<Vec<u8>>(),
            text.as_bytes().clone()
        );
        assert_eq!(tokens.len(), 7 + 2);
        assert_eq!(tok.de_tokenize(&tokens), text.to_string());
    }
}
