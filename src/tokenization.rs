use std::collections::HashMap;
use itertools::Itertools;
use crate::unicode::CS;

pub const UNK: &str = "<unk>";
pub const BOS: &str = "<bos>";
pub const EOS: &str = "<eos>";
pub const PAD: &str = "<pad>";
pub const SPECIAL_TOKENS: [&str; 4] = [UNK, BOS, EOS, PAD];

pub enum TokenizerType {
    Character(usize, usize),
    Byte(usize, usize),
}

pub enum TokenizationInfo {
    Empty,
    TokenGroups(Vec<usize>),
}

pub trait Tokenizer {
    fn get_vocab_size(&self) -> usize;

    fn get_unk_token_id(&self) -> usize;

    fn add_special_tokens(&mut self, special_tokens: &Vec<String>);

    fn tokenize(&self, s: &str, prefix: &Vec<String>, suffix: &Vec<String>) -> (Vec<usize>, TokenizationInfo);

    fn de_tokenize(&self, token_ids: &Vec<usize>) -> String;

    fn special_token_to_id(&self, token: &String) -> usize;

    fn id_to_special_token(&self, token_id: &usize) -> String;
}

pub struct CharTokenizer {
    num_prefix_tokens: usize,
    num_suffix_tokens: usize,
    vocab: HashMap<char, usize>,
    reverse_vocab: HashMap<usize, char>,
    special_vocab: HashMap<String, usize>,
    reverse_special_vocab: HashMap<usize, String>,
    unk_token_id: usize,
}

const CHARS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\"\"!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\"\" ";

impl CharTokenizer {
    pub fn new(num_prefix_tokens: usize, num_suffix_tokens: usize) -> CharTokenizer {
        let vocab = HashMap::from_iter(
            CHARS
                .chars()
                .unique()
                .enumerate()
                .map(|(tok_id, c)| (c, tok_id))
        );
        let reverse_vocab = HashMap::from_iter(
            vocab
                .iter()
                .map(|(token, token_id)| (*token_id, *token))
        );
        let special_vocab = HashMap::from_iter(
            SPECIAL_TOKENS
                .iter()
                .map(|s| s.to_string())
                .zip(vocab.len()..(vocab.len() + SPECIAL_TOKENS.len()))
        );
        let reverse_special_vocab = HashMap::from_iter(
            special_vocab
                .iter()
                .map(|(st, tok_id)| (*tok_id, st.clone()))
        );
        assert_eq!(vocab.len(), reverse_vocab.len());
        assert_eq!(special_vocab.len(), reverse_special_vocab.len());
        let unk_token_id = *special_vocab.get(UNK).expect("should not fail");
        CharTokenizer {
            num_prefix_tokens,
            num_suffix_tokens,
            vocab,
            reverse_vocab,
            special_vocab,
            reverse_special_vocab,
            unk_token_id,
        }
    }
}

impl Tokenizer for CharTokenizer {
    fn get_vocab_size(&self) -> usize {
        self.vocab.len() + self.special_vocab.len()
    }

    fn get_unk_token_id(&self) -> usize {
        self.unk_token_id
    }

    fn add_special_tokens(&mut self, special_tokens: &Vec<String>) {
        for st in special_tokens {
            let token_id = self.get_vocab_size();
            self.special_vocab.insert(st.clone(), token_id);
            self.reverse_special_vocab.insert(token_id, st.clone());
        }
    }

    fn tokenize(&self, s: &str, prefix: &Vec<String>, suffix: &Vec<String>) -> (Vec<usize>, TokenizationInfo) {
        assert_eq!(self.num_prefix_tokens, prefix.len());
        assert_eq!(self.num_suffix_tokens, suffix.len());
        (
            prefix
                .iter()
                .map(|t| self.special_token_to_id(t))
                .chain(s.chars().map(|c| self.vocab.get(&c).copied().unwrap_or(self.unk_token_id)))
                .chain(suffix.iter().map(|t| self.special_token_to_id(t)))
                .collect(),
            TokenizationInfo::Empty
        )
    }

    fn de_tokenize(&self, token_ids: &Vec<usize>) -> String {
        token_ids
            .iter()
            .filter(|&&i| i < self.vocab.len())
            .map(|i| self.reverse_vocab[i])
            .join("")
    }

    fn special_token_to_id(&self, token: &String) -> usize {
        self.special_vocab
            .get(token)
            .copied()
            .unwrap_or(self.unk_token_id)
    }

    fn id_to_special_token(&self, token_id: &usize) -> String {
        self.reverse_special_vocab
            .get(token_id)
            .cloned()
            .unwrap_or(UNK.to_string())
    }
}

pub struct ByteTokenizer {
    num_prefix_tokens: usize,
    num_suffix_tokens: usize,
    special_vocab: HashMap<String, usize>,
    reverse_special_vocab: HashMap<usize, String>,
    unk_token_id: usize,
}

impl ByteTokenizer {
    pub fn new(num_prefix_tokens: usize, num_suffix_tokens: usize) -> ByteTokenizer {
        let special_vocab = HashMap::from_iter(
            SPECIAL_TOKENS
                .iter()
                .zip(u8::MAX as usize..u8::MAX as usize + SPECIAL_TOKENS.len())
                .map(|(st, tok_id)| (st.to_string(), tok_id))
        );
        let reverse_special_vocab = HashMap::from_iter(
            special_vocab
                .iter()
                .map(|(token, token_id)| (*token_id, token.clone()))
        );
        let unk_token_id = *special_vocab.get(UNK).expect("should not fail");
        ByteTokenizer {
            num_prefix_tokens,
            num_suffix_tokens,
            special_vocab,
            reverse_special_vocab,
            unk_token_id,
        }
    }

    fn split(&self, s: &str) -> (Vec<u8>, Vec<usize>) {
        (
            s.as_bytes().into(),
            CS::new(s, true).cluster_lengths
        )
    }
}

impl Tokenizer for ByteTokenizer {
    fn get_vocab_size(&self) -> usize {
        u8::MAX as usize + self.special_vocab.len()
    }

    fn get_unk_token_id(&self) -> usize {
        self.unk_token_id
    }

    fn add_special_tokens(&mut self, special_tokens: &Vec<String>) {
        for st in special_tokens {
            let token_id = self.get_vocab_size();
            self.special_vocab.insert(st.clone(), token_id);
            self.reverse_special_vocab.insert(token_id, st.clone());
        }
    }

    fn tokenize(&self, s: &str, prefix: &Vec<String>, suffix: &Vec<String>) -> (Vec<usize>, TokenizationInfo) {
        assert_eq!(self.num_prefix_tokens, prefix.len());
        assert_eq!(self.num_suffix_tokens, suffix.len());
        let (bytes, info) = self.split(s);
        (
            prefix
                .iter()
                .map(|t| self.special_token_to_id(t))
                .chain(bytes.into_iter().map(usize::from))
                .chain(suffix.iter().map(|t| self.special_token_to_id(t)))
                .collect(),
            TokenizationInfo::TokenGroups(info)
        )
    }

    fn de_tokenize(&self, token_ids: &Vec<usize>) -> String {
        let bytes: Vec<u8> = token_ids
            .iter()
            .filter(|&&t| t < u8::MAX as usize)
            .map(|&t| t as u8)
            .collect();
        String::from_utf8(bytes).expect("invalid utf8")
    }

    fn special_token_to_id(&self, token: &String) -> usize {
        self.special_vocab
            .get(token)
            .copied()
            .unwrap_or(self.unk_token_id)
    }

    fn id_to_special_token(&self, token_id: &usize) -> String {
        self.reverse_special_vocab
            .get(token_id)
            .cloned()
            .unwrap_or(UNK.to_string())
    }
}

pub fn get_tokenizer(tokenizer: TokenizerType) -> Box<dyn Tokenizer> {
    match tokenizer {
        TokenizerType::Character(pfx, sfx) =>
            Box::new(CharTokenizer::new(pfx, sfx)),
        TokenizerType::Byte(pfx, sfx) =>
            Box::new(ByteTokenizer::new(pfx, sfx))
    }
}

#[cfg(test)]
mod tests {
    use crate::tokenization::{get_tokenizer, TokenizerType, BOS, EOS};

    #[test]
    fn test_char_tokenizer() {
        let pfx = vec![BOS.to_string()];
        let sfx = vec![EOS.to_string()];
        let tok = get_tokenizer(TokenizerType::Character(1, 1));
        let text = "a täst";
        let (tokens, _) = tok.tokenize(text, &pfx, &sfx);
        assert_eq!(tokens.len(), 6 + 2);
        assert_eq!(tokens[4], tok.get_unk_token_id());
        assert_eq!(tok.de_tokenize(&tokens), String::from("a tst"));
    }

    #[test]
    fn test_byte_tokenizer() {
        let pfx = vec![BOS.to_string()];
        let sfx = vec![EOS.to_string()];
        let tok = get_tokenizer(TokenizerType::Byte(1, 1));
        let text = "a täst";
        let (tokens, _) = tok.tokenize(text, &pfx, &sfx);
        assert_eq!(
            tokens[1..tokens.len() - 1].iter().map(|tok| *tok as u8).collect::<Vec<u8>>(),
            text.as_bytes().clone()
        );
        assert_eq!(tokens.len(), 7 + 2);
        assert_eq!(tok.de_tokenize(&tokens), text.to_string());
    }
}
