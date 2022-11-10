from typing import Any, List, Tuple, Dict, Union, Optional

from text_correction_utils.tokenization import constants

Tokenization = Tuple[List[int], Dict[str, Any]]
AdditionalTokens = Tuple[str]
BatchAdditionalTokens = Union[AdditionalTokens, List[AdditionalTokens]]


class Tokenizer:
    vocab: Dict[str, int]

    def __init__(
            self,
            default_prefix_tokens: Tuple[str],
            default_suffix_tokens: Tuple[str]
    ):
        self.reverse_vocab = {
            v: k for k, v in self.vocab.items()
        }
        self.special_token_ids = {
            self.vocab[token] for token in constants.EXTENDED_SPECIAL_TOKENS if token in self.vocab
        }
        self.eos_token_id = self.vocab[constants.EOS]
        self.bos_token_id = self.vocab[constants.BOS]
        self.unk_token_id = self.vocab[constants.UNK]
        self.pad_token_id = self.vocab[constants.PAD]

        self.default_prefix_tokens = default_prefix_tokens
        self.default_suffix_tokens = default_suffix_tokens

    def _check_token_id(self, t: int, with_special_tokens: bool) -> bool:
        return with_special_tokens or t not in self.special_token_ids

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def num_prefix_tokens(self) -> int:
        return len(self.default_prefix_tokens)

    @property
    def num_suffix_tokens(self) -> int:
        return len(self.default_suffix_tokens)

    def normalize(self, sequence: str) -> str:
        return self.normalize_batch([sequence])[0]

    def normalize_batch(self, sequences: List[str]) -> List[str]:
        return [whitespace_correction.clean(sequence) for sequence in sequences]

    def split(self, sequence: str) -> List[str]:
        return self.split_batch([sequence])[0]

    def split_batch(self, sequences: List[str]) -> List[List[str]]:
        raise NotImplementedError

    def tokenize(
            self,
            sequence: str,
            prefix_tokens: Optional[AdditionalTokens] = None,
            suffix_tokens: Optional[AdditionalTokens] = None
    ) -> Tokenization:
        return self.tokenize_batch([sequence], prefix_tokens, suffix_tokens)[0]

    def _check_prefix_suffix_tokens(
            self,
            sequences: List[str],
            prefix_tokens: Optional[BatchAdditionalTokens] = None,
            suffix_tokens: Optional[BatchAdditionalTokens] = None
    ) -> Tuple[BatchAdditionalTokens, BatchAdditionalTokens]:
        prefix_tokens = prefix_tokens or self.default_prefix_tokens
        suffix_tokens = suffix_tokens or self.default_suffix_tokens
        if isinstance(prefix_tokens, tuple):
            prefix_tokens = [prefix_tokens] * len(sequences)
        if isinstance(suffix_tokens, tuple):
            suffix_tokens = [suffix_tokens] * len(sequences)
        assert len(prefix_tokens) == len(sequences) \
               and all(len(tokens) == self.num_prefix_tokens for tokens in prefix_tokens), \
            f"expected {self.num_prefix_tokens} prefix tokens for all sequences, " \
            f"but got {prefix_tokens}"
        assert len(suffix_tokens) == len(sequences) \
               and all(len(tokens) == self.num_suffix_tokens for tokens in suffix_tokens), \
            f"expected {self.num_suffix_tokens} suffix tokens for all sequences, " \
            f"but got {suffix_tokens}"

        return prefix_tokens, suffix_tokens

    def tokenize_batch(
            self,
            sequences: List[str],
            prefix_tokens: Optional[BatchAdditionalTokens] = None,
            suffix_tokens: Optional[BatchAdditionalTokens] = None
    ) -> List[Tokenization]:
        prefix_tokens, suffix_tokens = self._check_prefix_suffix_tokens(sequences, prefix_tokens, suffix_tokens)
        tokenizations = []
        for prefix, suffix, split in zip(prefix_tokens, suffix_tokens, self.split_batch(sequences)):
            token_ids = list(self.token_to_id(token) for token in prefix)
            for token in split:
                token_ids.append(self.token_to_id(token))
            token_ids.extend(self.token_to_id(token) for token in suffix)
            tokenizations.append((token_ids, {}))
        return tokenizations

    def de_tokenize(self, token_ids: List[int], with_special_tokens: bool = True) -> str:
        return self.de_tokenize_batch([token_ids], with_special_tokens)[0]

    def de_tokenize_batch(self, token_ids: List[List[int]], with_special_tokens: bool = True) -> List[str]:
        raise NotImplementedError

    def id_to_token(self, token_id: int) -> str:
        return self.reverse_vocab.get(token_id, constants.UNK)

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)
