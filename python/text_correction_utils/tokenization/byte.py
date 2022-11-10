from typing import Tuple, Optional, List

from text_correction_utils.tokenization import Tokenizer, constants
from text_correction_utils.tokenization.base import BatchAdditionalTokens, Tokenization


class ByteTokenizer(Tokenizer):
    def __init__(
            self,
            default_prefix_tokens: Tuple[str],
            default_suffix_tokens: Tuple[str],
            additional_tokens: Optional[List[str]] = None
    ) -> None:
        self.vocab = {
            **{chr(i): i for i in range(256)},
            **{st: 256 + i for i, st in enumerate(constants.SPECIAL_TOKENS)}
        }
        if additional_tokens is not None:
            self.vocab = {
                **self.vocab,
                **{at: len(self.vocab) + i for i, at in enumerate(additional_tokens)}
            }
        super().__init__(default_prefix_tokens, default_suffix_tokens)

        self.special_token_bytes = {
            self.token_to_id(token): list(token.encode("utf8")) for token in constants.SPECIAL_TOKENS
        }

    def split_batch(self, sequences: List[str]) -> List[List[str]]:
        return [list(chr(b) for b in sequence.encode("utf8")) for sequence in sequences]

    def _split_batch_extended(
            self,
            sequences: List[str]
    ) -> List[Tuple[List[str], List[int]]]:
        batch = []
        for sequence in sequences:
            splits = []
            char_groups = [1] * self.num_prefix_tokens
            for char in sequence:
                char_bytes = char.encode("utf8")
                splits.extend(chr(b) for b in char_bytes)
                char_groups.append(len(char_bytes))
            char_groups.extend([1] * self.num_suffix_tokens)
            batch.append((splits, char_groups))
        return batch

    def tokenize_batch(
            self,
            sequences: List[str],
            prefix_tokens: Optional[BatchAdditionalTokens] = None,
            suffix_tokens: Optional[BatchAdditionalTokens] = None
    ) -> List[Tokenization]:
        prefix_tokens, suffix_tokens = self._check_prefix_suffix_tokens(sequences, prefix_tokens, suffix_tokens)
        tokenizations = []
        for prefix, suffix, (split, char_groups) in zip(
                prefix_tokens,
                suffix_tokens,
                self._split_batch_extended(sequences)
        ):
            token_ids = list(self.token_to_id(token) for token in prefix)
            for token in split:
                token_ids.append(self.token_to_id(token))
            token_ids.extend(self.token_to_id(token) for token in suffix)
            tokenizations.append((token_ids, {"char_groups": char_groups}))
        return tokenizations

    def de_tokenize_batch(self, token_ids: List[List[int]], with_special_tokens: bool = True) -> List[str]:
        sequences = []
        for ids in token_ids:
            token_bytes = []
            for token_id in ids:
                is_special_token = token_id in self.special_token_ids
                if with_special_tokens and is_special_token:
                    token_bytes.extend(self.special_token_bytes[token_id])
                elif not is_special_token:
                    token_bytes.append(token_id)
            sequences.append(bytes(token_bytes).decode("utf8"))
        return sequences
