import functools
import string
from typing import Tuple, Optional, List

from text_correction_utils.tokenization import Tokenizer, constants

_ALL_CHARS = string.ascii_letters + string.digits + string.punctuation + " "


class CharacterTokenizer(Tokenizer):
    def __init__(
            self,
            default_prefix_tokens: Tuple[str],
            default_suffix_tokens: Tuple[str],
            additional_tokens: Optional[List[str]] = None
    ) -> None:
        self.vocab = {
            **{c: i for i, c in enumerate(_ALL_CHARS)},
            **{st: len(_ALL_CHARS) + i for i, st in enumerate(constants.SPECIAL_TOKENS)}
        }
        if additional_tokens is not None:
            self.vocab = {
                **self.vocab,
                **{at: len(self.vocab) + i for i, at in enumerate(additional_tokens)}
            }
        super().__init__(default_prefix_tokens, default_suffix_tokens)

    def split_batch(self, sequences: List[str]) -> List[List[str]]:
        return [list(sequence) for sequence in sequences]

    def de_tokenize_batch(self, token_ids: List[List[int]], with_special_tokens: bool = True) -> List[str]:
        sequences = []
        filter_fn = functools.partial(self._check_token_id, with_special_tokens=with_special_tokens)
        for ids in token_ids:
            sequences.append(
                "".join(self.id_to_token(token_id) for token_id in filter(filter_fn, ids))
            )
        return sequences
