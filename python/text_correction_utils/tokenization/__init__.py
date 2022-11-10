from text_correction_utils.configuration import TokenizerConfig
from text_correction_utils.tokenization import constants
from text_correction_utils.tokenization.base import Tokenizer
from text_correction_utils.tokenization.byte import ByteTokenizer
from text_correction_utils.tokenization.char import CharacterTokenizer


def get_tokenizer_from_config(config: TokenizerConfig) -> Tokenizer:
    if config.name == "char":
        return CharacterTokenizer(config.default_prefix_tokens, config.default_suffix_tokens, config.additional_tokens)
    elif config.name == "byte":
        return ByteTokenizer(config.default_prefix_tokens, config.default_suffix_tokens, config.additional_tokens)
    else:
        raise ValueError(f"unknown tokenizer {config.name}")
