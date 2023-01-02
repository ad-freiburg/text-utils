import copy
from typing import Any, Dict

from text_correction_utils._internal import tokenization_rs

Tokenizer = tokenization_rs.Tokenizer
TokenizerConfig = tokenization_rs.TokenizerConfig
LanguageConfig = tokenization_rs.LanguageConfig
SpecialTokens = tokenization_rs.SpecialTokens
LanguageTokens = tokenization_rs.LanguageTokens


def tokenizer_config(cfg: Dict[str, Any]) -> TokenizerConfig:
    cfg = copy.deepcopy(cfg)
    if "language" in cfg:
        language = LanguageConfig(**cfg.pop("language"))
    else:
        language = None
    return TokenizerConfig(**cfg, language=language)


def tokenizer_from_config(cfg: Dict[str, Any]) -> Tokenizer:
    return Tokenizer.from_config(tokenizer_config(cfg))
