from typing import Dict, Any, Optional, List, Tuple

from text_correction_utils.configuration import BaseConfig


class TokenizerConfig(BaseConfig):
    required_arguments = {"name"}

    def __init__(
            self,
            name: str,
            default_prefix_tokens: Tuple[str],
            default_suffix_tokens: Tuple[str],
            additional_tokens: Optional[List[str]]
    ) -> None:
        self.name = name
        self.default_prefix_tokens = default_prefix_tokens
        self.default_suffix_tokens = default_suffix_tokens
        self.additional_tokens = additional_tokens

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TokenizerConfig":
        cls._check_required(d)
        config = TokenizerConfig(
            name=d["name"],
            default_prefix_tokens=d.get("default_prefix_tokens", ()),
            default_suffix_tokens=d.get("default_suffix_tokens", ()),
            additional_tokens=d.get("additional_tokens")
        )
        return config
