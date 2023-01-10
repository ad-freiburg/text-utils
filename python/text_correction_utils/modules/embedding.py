import math
import copy
from typing import Dict, List, Any, Optional, Tuple, Union

import einops
import torch
from torch import nn

from text_correction_utils import tokenization
from text_correction_utils.modules import grouping


class Embedding(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError


def embedding_from_config(cfg: Dict[str, Any], input_tokenizer: tokenization.Tokenizer) -> Embedding:
    cfg = copy.deepcopy(cfg)
    emb_type = cfg.pop("type")
    if emb_type == "standard":
        return StandardEmbedding(
            num_embeddings=input_tokenizer.vocab_size(),
            pad_token_id=input_tokenizer.pad_token_id(),
            **cfg
        )
    elif emb_type == "byte":
        return ByteEmbedding(
            num_embeddings=input_tokenizer.vocab_size(),
            pad_token_id=input_tokenizer.pad_token_id(),
            **cfg
        )
    else:
        raise ValueError(f"unknown embedding type {emb_type}")


class TokenEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, num_embeddings: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.padding_idx = padding_idx
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.scale = embedding_dim ** 0.5
        nn.init.normal_(self.emb.weight, mean=0, std=embedding_dim ** -0.5)
        if padding_idx is not None:
            nn.init.constant_(self.emb.weight[padding_idx], 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x) * self.scale


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(num_embeddings, embedding_dim, dtype=torch.float, requires_grad=False)
        position = torch.arange(0, num_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float) *
            -(math.log(10_000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_emb", pe)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        pos_emb = torch.index_select(self.pos_emb, 0, positions.reshape(-1))
        return pos_emb.view((*positions.shape, -1))


class LearnedPositionalEmbedding(TokenEmbedding):
    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__(embedding_dim, num_embeddings)


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        positional_embeddings: str,
        embedding_dim: int,
        max_length: int,
        dropout: float
    ):
        super().__init__()
        if positional_embeddings == "learned":
            self.pos_embedding = LearnedPositionalEmbedding(
                embedding_dim,
                max_length
            )
        elif positional_embeddings == "sinusoidal":
            self.pos_embedding = SinusoidalPositionalEmbedding(
                embedding_dim,
                max_length
            )
        else:
            assert positional_embeddings, "positional embeddings must be learned or sinusoidal, " \
                f"but got {positional_embeddings}"

        self.pos_drop = nn.Dropout1d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = einops.repeat(torch.arange(x.shape[1], device=x.device), "s -> b s", b=x.shape[0])
        return self.pos_drop(self.pos_embedding(positions))


class StandardEmbedding(Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        pad_token_id: int,
        dropout: float,
        max_length: Optional[int] = None,
        positional_embeddings: Optional[str] = None,
    ):
        super().__init__()
        self.positional_embeddings = positional_embeddings

        self.embedding = TokenEmbedding(
            embedding_dim,
            num_embeddings,
            pad_token_id
        )

        if positional_embeddings is not None:
            assert max_length is not None, "max length must be specified together with positional embeddings"
            self.pos_embedding = PositionalEmbedding(positional_embeddings, embedding_dim, max_length, dropout)
        else:
            self.pos_embedding = None

        self.token_drop = nn.Dropout1d(dropout)

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        emb = self.token_drop(self.embedding(x))
        if self.pos_embedding is not None:
            pos_emb = self.pos_embedding(x)
        else:
            pos_emb = None
        return emb, pos_emb


class ByteEmbedding(Embedding):
    def __init__(
        self,
        num_embeddings: int,
        groups: str,
        embedding_dim: int,
        pad_token_id: int,
        dropout: float,
        max_length: Optional[int] = None,
        positional_embeddings: Optional[str] = None,
        byte_embedding_dim: Optional[int] = None,
        aggregation: str = "mean"
    ):
        super().__init__()
        assert groups in {"bytes", "code_points"}, "groups must be either bytes or code_points"
        self.groups = groups

        if byte_embedding_dim is None:
            byte_embedding_dim = embedding_dim // 8
        else:
            assert byte_embedding_dim <= embedding_dim, \
                "byte embedding dim must be smaller than or equal to embedding dim"

        self.byte_grouping = grouping.Grouping(aggregation)
        if self.groups == "code_points":
            self.code_point_proj = nn.Linear(byte_embedding_dim, byte_embedding_dim, bias=False)
            self.code_point_grouping = grouping.Grouping(aggregation)

        self.out_proj = nn.Linear(byte_embedding_dim, embedding_dim, bias=False)

        assert num_embeddings >= 256, "number of embeddings must be at least 256, one for each byte, \
all additional embeddings are assumed to be special tokens"

        self.embedding = TokenEmbedding(byte_embedding_dim, num_embeddings, pad_token_id)

        if positional_embeddings is not None:
            assert max_length is not None, "max length must be specified together with positional embeddings"
            self.pos_embedding = PositionalEmbedding(positional_embeddings, embedding_dim, max_length, dropout)
        else:
            self.pos_embedding = None

        self.token_drop = nn.Dropout1d(dropout)

    def forward(
        self,
        x: torch.Tensor,
        byte_groups: Optional[List[List[int]]] = None,
        code_point_groups: Optional[List[List[int]]] = None,
        **_: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        byte_emb = self.embedding(x)

        if self.groups == "code_points":
            assert byte_groups is not None and code_point_groups is not None
            byte_emb, _ = self.byte_grouping(byte_emb, groups=byte_groups)
            code_point_emb = self.code_point_proj(byte_emb)
            emb, _ = self.code_point_grouping(code_point_emb, groups=code_point_groups)
        else:
            assert byte_groups is not None
            emb, _ = self.byte_grouping(byte_emb, groups=byte_groups)

        emb = self.out_proj(emb)
        emb = self.token_drop(emb)
        if self.pos_embedding is not None:
            pos_emb = self.pos_embedding(emb)
        else:
            pos_emb = None
        return emb, pos_emb
