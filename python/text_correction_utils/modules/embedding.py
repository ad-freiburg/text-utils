import math
import copy
from typing import Dict, Any, Optional, Tuple

import einops
import torch
from torch import nn

from text_correction_utils import tokenization


class Embedding(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Any
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


class Alibi(nn.Module):
    def __init__(self, heads: int):
        super().__init__()
        self.heads = heads
        self.register_buffer(
            "slopes",
            torch.tensor(self.get_slopes(self.heads)).unsqueeze(-1).unsqueeze(-1) * -1,
            persistent=False
        )
        # mask has shape [b * n, s, s]
        self.mask: Optional[torch.Tensor] = None

    # mostly copied from original implementation at
    # https://github.com/ofirpress/attention_with_linear_biases
    @classmethod
    def get_slopes(cls, n: int):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            # In the paper, we only train models that have 2^a heads for some a. This function has
            return get_slopes_power_of_2(n)
        else:
            # some good properties that only occur when the input is a power of 2. To maintain that even
            # when the number of heads is not a power of 2, we use this workaround.
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + cls.get_slopes(
                2*closest_power_of_2
            )[0::2][:n-closest_power_of_2]

    def get_mask(self, s: int) -> torch.Tensor:
        r = torch.arange(s)
        rel_pos = r[None, :] - r[:, None]
        rel_pos = einops.repeat(torch.abs(rel_pos), "s t -> n s t", n=self.heads)
        return rel_pos.to(non_blocking=True, device=self.slopes.device) * self.slopes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s = x.shape[:2]

        if self.mask is None or s > self.mask.shape[1]:
            self.mask = self.get_mask(s)

        return einops.repeat(self.mask[:, :s, :s], "n s t -> (b n) s t", b=b)


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
        positions = einops.repeat(
            torch.arange(x.shape[1], device=x.device),
            "s -> b s",
            b=x.shape[0]
        )
        return self.pos_drop(self.pos_embedding(positions))


class StandardEmbedding(Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        pad_token_id: int,
        dropout: float,
        token_dropout: float = 0.0,
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

        self.drop = nn.Dropout(dropout)
        self.token_drop = nn.Dropout1d(token_dropout)

    def forward(
        self,
        x: torch.Tensor,
        **_: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        emb = self.token_drop(self.drop(self.embedding(x)))
        if self.pos_embedding is not None:
            pos_emb = self.token_drop(self.drop(self.pos_embedding(x)))
        else:
            pos_emb = None
        return emb, pos_emb
