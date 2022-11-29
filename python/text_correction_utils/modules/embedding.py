import math
from typing import Optional, Tuple

import torch
from torch import nn


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
        pos_emb = torch.index_select(self.pe, 0, positions.reshape(-1))
        return pos_emb.view((*positions.shape, -1))


class LearnedPositionalEmbedding(TokenEmbedding):
    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__(embedding_dim, num_embeddings)


class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            pad_token_id: int,
            dropout: float,
            positional_embeddings: str,
            max_length: int
    ):
        super().__init__()
        self.positional_embeddings = positional_embeddings

        self.embedding = TokenEmbedding(
            embedding_dim,
            num_embeddings,
            pad_token_id
        )
        nn.init.normal_(self.embedding.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(self.embedding.weight[pad_token_id], 0)

        if self.positional_embeddings == "learned":
            self.pos_embedding = LearnedPositionalEmbedding(
                embedding_dim,
                max_length
            )
        elif self.positional_embeddings == "sinusoidal":
            self.pos_embedding = SinusoidalPositionalEmbedding(
                embedding_dim,
                max_length
            )
        else:
            assert self.positional_embeddings, "positional embeddings must be one of learned, sinusoidal, or none, " \
                                               f"but got {self.positional_embeddings}"

        self.token_drop = nn.Dropout1d(dropout)
        self.pos_drop = nn.Dropout1d(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        emb = self.token_drop(self.embedding(x))
        if self.positional_embeddings != "none":
            positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            pos_emb = self.pos_drop(self.pos_embedding(positions))
        else:
            pos_emb = None
        return emb, pos_emb
