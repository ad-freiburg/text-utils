import functools
import math
from typing import Callable, List, Tuple

import torch
from torch import nn


class Grouping(nn.Module):
    """
    >>> t = torch.randn(1, 20, 128)
    >>> g = Grouping("mean")
    >>> groups = [[1] * 20]
    >>> grouped, lengths = g(t, groups)
    >>> tuple(grouped.shape)
    (1, 20, 128)
    >>> lengths
    [20]
    >>> torch.allclose(t, grouped)
    True
    >>> groups = [[3, 8, 1, 8]]
    >>> grouped, lengths = g(t, groups)
    >>> tuple(grouped.shape)
    (1, 4, 128)
    >>> lengths
    [4]
    >>> torch.allclose(t[0, 0:3].mean(0), grouped[0][0])
    True
    >>> torch.allclose(t[0, 3:11].mean(0), grouped[0][1])
    True
    >>> torch.allclose(t[0, 11:12].mean(0), grouped[0][2])
    True
    >>> torch.allclose(t[0, 12:20].mean(0), grouped[0][3])
    True
    """

    def __init__(self, aggregation: str = "mean"):
        super().__init__()
        if aggregation == "mean":
            self.pow = -1
        elif aggregation == "sum":
            self.pow = 1
        else:
            raise ValueError(f"unknown aggregation '{aggregation}', must be either 'mean' or 'sum'")

    def forward(self, feats: torch.Tensor, groups: List[List[int]]) -> Tuple[torch.Tensor, List[int]]:
        assert feats.ndim == 3, f"feats must have a shape of [B, S, H], but got {feats.shape}"
        b, s, _ = feats.shape
        group_lengths = [len(group) for group in groups]
        max_group_length = max(group_lengths)
        # create sparse weight matrix of dense shape [B, max(G), S]
        indices = [[], [], []]
        values = []
        for i, group in enumerate(groups):
            cum_group_length = 0
            for j, g in enumerate(group):
                indices[0].extend([i] * g)
                indices[1].extend([j] * g)
                indices[2].extend(list(range(cum_group_length, cum_group_length + g)))
                values.extend([math.pow(g, self.pow)] * g)
                cum_group_length += g
        weights = torch.sparse_coo_tensor(indices, values, size=(b, max_group_length, s), device=feats.device)
        return torch.bmm(weights, feats), group_lengths
