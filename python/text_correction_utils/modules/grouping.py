import functools
from typing import Callable, List, Tuple

import torch
from torch import nn


def _aggregation_fn(aggregation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if aggregation == "mean":
        return functools.partial(torch.mean, dim=1)
    elif aggregation == "sum":
        return functools.partial(torch.sum, dim=1)
    else:
        raise ValueError(f"aggregation must be mean or sum, but got {aggregation}")


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
        self.agg_fn = _aggregation_fn(aggregation)

    def forward(self, feats: torch.Tensor, groups: List[List[int]]) -> Tuple[torch.Tensor, List[int]]:
        assert feats.ndim == 3, f"feats must have a shape of [B, S, H], but got {feats.shape}"
        b, s, h = feats.shape
        group_lengths = [len(group) for group in groups]
        max_group_length = max(group_lengths)
        agg_feats = torch.zeros(b, max_group_length, h, dtype=feats.dtype, device=feats.device)
        for i, group in enumerate(groups):
            if len(group) == 0:
                continue

            start = 0
            end = 1
            group_start = 0
            total_group_length = group[0]
            while end < len(group):
                if group[start] == group[end]:
                    total_group_length += group[end]
                    end += 1
                    continue

                agg_feats[i, start:end] = self.agg_fn(
                    feats[i, group_start:group_start + total_group_length].view(
                        total_group_length // group[start], group[start], -1
                    )
                )
                start = end
                group_start = group_start + total_group_length
                total_group_length = 0

            agg_feats[i, start:end] = self.agg_fn(
                feats[i, group_start:group_start + total_group_length].view(
                    total_group_length // group[start], group[start], -1
                )
            )

        return agg_feats, group_lengths
