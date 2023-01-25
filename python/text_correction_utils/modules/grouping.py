from typing import Any, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.cuda import amp


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

    def __init__(self, aggregation: str = "mean", group_name: str = "groups"):
        super().__init__()
        assert aggregation in {"mean", "sum"}, "aggregation must be either 'mean' or 'sum'"
        if aggregation == "mean":
            self.pow = -1
        else:
            self.pow = 0
        self.group_name = group_name

    def forward(self, feats: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, List[int]]:
        assert feats.ndim == 3, f"feats must have a shape of [B, S, H], but got {feats.shape}"
        assert self.group_name in kwargs, \
            f"expected group {self.group_name} in kwargs, but got {list(kwargs)}"

        indices, values, size, group_lengths = kwargs[self.group_name]
        assert isinstance(indices, np.ndarray) and isinstance(values, np.ndarray)
        weights = torch.sparse_coo_tensor(indices, values, size, device=feats.device)

        with amp.autocast(enabled=False):
            return torch.bmm(weights.float(), feats.float()), group_lengths
