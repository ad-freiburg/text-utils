from typing import Any, Tuple, Dict

import torch
from torch import nn
from torch.cuda import amp


class Grouping(nn.Module):
    def __init__(
        self,
        aggregation: str = "mean",
        group_name: str = "groups",
        group_lengths: str = "group_lengths",
        group_padding_mask: str = "group_padding_mask"
    ):
        super().__init__()
        assert aggregation in {"mean", "sum"}, "aggregation must be either 'mean' or 'sum'"
        if aggregation == "mean":
            self.pow = -1
        else:
            self.pow = 0
        self.group_name = group_name
        self.group_lengths = group_lengths
        self.group_padding_mask = group_padding_mask

    def forward(self, feats: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        assert feats.ndim == 3, f"feats must have a shape of [B, S, H], but got {feats.shape}"
        assert self.group_name in kwargs, \
            f"expected group {self.group_name} in kwargs, but got {list(kwargs)}"

        (indices, values, size, lengths), padding_mask = kwargs[self.group_name]
        weights = torch.sparse_coo_tensor(indices, values, size, device=feats.device)
        kwargs[self.group_lengths] = lengths
        kwargs[self.group_padding_mask] = padding_mask

        with amp.autocast(enabled=False):
            return torch.bmm(weights.float(), feats.float()), kwargs
