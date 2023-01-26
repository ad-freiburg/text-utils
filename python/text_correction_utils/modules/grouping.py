from typing import Any, List, Tuple, Dict

import torch
from torch import nn
from torch.cuda import amp


class Grouping(nn.Module):
    def __init__(self, aggregation: str = "mean", group_name: str = "groups"):
        super().__init__()
        assert aggregation in {"mean", "sum"}, "aggregation must be either 'mean' or 'sum'"
        if aggregation == "mean":
            self.pow = -1
        else:
            self.pow = 0
        self.group_name = group_name

    def forward(self, feats: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, List[int], Dict[str, Any]]:
        assert feats.ndim == 3, f"feats must have a shape of [B, S, H], but got {feats.shape}"
        assert self.group_name in kwargs, \
            f"expected group {self.group_name} in kwargs, but got {list(kwargs)}"

        (indices, values, size, group_lengths), group_padding_mask = kwargs[self.group_name]
        weights = torch.sparse_coo_tensor(indices, values, size, device=feats.device)
        # overwrite padding mask with new padding mask after grouping
        kwargs["padding_mask"] = group_padding_mask.to(non_blocking=True, device=feats.device)

        with amp.autocast(enabled=False):
            return torch.bmm(weights.float(), feats.float()), group_lengths, kwargs
