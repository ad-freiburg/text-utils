import copy
from typing import Dict, Any

from torch import nn, optim


def optimizer_from_config(
    model: nn.Module,
    cfg: Dict[str, Any]
) -> optim.Optimizer:
    cfg = copy.deepcopy(cfg)
    opt_type = cfg.pop("type")
    if opt_type == "adamw":
        return optim.AdamW(
            model.parameters(),
            **cfg
        )
    elif opt_type == "adam":
        return optim.Adam(
            model.parameters(),
            **cfg
        )
    elif opt_type == "sgd":
        return optim.SGD(
            model.parameters(),
            **cfg
        )
    else:
        raise ValueError(f"Unknown optimizer type {opt_type}")
