import copy
from typing import Dict, Any, Iterator, Tuple, List

from torch import nn, optim


def _select_params(
    params: Iterator[Tuple[str, nn.Parameter]],
    prefixes: List[str],
    include: bool
) -> Iterator[nn.Parameter]:
    for name, param in params:
        if include and any(name.startswith(prefix) for prefix in prefixes):
            yield param
        elif not include and all(not name.startswith(prefix) for prefix in prefixes):
            yield param


def optimizer_from_config(
    model: nn.Module,
    cfg: Dict[str, Any]
) -> optim.Optimizer:
    cfg = copy.deepcopy(cfg)
    opt_type = cfg.pop("type")
    param_groups = cfg.pop("param_groups", None)
    params = []
    exclude_prefixes = []
    if param_groups is not None:
        params = []
        for group in param_groups:
            prefix = group.pop("prefix")
            exclude_prefixes.append(prefix)
            group_params = _select_params(model.named_parameters(), [prefix], True)
            if group.pop("fix", False):
                for param in group_params:
                    param.requires_grad = False
                continue
            params.append({
                "params": group_params,
                **group
            })

    params.append({
        "params": _select_params(model.named_parameters(), exclude_prefixes, False),
        **cfg
    })

    if opt_type == "adamw":
        return optim.AdamW(
            params,
            **cfg
        )
    elif opt_type == "adam":
        return optim.Adam(
            params,
            **cfg
        )
    elif opt_type == "sgd":
        return optim.SGD(
            params,
            **cfg
        )
    else:
        raise ValueError(f"unknown optimizer type {opt_type}")
