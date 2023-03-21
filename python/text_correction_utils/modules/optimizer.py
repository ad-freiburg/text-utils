import copy
from typing import Dict, Any, Iterator, Tuple, List, Optional, Callable

from torch import nn, optim
try:
    from bitsandbytes import optim as optim_8bit
    _8BIT_OPTIMIZERS = True
except ImportError:
    _8BIT_OPTIMIZERS = False


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
    cfg: Dict[str, Any],
    additional_optimizer_fn: Optional[Callable[
        [nn.Module, Dict[str, Any]],
        optim.Optimizer
    ]] = None
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

    optim_bits = int(cfg.get("optim_bits", 32))
    assert optim_bits in [32, 8], f"optim_bits must be 32 or 8, got {optim_bits}"
    use_8bit = optim_bits == 8
    if use_8bit:
        assert _8BIT_OPTIMIZERS, "8-bit optimizers not available"

    if opt_type == "adamw":
        optim_cls = optim_8bit.AdamW if use_8bit else optim.AdamW
    elif opt_type == "adam":
        optim_cls = optim_8bit.Adam if use_8bit else optim.Adam
    elif opt_type == "sgd":
        optim_cls = optim_8bit.SGD if use_8bit else optim.SGD
    else:
        if additional_optimizer_fn is not None:
            return additional_optimizer_fn(model, cfg)
        raise ValueError(f"unknown optimizer type {opt_type}")

    return optim_cls(params, **cfg)
