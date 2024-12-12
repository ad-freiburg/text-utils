import copy
from typing import Dict, Any, Iterator, Tuple, Optional, Callable

from torch import nn, optim


def _select_params_and_modules(
    modules: Iterator[Tuple[str, nn.Module]], prefix: str
) -> Iterator[Tuple[str, nn.Module, nn.Parameter]]:
    for name, mod in modules:
        for p_name, param in mod.named_parameters(prefix=name, recurse=False):
            if p_name.startswith(prefix):
                yield p_name, mod, param


def optimizer_from_config(
    model: nn.Module,
    cfg: Dict[str, Any],
    additional_optimizer_fn: Optional[
        Callable[[nn.Module, Dict[str, Any]], optim.Optimizer]
    ] = None,
) -> optim.Optimizer:
    cfg = copy.deepcopy(cfg)
    opt_type = cfg.pop("type")

    if opt_type == "adamw":
        optim_cls = optim.AdamW
    elif opt_type == "adam":
        optim_cls = optim.Adam
    elif opt_type == "sgd":
        optim_cls = optim.SGD
    else:
        if additional_optimizer_fn is not None:
            return additional_optimizer_fn(model, cfg)
        raise ValueError(f"unknown optimizer type {opt_type}")

    param_groups: list[dict[str, Any]] = cfg.pop("param_groups", [{"prefix": None}])
    assert len(param_groups) > 0, "param_groups must be non-empty"

    weight_decay_modules: dict[str, list[str]] | str = cfg.pop(
        "weight_decay_modules", "all"
    )
    all = set(name for name, p in model.named_parameters() if p.requires_grad)
    params = []
    names = set()
    for group in param_groups:
        prefix = group.pop("prefix") or ""
        fix = group.pop("fix", False)

        no_decay = set()
        decay = set()
        param_dict = {}
        for name, mod, param in _select_params_and_modules(
            model.named_modules(), prefix
        ):
            if name not in all:
                # this should only happen for shared
                # or non-trainable parameters
                continue

            if fix:
                param.requires_grad = False
                continue

            names.add(name)
            param_dict[name] = param
            mod_name = mod.__class__.__name__
            if weight_decay_modules == "all" or (
                isinstance(weight_decay_modules, dict)
                and mod_name in weight_decay_modules
                and any(
                    name.endswith(suffix) for suffix in weight_decay_modules[mod_name]
                )
            ):
                decay.add(name)
            else:
                no_decay.add(name)

        assert len(decay & no_decay) == 0
        assert len(param_dict.keys() - (decay | no_decay)) == 0

        if len(decay) > 0:
            params.append(
                {
                    "params": [param_dict[name] for name in sorted(list(decay))],
                    **(cfg | group),
                }
            )
        if len(no_decay) > 0:
            params.append(
                {
                    "params": [param_dict[name] for name in sorted(list(no_decay))],
                    **(cfg | group | {"weight_decay": 0.0}),
                }
            )

    unused = all - names
    assert (
        len(unused) == 0
    ), f"parameter groups dont match trainable model parameters: {unused}"

    return optim_cls(params, **cfg)
