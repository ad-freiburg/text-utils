import copy
from typing import Dict, Any, Iterator, Tuple, List, Optional, Callable

from torch import nn, optim
try:
    from bitsandbytes import optim as optim_8bit
    _8BIT_OPTIMIZERS = True
except ImportError:
    _8BIT_OPTIMIZERS = False


def _select_params_and_modules(
    modules: Iterator[Tuple[str, nn.Module]],
    prefixes: List[str],
) -> Iterator[Tuple[str, nn.Module, nn.Parameter]]:
    for name, mod in modules:
        for p_name, param in mod.named_parameters(prefix=name, recurse=False):
            if any(p_name.startswith(prefix) for prefix in prefixes):
                yield p_name, mod, param


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
    param_groups = cfg.pop("param_groups", [{"prefix": ""}])

    all = set(name for name, p in model.named_parameters() if p.requires_grad)
    params = []
    names = set()
    for group in param_groups:
        prefix = group.pop("prefix")
        fix = group.pop("fix", False)
        no_decay = set()
        decay = set()
        param_dict = {}
        for name, mod, param in _select_params_and_modules(model.named_modules(), [prefix]):
            if name not in all:
                # this should only happen for shared parameters
                continue
            if fix:
                param.requires_grad = False
                continue
            names.add(name)
            param_dict[name] = param
            if isinstance(mod, nn.Linear) and name.endswith("weight"):
                decay.add(name)
            else:
                no_decay.add(name)

        assert len(decay & no_decay) == 0
        assert len(param_dict.keys() - (decay | no_decay)) == 0

        if len(decay) > 0:
            params.append({
                "params": [param_dict[name] for name in sorted(list(decay))],
                **(cfg | group)
            })
        if len(no_decay) > 0:
            params.append({
                "params": [param_dict[name] for name in sorted(list(no_decay))],
                **(cfg | group | {"weight_decay": 0.0})
            })

    unused = all - names
    assert len(unused) == 0, \
        f"parameter groups dont match trainable model parameters: {unused}"

    optim_bits = int(cfg.get("optim_bits", 32))
    assert optim_bits in [32, 8], f"optim_bits must be 32 or 8, got {optim_bits}"
    use_8bit = optim_bits == 8
    if use_8bit:
        assert _8BIT_OPTIMIZERS, "8-bit optimizers not available"
    else:
        cfg.pop("optim_bits", None)

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
