import copy
import math
from functools import reduce
from typing import Union, List, Any, Dict

from torch import optim


def _warmup(optimizer: optim.Optimizer, num_steps: int) -> optim.lr_scheduler.LinearLR:
    return optim.lr_scheduler.LinearLR(optimizer, 1e-4, 1, num_steps)


def _check_warmup_steps(warmup_steps: Union[float, int], train_steps: int) -> int:
    if isinstance(warmup_steps, float):
        warmup_steps = round(train_steps * warmup_steps)
    assert 0 <= warmup_steps <= train_steps, \
        f"warmup steps should be larger or equal zero and smaller or equal to training steps, " \
        f"but got {warmup_steps} and {train_steps} training steps"
    return warmup_steps


def linear_with_warmup(
    optimizer: optim.Optimizer,
    training_steps: int,
    warmup_steps: Union[float, int]
) -> optim.lr_scheduler.SequentialLR:
    """

    Lr scheduler that warms up linearly, then decays the
    learning rate linearly

    :param optimizer: optimizer instance
    :param training_steps: number of training steps
    :param warmup_steps: number of warmup steps
    :return: lr scheduler
    """
    warmup_steps = _check_warmup_steps(warmup_steps, training_steps)
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            _warmup(optimizer, warmup_steps),
            optim.lr_scheduler.LinearLR(optimizer, 1, 0, training_steps - warmup_steps)
        ],
        [warmup_steps]
    )


def cosine_with_warmup(
    optimizer: optim.Optimizer,
    training_steps: int,
    warmup_steps: Union[float, int]
) -> optim.lr_scheduler.SequentialLR:
    """

    Lr scheduler that warms up linearly, then decays the
    learning rate using a cosine schedule

    :param optimizer: optimizer instance
    :param training_steps: number of training steps
    :param warmup_steps: number of warmup steps
    :return: lr scheduler
    """
    warmup_steps = _check_warmup_steps(warmup_steps, training_steps)

    def _cosine(step: int) -> float:
        frac = (step - warmup_steps) / max(1.0, training_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * frac)))

    return optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            _warmup(optimizer, warmup_steps),
            optim.lr_scheduler.LambdaLR(optimizer, _cosine)
        ],
        [warmup_steps]
    )


def multi_step_with_warmup(
    optimizer: optim.Optimizer,
    training_steps: int,
    warmup_steps: Union[float, int],
    steps: List[Union[float, int]],
    factors: List[float]
) -> optim.lr_scheduler.SequentialLR:
    """

    Lr scheduler that warms up linearly, then steps the
    learning rate at the given stepping points with the given factors

    :param optimizer: optimizer instance
    :param training_steps: number of training steps
    :param warmup_steps: number of warmup steps
    :param steps: list of stepping points, either as absolute or relative values
    :param factors: list of stepping factors
    :return: lr scheduler
    """
    warmup_steps = _check_warmup_steps(warmup_steps, training_steps)
    assert len(steps) == len(factors), "expected a factor for every step"
    steps = [
        step if isinstance(step, int) else round((training_steps - warmup_steps) * step) + warmup_steps
        for step in steps
    ]
    assert steps == sorted(steps), "steps must be given in sorted order"
    assert all(warmup_steps <= step <= training_steps for step in steps), \
        "each step must lie between the number of warmup steps and the number of total training steps"

    def _multi_step(step: int) -> float:
        idx = 0
        for step_at in steps:
            if step >= step_at:
                idx += 1

        return reduce(lambda a, b: a * b, factors[:idx], 1)

    return optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            _warmup(optimizer, warmup_steps),
            optim.lr_scheduler.LambdaLR(optimizer, _multi_step)
        ],
        [warmup_steps]
    )


def constant_with_warmup(
    optimizer: optim.Optimizer,
    training_steps: int,
    warmup_steps: Union[float, int],
) -> optim.lr_scheduler.SequentialLR:
    """

    Lr scheduler that warms up linearly, then keeps the learning rate constant

    :param optimizer: optimizer instance
    :param training_steps: number of training steps
    :param warmup_steps: number of warmup steps
    :return: lr scheduler
    """
    return multi_step_with_warmup(optimizer, training_steps, warmup_steps, [], [])


def lr_scheduler_from_config(
    optimizer: optim.Optimizer,
    training_steps: int,
    cfg: Dict[str, Any]
) -> optim.lr_scheduler.SequentialLR:
    cfg = copy.deepcopy(cfg)
    lr_type = cfg.pop("type")
    if lr_type == "linear_with_warmup":
        return linear_with_warmup(optimizer, training_steps, **cfg)
    elif lr_type == "cosine_with_warmup":
        return cosine_with_warmup(optimizer, training_steps, **cfg)
    elif lr_type == "multi_step_with_warmup":
        return multi_step_with_warmup(optimizer, training_steps, **cfg)
    elif lr_type == "constant_with_warmup":
        return constant_with_warmup(optimizer, training_steps, **cfg)
    else:
        raise ValueError(f"unknown lr scheduler type {lr_type}")
