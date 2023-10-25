import copy
import math
from functools import reduce
from typing import Union, List, Any, Dict, Callable, Optional

from torch import optim


def _warmup(optimizer: optim.Optimizer, num_steps: int) -> optim.lr_scheduler.LinearLR:
    return optim.lr_scheduler.LinearLR(optimizer, 1e-4, 1, num_steps)


def _check_warmup_steps(
    warmup_steps: Union[float, int],
    train_steps: int,
    world_size: int,
) -> int:
    if isinstance(warmup_steps, float):
        warmup_steps = round(train_steps * warmup_steps)
    elif isinstance(warmup_steps, int):
        warmup_steps //= world_size
    else:
        raise TypeError(
            f"expected warmup steps to be either float or int, but got {type(warmup_steps)}"
        )
    assert 0 <= warmup_steps <= train_steps, \
        f"warmup steps should be larger or equal zero and smaller or equal to training steps, " \
        f"but got {warmup_steps} and {train_steps} training steps"
    return warmup_steps


def linear_with_warmup(
    optimizer: optim.Optimizer,
    training_steps: int,
    world_size: int,
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
    warmup_steps = _check_warmup_steps(warmup_steps, training_steps, world_size)
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
    world_size: int,
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
    warmup_steps = _check_warmup_steps(warmup_steps, training_steps, world_size)

    def _cosine(step: int) -> float:
        frac = step / max(1.0, training_steps - warmup_steps)
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
    world_size: int,
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
    warmup_steps = _check_warmup_steps(warmup_steps, training_steps, world_size)
    assert len(steps) == len(factors), "expected a factor for every step"
    milestones = []
    for step in steps:
        if isinstance(step, int):
            milestones.append(step - warmup_steps)
        else:
            milestones.append(training_steps * step - warmup_steps)
    assert milestones == sorted(milestones), "steps must be given in sorted order"
    assert all(warmup_steps <= step <= training_steps for step in milestones), \
        "each step must lie between the number of warmup steps and the number of total training steps"

    def _multi_step(step: int) -> float:
        idx = 0
        for step_at in milestones:
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
    world_size: int,
    warmup_steps: Union[float, int],
) -> optim.lr_scheduler.SequentialLR:
    """

    Lr scheduler that warms up linearly, then keeps the learning rate constant

    :param optimizer: optimizer instance
    :param training_steps: number of training steps
    :param warmup_steps: number of warmup steps
    :return: lr scheduler
    """
    return multi_step_with_warmup(optimizer, training_steps, world_size, warmup_steps, [], [])


def cont_sqrt_with_warmup(
    optimizer: optim.Optimizer,
    training_steps: int,
    world_size: int,
    warmup_steps: Union[float, int],
) -> optim.lr_scheduler.SequentialLR:
    """

    Lr scheduler that warms up linearly, then decays the
    learning rate using a square root schedule on the
    ratio of warmup steps to current step

    :param optimizer: optimizer instance
    :param warmup_steps: number of warmup steps
    :return: lr scheduler
    """
    warmup_steps = _check_warmup_steps(warmup_steps, training_steps, world_size)

    def _sqrt(step: int) -> float:
        return math.sqrt(warmup_steps / max(1, step + warmup_steps))

    return optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            _warmup(optimizer, warmup_steps),
            optim.lr_scheduler.LambdaLR(optimizer, _sqrt)
        ],
        [warmup_steps]
    )


def lr_scheduler_from_config(
    optimizer: optim.Optimizer,
    steps: int,
    world_size: int,
    cfg: Dict[str, Any],
    additional_lr_scheduler_fn: Optional[Callable[
        [optim.Optimizer, int, int, Dict[str, Any]],
        optim.lr_scheduler.SequentialLR
    ]] = None
) -> optim.lr_scheduler.SequentialLR:
    cfg = copy.deepcopy(cfg)
    lr_type = cfg.pop("type")
    if lr_type == "linear_with_warmup":
        return linear_with_warmup(optimizer, steps, world_size, **cfg)
    elif lr_type == "cosine_with_warmup":
        return cosine_with_warmup(optimizer, steps, world_size, **cfg)
    elif lr_type == "multi_step_with_warmup":
        return multi_step_with_warmup(optimizer, steps, world_size, **cfg)
    elif lr_type == "constant_with_warmup":
        return constant_with_warmup(optimizer, steps, world_size, **cfg)
    elif lr_type == "cont_sqrt_with_warmup":
        return cont_sqrt_with_warmup(optimizer, steps, world_size, **cfg)
    else:
        if additional_lr_scheduler_fn is not None:
            return additional_lr_scheduler_fn(optimizer, steps, world_size, cfg)
        raise ValueError(f"unknown lr scheduler type {lr_type}")


def multi_step(
    training_items: int,
    max_length: int,
    steps: List[float],
    factors: List[float]
) -> Callable[[int], int]:
    assert steps == sorted(steps), f"steps must be given in sorted order, got {steps}"
    assert len(steps) == len(factors), f"expected a factor for every step, got {factors}"
    steps = [step_at * training_items for step_at in steps]

    def _multi_step(seen_items: int) -> int:
        idx = 0
        for step_at in steps:
            if seen_items >= step_at:
                idx += 1
        return round(max_length * reduce(lambda a, b: a * b, factors[:idx], 1.0))

    return _multi_step


def max_length_scheduler_from_config(
    training_items: int,
    max_length: int,
    cfg: Dict[str, Any],
    additional_max_length_scheduler_fn: Optional[Callable[
        [int, Dict[str, Any]],
        Callable[[int], int]
    ]] = None
) -> Callable[[int], int]:
    cfg = copy.deepcopy(cfg)
    mls_type = cfg.pop("type")
    if mls_type == "multi_step":
        return multi_step(training_items, max_length, **cfg)
    else:
        if additional_max_length_scheduler_fn is not None:
            return additional_max_length_scheduler_fn(training_items, cfg)
        raise ValueError(f"unknown lr scheduler type {mls_type}")
