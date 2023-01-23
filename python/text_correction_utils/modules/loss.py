import copy
import math
from typing import Dict, Any, List, Optional, Callable

import einops
import torch
from torch import nn


def _loss_schedule(training_steps: int, schedule_type: str) -> Callable[[int], float]:
    if schedule_type == "linear":
        return lambda step: max(0.0, 1.0 - step / training_steps)
    elif schedule_type == "cosine":
        def _cosine(step: int):
            frac = min(1.0, step / training_steps)
            return 0.5 * (1.0 + math.cos(math.pi * frac))
        return _cosine
    else:
        raise ValueError(f"unknown schedule type {schedule_type}")


class FocalLoss(nn.Module):
    # copied and modified from https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
    def __init__(
        self,
        alpha: Optional[List[float]],
        gamma: float,
        reduction: str = "mean",
        ignore_index: int = -100,
        gamma_schedule: Optional[Callable[[int], float]] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.init_gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.nll_loss = nn.NLLLoss(
            weight=torch.as_tensor(alpha, dtype=torch.float) if alpha is not None else None,
            reduction="none",
            ignore_index=ignore_index
        )
        self.gamma_schedule = gamma_schedule
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert outputs.ndim == 2 and labels.ndim == 1
        unignored_mask = labels != self.ignore_index
        labels = labels[unignored_mask]
        if len(labels) == 0:
            return torch.tensor(0, device=outputs.device, dtype=torch.float)
        outputs = outputs[unignored_mask]

        log_p = torch.log_softmax(outputs, dim=-1)
        ce = self.nll_loss(log_p, labels)

        log_pt = log_p[torch.arange(len(outputs), device=outputs.device), labels]
        pt = log_pt.exp()
        gamma = self.init_gamma
        if self.gamma_schedule is not None:
            gamma *= self.gamma_schedule(self._step.item())
        focal_term = torch.pow((1 - pt).clamp(0, 1), gamma)
        ce = focal_term * ce

        if self.reduction == "mean":
            ce = ce.mean()
        elif self.reduction == "sum":
            ce = ce.sum()
        return ce

    def step(self):
        self._step += 1


class SeqLoss(nn.Module):
    """
    Wrapper class for sequence losses. Rearranges outputs and labels to use with standard Pytorch losses.
    """

    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # outputs are expected to be of shape [B, S, C], reshape to [B * S, C]
        outputs = einops.rearrange(outputs, "b s c -> (b s) c")
        # labels are expected to be of shape [B, S], reshape to [B * S]
        labels = einops.rearrange(labels, "b s -> (b s)")
        return self.loss(outputs, labels)

    def step(self):
        if hasattr(self.loss, "step"):
            self.loss.step()


def loss_from_config(
    training_steps: int,
    cfg: Dict[str, Any],
) -> nn.Module:
    cfg = copy.deepcopy(cfg)
    loss_type = cfg.pop("type")
    if loss_type == "sequence_cross_entropy":
        cfg["type"] = "cross_entropy"
        loss = loss_from_config(training_steps, cfg)
        return SeqLoss(loss=loss)

    elif loss_type == "cross_entropy":
        weight = cfg.get("weights", None)
        weight = torch.tensor(weight, dtype=torch.float) if weight is not None else None
        loss = nn.CrossEntropyLoss(ignore_index=cfg.get("ignore_index", -1), weight=weight)
        return loss

    elif loss_type == "binary_cross_entropy":
        weight = cfg.get("weight", None)
        weight = torch.tensor(weight, dtype=torch.float) if weight is not None else None
        loss = nn.BCELoss(weight=weight)
        return loss

    elif loss_type == "focal":
        weight = cfg.get("weight", None)
        if "gamma_schedule" in cfg:
            schedule = _loss_schedule(training_steps, cfg["gamma_schedule"])
        else:
            schedule = None
        loss = FocalLoss(
            weight,
            gamma=cfg.get("gamma", 2.),
            ignore_index=cfg.get("ignore_index", -1),
            gamma_schedule=schedule
        )
        return loss

    elif loss_type == "sequence_focal":
        cfg["type"] = "focal"
        loss = loss_from_config(training_steps, cfg)
        return SeqLoss(loss=loss)

    else:
        raise ValueError(f"unknown loss type {loss_type}")
