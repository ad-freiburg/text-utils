import logging

import torch
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from text_utils import data


class TensorboardLogger:
    def log_tensorboard(self, writer: SummaryWriter, step: int):
        raise NotImplementedError

    def log_info(self, logger: logging.Logger, step: int):
        raise NotImplementedError


class DistAverageTracker(TensorboardLogger):
    def __init__(
        self,
        name: str,
        device: torch.device,
        fmt: str = ".2f",
        val_reduce_op: dist.ReduceOp = dist.ReduceOp.SUM,
        count_reduce_op: dist.ReduceOp = dist.ReduceOp.SUM,
        output_op: str = "mean"
    ):
        assert output_op in ["mean", "sum"]
        self.name = name
        self.val_tensor = torch.zeros(1, dtype=torch.float, device=device)
        self.count_tensor = torch.zeros(1, dtype=torch.float, device=device)
        self.fmt = fmt
        self.val_reduce_op = val_reduce_op
        self.count_reduce_op = count_reduce_op
        self.output_op = output_op
        self.synced = False

    def add(self, v: float | int, count: int = 1):
        self.val_tensor[0] += v
        self.count_tensor[0] += count

    def sync(self):
        dist.all_reduce(self.val_tensor, op=self.val_reduce_op)
        dist.all_reduce(self.count_tensor, op=self.count_reduce_op)
        self._synced = True

    @property
    def value(self) -> float:
        assert self._synced, "call sync() before accessing value"
        if self.output_op == "mean":
            return self.val_tensor.item() / max(1, self.count_tensor.item())
        else:
            return self.val_tensor.item()

    def log_tensorboard(self, writer: SummaryWriter, step: int):
        writer.add_scalar(
            self.name,
            self.value,
            step
        )

    def log_info(self, logger: logging.Logger, step: int):
        logger.info(f"[step {step}] {self.name} = {self.value:{self.fmt}}")

    def reset(self):
        self.val_tensor[0] = 0
        self.count_tensor[0] = 0
        self._synced = False


class TensorboardMetric(TensorboardLogger):
    def set_values(self, items: list[data.TrainItem], outputs: torch.Tensor):
        raise NotImplementedError
