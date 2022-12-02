import logging
from typing import Union

from torch.utils.tensorboard import SummaryWriter


class AverageTracker:
    def __init__(self, name: str, fmt: str = ".2f"):
        self.name = name
        self.values = []
        self.fmt = fmt

    def add(self, v: Union[float, int]):
        self.values.append(v)

    @property
    def value(self) -> float:
        return sum(self.values) / max(len(self.values), 1)

    def log_tensorboard(self, writer: SummaryWriter, step: int):
        writer.add_scalar(
            self.name,
            self.value,
            step
        )

    def log_info(self, logger: logging.Logger, step: int):
        logger.info(f"[step {step}] {self.name} = {self.value:{self.fmt}}")

    def reset(self):
        self.values.clear()
