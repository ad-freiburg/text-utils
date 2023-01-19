import logging
from typing import Optional, Union, Dict, Any, List
from typing_extensions import override

import torch
from torch.utils.tensorboard import SummaryWriter
from text_correction_utils import data, whitespace, tokenization


class TensorboardLogger:
    def log_tensorboard(self, writer: SummaryWriter, step: int):
        raise NotImplementedError

    def log_info(self, logger: logging.Logger, step: int):
        raise NotImplementedError


class AverageTracker(TensorboardLogger):
    def __init__(self, name: str, fmt: str = ".2f"):
        self.name = name
        self.values = []
        self.fmt = fmt

    def add(self, v: Union[float, int]):
        self.values.append(v)

    @property
    def value(self) -> float:
        return sum(self.values) / max(len(self.values), 1)

    @override
    def log_tensorboard(self, writer: SummaryWriter, step: int):
        writer.add_scalar(
            self.name,
            self.value,
            step
        )

    @override
    def log_info(self, logger: logging.Logger, step: int):
        logger.info(f"[step {step}] {self.name} = {self.value:{self.fmt}}")

    def reset(self):
        self.values.clear()


class TensorboardMetric(TensorboardLogger):
    def set_values(self, items: List[data.Item], outputs: torch.Tensor):
        raise NotImplementedError


class WhitespaceCorrectionMetric(TensorboardMetric):
    def __init__(self, name: str, input_tokenizer: tokenization.Tokenizer, max_items: Optional[int] = None):
        self.items = None
        self.outputs = None
        self.max_items = max_items
        self.name = name
        self.input_tokenizer = input_tokenizer

    @override
    def set_values(self, items: List[data.Item], outputs: torch.Tensor):
        self.items = items
        self.outputs = torch.argmax(outputs, dim=-1).tolist()

    def _get_string(self) -> str:
        assert self.items is not None and self.outputs is not None, "call set_values before logging"
        strings = []
        for item, output in zip(self.items, self.outputs):
            if self.max_items is not None and len(strings) >= self.max_items:
                break

            start = self.input_tokenizer.num_prefix_tokens()
            end = len(item.tokenization.token_ids) - self.input_tokenizer.num_suffix_tokens()
            if "token_groups" == item.tokenization.info["type"]:
                if "code_point_groups" in item.tokenization.info:
                    group_name = "code_point_groups"
                else:
                    group_name = "byte_groups"
                end = len(item.tokenization.info[group_name]["groups"][-1]) - self.input_tokenizer.num_suffix_tokens()

            repair_ops = []
            for pred in output[start:end]:
                if pred == 0:
                    repair_ops.append("k")
                elif pred == 1:
                    repair_ops.append("i")
                elif pred == 2:
                    repair_ops.append("d")
                else:
                    raise RuntimeError(f"expected repair tokens to be either 0, 1, or 2, but got {pred}")

            target_ops = whitespace.operations(item.data.processed, item.data.original)
            repaired = whitespace.repair(item.data.processed, repair_ops)
            strings.append(
                "\n".join([
                    f"Input      : {item.data.processed}\n",
                    f"Target     : {item.data.original}\n",
                    f"Target pred: {''.join(target_ops)}\n",
                    f"Prediction : {''.join(repair_ops)}\n",
                    f"Repaired   : {repaired}"
                ])
            )

        return ("\n" + "-" * 80 + "\n").join(strings) + "\n"

    @override
    def log_tensorboard(self, writer: SummaryWriter, step: int):
        s = self._get_string()
        writer.add_text(self.name, s, step)

    @override
    def log_info(self, logger: logging.Logger, step: int):
        s = self._get_string()
        logger.info(f"[step {step}] {self.name}\n{s}")


def metrics_from_config(
    cfg: Dict[str, Any],
    input_tokenizer: tokenization.Tokenizer,
    output_tokenizer: tokenization.Tokenizer,
    prefix: Optional[str] = None
) -> List[TensorboardMetric]:
    metrics = []
    if prefix is not None and not prefix.endswith("_"):
        prefix += "_"
    for metric_type, metric_opts in cfg.items():
        if metric_type == "whitespace_correction":
            metric = WhitespaceCorrectionMetric(f"{prefix}whitespace_correction", input_tokenizer, **metric_opts)
        else:
            raise ValueError(f"unknown metric type {metric_type}")
        metrics.append(metric)
    return metrics
