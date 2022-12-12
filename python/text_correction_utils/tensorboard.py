import logging
from typing import Optional, Union
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
    def set_values(self, batch: data.ItemBatch, outputs: torch.Tensor):
        raise NotImplementedError


class WhitespaceCorrectionMetric(TensorboardMetric):
    def __init__(self, name: str, input_tokenizer: tokenization.Tokenizer, max_items: Optional[int] = None):
        self.batch = None
        self.outputs = None
        self.max_items = max_items
        self.name = name
        self.input_tokenizer = input_tokenizer

    @override
    def set_values(self, batch: data.ItemBatch, outputs: torch.Tensor):
        self.batch = batch
        self.outputs = torch.argmax(outputs, dim=-1).tolist()

    def _get_string(self) -> str:
        assert self.batch is not None and self.outputs is not None, "call set_values before logging"
        strings = []
        for item, output in zip(self.batch.items, self.outputs):
            if self.max_items is not None and len(strings) >= self.max_items:
                break

            start = self.input_tokenizer.num_prefix_tokens()
            end = len(item.tokenization.token_ids) - self.input_tokenizer.num_suffix_tokens()
            if "token_groups" == item.tokenization.info["type"]:
                end = len(item.tokenization.info["groups"]) - self.input_tokenizer.num_suffix_tokens()

            pred_ops = []
            repair_ops = []
            for pred in output[start:end]:
                if pred == 0:
                    repair_ops.append("K")
                    pred_ops.append(whitespace.WhitespaceOperation.Keep)
                elif pred == 1:
                    repair_ops.append("I")
                    pred_ops.append(whitespace.WhitespaceOperation.Insert)
                elif pred == 2:
                    repair_ops.append("D")
                    pred_ops.append(whitespace.WhitespaceOperation.Delete)
                else:
                    raise RuntimeError(f"expected repair tokens to be either 0, 1, or 2, but got {pred}")
            target_ops = []
            for op in whitespace.operations(item.data.processed, item.data.original):
                if op == whitespace.WhitespaceOperation.Keep:
                    target_ops.append("K")
                elif op == whitespace.WhitespaceOperation.Insert:
                    target_ops.append("I")
                else:
                    target_ops.append("D")
            repaired = whitespace.repair(item.data.processed, pred_ops)
            strings.append(
                "\n".join([
                    f"Input      : {item.data.processed}",
                    f"Target     : {item.data.original}",
                    f"Target pred: {''.join(target_ops)}",
                    f"Prediction : {''.join(repair_ops)}",
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
