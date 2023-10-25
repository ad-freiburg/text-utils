import logging
from typing import Optional, Union, Dict, Any, List, Tuple

import torch
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from text_utils import data, whitespace, tokenization


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
    ):
        self.name = name
        self.val_tensor = torch.zeros(1, dtype=torch.float, device=device)
        self.count_tensor = torch.zeros(1, dtype=torch.float, device=device)
        self.fmt = fmt
        self.val_reduce_op = val_reduce_op
        self.count_reduce_op = count_reduce_op
        self.synced = False

    def add(self, v: Union[float, int], count: int = 1):
        self.val_tensor[0] += v
        self.count_tensor[0] += count

    def sync(self):
        dist.all_reduce(self.val_tensor, op=self.val_reduce_op)
        dist.all_reduce(self.count_tensor, op=self.count_reduce_op)
        self._synced = True

    @property
    def value(self) -> float:
        assert self._synced
        return self.val_tensor.item() / max(1, self.count_tensor.item())

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
    def set_values(self, items: List[data.Item], outputs: torch.Tensor):
        raise NotImplementedError


class TextGenerationMetric(TensorboardMetric):
    def __init__(
        self,
        name: str,
        input_tokenizer: tokenization.Tokenizer,
        output_tokenizer: tokenization.Tokenizer,
        max_items: Optional[int] = None,
    ):
        self.items = None
        self.outputs = None
        self.max_items = max_items
        self.name = name
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer

    def set_values(self, items: List[data.Item], outputs: torch.Tensor):
        self.items = items
        self.outputs = torch.argmax(outputs, dim=-1).tolist()

    def _get_string(self) -> str:
        assert self.items is not None and self.outputs is not None, "call set_values before logging"
        strings = []
        for item, output in zip(self.items, self.outputs):
            if self.max_items is not None and len(strings) >= self.max_items:
                break

            input = self.input_tokenizer.de_tokenize(
                item.tokenization.token_ids,
                ignore_special_tokens=False
            )
            prediction = self.output_tokenizer.de_tokenize(
                output,
                ignore_special_tokens=False
            )
            strings.append(
                "\n".join([
                    f"Lang      : {item.data.language}\n",
                    f"Input     : {input}\n",
                    f"Target    : {item.data.target}\n",
                    f"Prediction: {prediction}\n",
                ])
            )

        return ("\n" + "-" * 80 + "\n").join(strings) + "\n"

    def log_tensorboard(self, writer: SummaryWriter, step: int):
        s = self._get_string()
        writer.add_text(self.name, s, step)

    def log_info(self, logger: logging.Logger, step: int):
        s = self._get_string()
        logger.info(f"[step {step}] {self.name}\n{s}")


class WhitespaceCorrectionMetric(TensorboardMetric):
    def __init__(
        self,
        name: str,
        input_tokenizer: tokenization.Tokenizer,
        max_items: Optional[int] = None,
        multi_layer: bool = False
    ):
        self.items = None
        self.outputs = None
        self.max_items = max_items
        self.name = name
        self.input_tokenizer = input_tokenizer
        self.multi_layer = multi_layer

    def set_values(self, items: List[data.Item], outputs: torch.Tensor):
        self.items = items
        self.outputs = torch.argmax(outputs, dim=-1).tolist()

    def _get_strings_and_acc(self, outputs) -> Tuple[str, float]:
        assert self.items is not None
        strings = []
        correct = 0
        total = 0
        for item, output in zip(self.items, outputs):
            if self.max_items is not None and len(strings) >= self.max_items:
                break

            start = self.input_tokenizer.num_prefix_tokens()
            end = len(item.tokenization.token_ids) - self.input_tokenizer.num_suffix_tokens()
            if "token_groups" == item.tokenization.info["type"]:
                if "code_point_groups" in item.tokenization.info:
                    group_name = "code_point_groups"
                else:
                    group_name = "byte_groups"
                end = len(item.tokenization.info[group_name]["groups"]) - self.input_tokenizer.num_suffix_tokens()

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

            target_ops = whitespace.operations(item.data.input, item.data.target)
            correct += sum(to == po for to, po in zip(target_ops, repair_ops))
            total += len(target_ops)

            repaired = whitespace.repair(item.data.input, repair_ops)
            strings.append(
                "\n".join([
                    f"Lang       : {item.data.language}\n",
                    f"Input      : {item.data.input}\n",
                    f"Target     : {item.data.target}\n",
                    f"Target pred: {''.join(target_ops)}\n",
                    f"Prediction : {''.join(repair_ops)}\n",
                    f"Repaired   : {repaired}"
                ])
            )

        return ("\n" + "-" * 80 + "\n").join(strings) + "\n", correct / total

    def log_tensorboard(self, writer: SummaryWriter, step: int):
        if self.multi_layer:
            accuracies = {}
            for layer in range(len(self.outputs)):
                s, acc = self._get_strings_and_acc(self.outputs[layer])
                # text only for last layer
                if layer == len(self.outputs) - 1:
                    writer.add_text(self.name, s, step)
                accuracies[f"layer_{layer}"] = acc
            writer.add_scalars(
                f"{self.name}_accuracy",
                accuracies,
                step
            )
        else:
            s, acc = self._get_strings_and_acc(self.outputs)
            writer.add_text(self.name, s, step)
            writer.add_scalar(
                f"{self.name}_accuracy",
                acc,
                step
            )

    def log_info(self, logger: logging.Logger, step: int):
        if self.multi_layer:
            for layer in range(len(self.outputs)):
                s, acc = self._get_strings_and_acc(self.outputs[layer])
                # text only for last layer
                if layer == len(self.outputs) - 1:
                    logger.info(f"[step {step}] {self.name}\n{s}")
                logger.info(
                    f"[step {step}] {self.name} "
                    f"layer {layer} accuracy: {100 * acc:.2f}"
                )
        else:
            s, acc = self._get_strings_and_acc(self.outputs)
            logger.info(f"[step {step}] {self.name}\n{s}")
            logger.info(
                f"[step {step}] {self.name} "
                f"accuracy: {100 * acc:.2f}"
            )


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
        elif metric_type == "text_generation":
            metric = TextGenerationMetric(f"{prefix}text_generation", input_tokenizer, output_tokenizer, **metric_opts)
        else:
            raise ValueError(f"unknown metric type {metric_type}")
        metrics.append(metric)
    return metrics
