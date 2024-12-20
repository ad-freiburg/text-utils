import collections
import math
import os
import pprint
import sys
from typing import Any, Callable, Iterator

import torch
from torch import nn
from torch.backends import cuda, cudnn

from text_utils import (
    api,
    configuration,
    data,
    io,
    logging,
)
from text_utils.api.utils import Device, get_devices

__all__ = ["ModelInfo"]

ModelInfo = collections.namedtuple("ModelInfo", ["name", "description", "tags"])


class TextProcessor:
    task: str
    pretrained: bool = False
    devices: list[torch.device]

    @classmethod
    def available_models(cls) -> list[ModelInfo]:
        raise NotImplementedError

    @classmethod
    def default_model(cls) -> ModelInfo | None:
        available_models = cls.available_models()
        if len(available_models) == 0:
            return None
        for info in available_models:
            if "default" in info.tags:
                return info
        return available_models[0]

    @classmethod
    def _model_url(cls, model: str) -> str:
        raise NotImplementedError

    @classmethod
    def download_dir(cls) -> str:
        task_name = cls.task.upper().replace(" ", "_")
        return os.environ.get(
            f"{task_name}_DOWNLOAD_DIR",
            os.path.join(os.path.dirname(__file__), ".download", task_name),
        )

    @classmethod
    def cache_dir(cls) -> str:
        task_name = cls.task.upper().replace(" ", "_")
        return os.environ.get(
            f"{task_name}_CACHE_DIR",
            os.path.join(os.path.dirname(__file__), ".cache", task_name),
        )

    @classmethod
    def from_pretrained(
        cls,
        model: str | None = None,
        device: Device = "cuda",
        download_dir: str | None = None,
        cache_dir: str | None = None,
        force_download: bool = False,
    ):
        if model is None:
            default = cls.default_model()
            assert default is not None, "No default model available"
            model = default.name

        assert model is not None
        assert any(model == m.name for m in cls.available_models()), (
            f"Model {model} does not match any of the available models:\n"
            f"{pprint.pformat(cls.available_models())}"
        )

        logger = logging.get_logger(f"{cls.task.upper()} DOWNLOAD")
        model_url = cls._model_url(model)
        if download_dir is None:
            download_dir = cls.download_dir()
        if cache_dir is None:
            cache_dir = cls.cache_dir()
        sub_cache_dir = model.lower().replace(" ", "_")
        zip_dir = api.download_zip(
            model,
            model_url,
            download_dir,
            cache_dir,
            sub_cache_dir,
            force_download,
            logger,
        )
        sub_dirs = os.listdir(zip_dir)
        assert len(sub_dirs) == 1, (
            f"Expected extracted zip for model {model} to contain "
            f"one subdirectory, but got {len(sub_dirs)}:\n{pprint.pformat(sub_dirs)}"
        )
        # mark processor as pretrained
        cls.pretrained = True
        return cls.from_experiment(os.path.join(zip_dir, sub_dirs[0]), device)

    @classmethod
    def from_experiment(
        cls, experiment_dir: str, device: Device = "cuda", last: bool = False
    ):
        cfg = configuration.load_config_from_experiment(experiment_dir)
        model = cls._model_from_config(cfg, device)
        ckpt_keys = ["best", "last"]
        if last:
            ckpt_keys = reversed(ckpt_keys)

        for ckpt in ckpt_keys:
            ckpt_path = os.path.join(
                experiment_dir, "checkpoints", f"checkpoint_{ckpt}.pt"
            )
            if not os.path.exists(ckpt_path):
                continue

            checkpoint = io.load_checkpoint(ckpt_path)
            model.load_state_dict(checkpoint["model_state_dict"])

        model = model.eval().requires_grad_(False)
        return cls(model, cfg, device)

    @property
    def name(self) -> str:
        raise NotImplementedError

    @classmethod
    def _model_from_config(cls, cfg: dict[str, Any], device: Device) -> nn.Module:
        raise NotImplementedError

    @property
    def max_length(self) -> int:
        raise NotImplementedError

    def __init__(
        self, model: nn.Module, cfg: dict[str, Any], device: Device = "cuda"
    ) -> None:
        self.cfg = cfg
        self.logger = logging.get_logger(self.task.upper())
        self.logger.debug(f"Got config:\n{self.cfg}")

        torch.set_num_threads(len(os.sched_getaffinity(0)))
        torch.use_deterministic_algorithms(False)
        cudnn.benchmark = True
        cuda.matmul.allow_tf32 = True

        self.model = model
        self.to(device)

    def _process(
        self,
        iter: Iterator[str],
        inference_fn: Callable[[data.InferenceBatch], list[Any]],
        postprocessing_fn: Callable[[list[data.InferenceItem], list[Any]], Any],
        progress_desc: str,
        batch_size: int = 16,
        batch_max_tokens: int | None = None,
        sort: bool = True,
        num_threads: int | None = None,
        progress_total: int | None = None,
        progress_unit: str = "it",
        show_progress: bool = False,
        **kwargs: Any,
    ) -> Iterator[Any]:
        if num_threads is None:
            num_threads = min(len(os.sched_getaffinity(0)), 4)

        if batch_max_tokens is None:
            batch_limit = max(1, batch_size)
            batch_limit_type = "batch_size"
            buffer_size = batch_limit
        else:
            batch_limit = max(batch_max_tokens, self.max_length)
            batch_limit_type = "padded_item_size"
            min_items_per_batch = math.ceil(batch_limit / self.max_length)
            buffer_size = min_items_per_batch

        if sort:
            prefetch_factor = sys.maxsize
        else:
            prefetch_factor = 1

        inference_cfg = {
            "tokenizer": self.cfg["inference"]["tokenizer"],
            "window": self.cfg["inference"].get("window", {"type": "full"}),
            "num_threads": num_threads,
            "batch_limit": batch_limit,
            "buffer_size": buffer_size,
            "prefetch_factor": prefetch_factor,
            "batch_limit_type": batch_limit_type,
            "sort": sort,
        }
        inference_cfg.update(kwargs)
        loader = data.InferenceLoader.from_iterator(iter, **inference_cfg)

        pbar = api.progress_bar(
            progress_desc, progress_total, progress_unit, show_progress
        )
        if sort:
            results = {}
            for batch in loader:
                with torch.inference_mode():
                    outputs = inference_fn(batch)

                for item, output in zip(batch.items(), outputs):
                    if item.item_idx not in results:
                        results[item.item_idx] = {}
                        if progress_unit == "it":
                            pbar.update(1)

                    if progress_unit == "byte":
                        pbar.update(item.window_bytes())

                    results[item.item_idx][item.window_idx] = (item, output)

            outputs = []
            for item_idx in range(len(results)):
                window_items = []
                window_outputs = []
                for window_idx in range(len(results[item_idx])):
                    item, output = results[item_idx][window_idx]
                    window_items.append(item)
                    window_outputs.append(output)

                yield postprocessing_fn(window_items, window_outputs)

        else:
            # not sorted, we can yield as we go
            prev_item_idx = 0
            window_items = []
            window_outputs = []
            for batch in loader:
                with torch.inference_mode():
                    outputs = inference_fn(batch)

                for item, output in zip(batch.items(), outputs):
                    if item.item_idx == prev_item_idx:
                        window_items.append(item)
                        window_outputs.append(output)
                        continue

                    yield postprocessing_fn(window_items, window_outputs)
                    if progress_unit == "byte":
                        pbar.update(sum(item.window_bytes() for item in window_items))
                    elif progress_unit == "it":
                        pbar.update(1)

                    prev_item_idx = item.item_idx
                    window_items = [item]
                    window_outputs = [output]

            # dont forget to yield final item
            yield postprocessing_fn(window_items, window_outputs)

        pbar.close()

    def to(self, device: Device) -> "TextProcessor":
        self.devices = get_devices(device)
        assert len(self.devices) == 1, (
            "only a single device supported by default, implement custom to() if you need "
            "multi-device support"
        )
        self.model = self.model.to(self.devices[0])
        return self
