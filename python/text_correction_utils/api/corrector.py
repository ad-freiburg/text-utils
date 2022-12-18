import collections
import math
import os
import pprint
from typing import Dict, Iterable, List, Optional, Sized, Union, Tuple, Iterator, Any

import torch
from torch import autocast, nn

from text_correction_utils import api, logging, configuration, io, data

__all__ = ["ModelInfo"]

ModelInfo = collections.namedtuple("ModelInfo", ["name", "description", "tags"])


class Corrector:
    task: str

    @classmethod
    def _task_upper(cls) -> str:
        return cls.task.upper().replace(" ", "_")

    @classmethod
    def available_models(cls) -> List[ModelInfo]:
        raise NotImplementedError

    @classmethod
    def _model_url(cls, model: str) -> str:
        raise NotImplementedError

    @classmethod
    def download_dir(cls) -> str:
        return os.environ.get(
            f"{cls._task_upper()}_DOWNLOAD_DIR",
            os.path.join(os.path.dirname(__file__), ".download", cls._task_upper())
        )

    @classmethod
    def cache_dir(cls) -> str:
        return os.environ.get(
            f"{cls._task_upper()}_CACHE_DIR",
            os.path.join(os.path.dirname(__file__), ".cache", cls._task_upper())
        )

    @classmethod
    def from_pretrained(
            cls,
            model: str,
            device: Union[str, int] = "cuda",
            download_dir: Optional[str] = None,
            cache_dir: Optional[str] = None,
            force_download: bool = False
    ):
        assert any(model == m.name for m in cls.available_models()), \
            f"model {model} does not match any of the available models:\n{pprint.pformat(cls.available_models())}"

        logger = logging.get_logger(f"{cls._task_upper()}_DOWNLOAD")
        model_url = cls._model_url(model)
        if download_dir is None:
            download_dir = cls.download_dir()
        if cache_dir is None:
            cache_dir = cls.cache_dir()
        sub_cache_dir = model.lower().replace(" ", "_")
        zip_dir = api.download_zip(model, model_url, download_dir, cache_dir, sub_cache_dir, force_download, logger)
        sub_dirs = os.listdir(zip_dir)
        assert len(sub_dirs) == 1, f"expected extracted zip for model {model} to contain \
one subdirectory, but got {len(sub_dirs)}:\n{pprint.pformat(sub_dirs)}"
        return cls(os.path.join(zip_dir, sub_dirs[0]), device)

    @classmethod
    def from_experiment(
            cls,
            experiment_dir: str,
            device: Union[str, int] = "cuda"
    ):
        return cls(experiment_dir, device)

    @property
    def name(self) -> str:
        raise NotImplementedError

    @classmethod
    def _model_from_config(cls, cfg: Dict[str, Any]) -> nn.Module:
        raise NotImplementedError

    @property
    def max_length(self) -> int:
        raise NotImplementedError

    @property
    def context_length(self) -> int:
        raise NotImplementedError

    def __init__(
        self,
            model_dir: str,
            device: Union[str, int]
    ) -> None:
        self.logger = logging.get_logger(self._task_upper())

        torch.set_num_threads(len(os.sched_getaffinity(0)))
        torch.use_deterministic_algorithms(False)

        if device != "cpu" and not torch.cuda.is_available():
            self.logger.info("could not find a GPU, using CPU as fallback option")
            device = "cpu"

        self.device = torch.device(device)

        self.cfg = configuration.load_config(os.path.join(model_dir, "config.yaml"))
        self.logger.debug(f"loaded config:\n{self.cfg}")

        self.model = self._model_from_config(self.cfg)
        best_checkpoint_path = os.path.join(model_dir, "checkpoints", "checkpoint_best.pt")
        best_checkpoint = io.load_checkpoint(best_checkpoint_path)
        self.model.load_state_dict(best_checkpoint["model_state_dict"])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)

        self._mixed_precision_dtype = torch.float32
        self._inference_loader_cfg = self._build_inference_loader_config()

    def _build_inference_loader_config(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _prepare_batch(self, batch: data.InferenceItemBatch) -> Dict[str, Any]:
        raise NotImplementedError

    @torch.inference_mode()
    def _run_model(self, batch: data.InferenceItemBatch) -> Any:
        # this is a slight hack for now, because fp32 on cpu throws an error even when enabled=False
        inputs = self._prepare_batch(batch)
        if self.mixed_precision_enabled:
            with autocast(
                    device_type=self.device.type,
                    dtype=self._mixed_precision_dtype,
                    enabled=self.mixed_precision_enabled
            ):
                outputs, _ = self.model(**inputs)
        else:
            outputs, _ = self.model(**inputs)
        return outputs

    def _process_results(self, items: List[data.InferenceItem], outputs: List[Any]) -> Any:
        raise NotImplementedError

    def _get_loader(
            self,
            inputs: Union[List[str], Iterator[str], Iterator[Tuple[str, Optional[str]]]],
            input_type: str,
            languages: Optional[List[str]],
            batch_size: int = 16,
            batch_max_tokens: Optional[int] = None,
            sort: bool = True,
            num_threads: Optional[int] = None,
            **kwargs: Dict[str, Any]
    ) -> data.InferenceLoader:
        if num_threads is None:
            num_threads = min(len(os.sched_getaffinity(0)), 4)
        if batch_max_tokens is None:
            batch_limit = prefetch_factor = batch_size
            batch_limit_type = "batch_size"
            buffer_size = batch_limit * batch_limit
        else:
            batch_limit = max(batch_max_tokens, self.max_length)
            batch_limit_type = "padded_item_size"
            min_items_per_batch = math.ceil(batch_limit / self.max_length)
            buffer_size = min_items_per_batch * min_items_per_batch
            prefetch_factor = min_items_per_batch

        self._inference_loader_cfg.update({
            "num_threads": num_threads,
            "batch_limit": batch_limit,
            "buffer_size": buffer_size,
            "prefetch_factor": prefetch_factor,
            "batch_limit_type": batch_limit_type,
            "sort": sort
        })
        self._inference_loader_cfg.update(kwargs)
        if input_type == "files":
            self._inference_loader_cfg.update({
                "languages": languages
            })
            loader = data.InferenceLoader.from_files(
                inputs,
                **self._inference_loader_cfg
            )
        elif input_type == "sequences":
            self._inference_loader_cfg["num_threads"] = 0
            loader = data.InferenceLoader.from_iterator(
                ((seq, None, languages[i] if languages is not None else None)
                 for i, seq in enumerate(inputs)),
                **self._inference_loader_cfg
            )
        elif input_type == "iterator":
            self._inference_loader_cfg["num_threads"] = 0
            loader = data.InferenceLoader.from_iterator(
                ((seq, None, lang) for seq, lang in inputs), **self._inference_loader_cfg
            )
        else:
            raise ValueError(f"unknown input type {input_type}")
        return loader

    def _correct_sorted(self, loader: data.InferenceLoader) -> List[Any]:
        results = {}
        for batch in loader:
            outputs = self._run_model(batch)
            for item, output in zip(batch, outputs):
                if item.item_idx not in results:
                    results[item.item_idx] = {}
                results[item.item_idx][item.window_idx] = (item, output)
        outputs = []
        for item_idx in range(len(results)):
            window_items = []
            window_outputs = []
            for window_idx in range(len(results[item_idx])):
                item, output = results[item_idx][window_idx]
                window_items.append(item)
                window_outputs.append(output)
            outputs.append(self._process_results(window_items, window_outputs))
        return outputs

    def _correct_unsorted(self, loader: data.InferenceLoader) -> Iterator[Any]:
        prev_item_idx = 0
        window_items = []
        window_outputs = []
        for batch in loader:
            outputs = self._run_model(batch)
            for item, output in zip(batch, outputs):
                if item.item_idx == prev_item_idx:
                    window_items.append(item)
                    window_outputs.append(output)
                    continue
                yield self._process_results(window_items, window_outputs)
                prev_item_idx = item.item_idx
                window_items = [item]
                window_outputs = [output]
        # dont forget to yield final item
        yield self._process_results(window_items, window_outputs)

    def to(self, device: Union[str, int]) -> "Corrector":
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self

    def set_precision(self, precision: str) -> None:
        assert precision in {"fp32", "fp16", "bfp16"}

        if precision == "fp32":
            mixed_precision_dtype = torch.float32
        elif precision == "fp16":
            mixed_precision_dtype = torch.float16
        else:
            mixed_precision_dtype = torch.bfloat16

        if self.device.type == "cpu" and precision == "fp16":
            self.logger.info("Setting precision to bfp16 instead of fp16, because fp16 is not supported on CPU yet")
            mixed_precision_dtype = torch.bfloat16

        self._mixed_precision_dtype = mixed_precision_dtype

    @property
    def mixed_precision_enabled(self) -> bool:
        return self._mixed_precision_dtype != torch.float32
