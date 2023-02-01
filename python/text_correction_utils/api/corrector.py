import collections
import math
import sys
import os
import pprint
from typing import Dict, List, Optional, Union, Tuple, Iterator, Any

from tqdm import tqdm
import torch
from torch import autocast, nn
from torch.backends import cudnn

from text_correction_utils import api, logging, configuration, io, data

__all__ = ["ModelInfo"]

ModelInfo = collections.namedtuple("ModelInfo", ["name", "description", "tags"])


class TextCorrector:
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

    def supported_languages(self) -> List[str]:
        return []

    @classmethod
    def supported_input_formats(cls) -> List[str]:
        return ["text", "text_language"]

    @classmethod
    def supported_output_formats(cls) -> List[str]:
        return cls.supported_input_formats()

    def __init__(
        self,
            model_dir: str,
            device: Union[str, int]
    ) -> None:
        self.logger = logging.get_logger(self._task_upper())

        torch.set_num_threads(len(os.sched_getaffinity(0)))
        torch.use_deterministic_algorithms(False)
        cudnn.benchmark = True

        if device != "cpu" and not torch.cuda.is_available():
            self.logger.info("could not find a GPU, using CPU as fallback option")
            device = "cpu"

        self.device = torch.device(device)

        info = configuration.load_config(os.path.join(model_dir, "info.yaml"))
        self.logger.debug(f"loaded info:\n{info}")
        self.cfg = configuration.load_config(os.path.join(model_dir, info["config_name"]))
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

    def _prepare_batch(self, batch: data.InferenceBatch) -> Dict[str, Any]:
        raise NotImplementedError

    def _inference(self, inputs: Dict[str, Any]) -> Any:
        raise NotImplementedError

    @torch.inference_mode()
    def _run_model(self, batch: data.InferenceBatch) -> Any:
        inputs = self._prepare_batch(batch)
        # this is a slight hack for now, because fp32 on cpu throws an error even when enabled=False
        if self.mixed_precision_enabled:
            with autocast(
                    device_type=self.device.type,
                    dtype=self._mixed_precision_dtype,
                    enabled=self.mixed_precision_enabled
            ):
                outputs = self._inference(inputs)
        else:
            outputs = self._inference(inputs)
        return outputs

    def _process_results(
        self,
        items: List[data.InferenceItem],
        outputs: List[Any]
    ) -> data.InferenceData:
        raise NotImplementedError

    def _get_loader(
            self,
            inputs: Union[Tuple[List[str], Optional[List[str]]], Iterator[data.InferenceData]],
            batch_size: int = 16,
            batch_max_tokens: Optional[int] = None,
            sort: bool = True,
            num_threads: Optional[int] = None,
            **kwargs: Any
    ) -> data.InferenceLoader:
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

        if sorted:
            prefetch_factor = sys.maxsize
        else:
            prefetch_factor = 1

        self._inference_loader_cfg.update({
            "num_threads": num_threads,
            "batch_limit": batch_limit,
            "buffer_size": buffer_size,
            "prefetch_factor": prefetch_factor,
            "batch_limit_type": batch_limit_type,
            "sort": sort
        })
        self._inference_loader_cfg.update(kwargs)
        if isinstance(inputs, tuple):
            files, languages = inputs
            loader = data.InferenceLoader.from_files(
                files=files,
                languages=languages,
                **self._inference_loader_cfg
            )
        elif isinstance(inputs, Iterator):
            # threading currently not supported with python iterators
            self._inference_loader_cfg["num_threads"] = 0
            loader = data.InferenceLoader.from_iterator(
                inputs,
                **self._inference_loader_cfg
            )
        else:
            raise ValueError(
                f"unknown input type {type(inputs)}, must either be a tuple of files and languages or an iterator \
                over sequence language pairs"
            )

        return loader

    def _pbar(
        self,
        progress_desc: str,
        progress_total: int,
        progress_unit: str = "seq",
        show_progress: bool = False,
    ) -> tqdm:
        if progress_unit == "seq":
            return api.sequence_progress_bar(progress_desc, progress_total, not show_progress)
        elif progress_unit == "byte":
            return api.byte_progress_bar(progress_desc, progress_total, not show_progress)
        else:
            raise ValueError(f"unknown progress unit {progress_unit}, must be either 'seq' or 'byte'")

    def _correct_sorted(
        self,
        loader: data.InferenceLoader,
        progress_desc: str,
        progress_total: int,
        progress_unit: str = "seq",
        show_progress: bool = False,
    ) -> List[data.InferenceData]:
        results = {}
        pbar = self._pbar(progress_desc, progress_total, progress_unit, show_progress)
        for batch in loader:
            outputs = self._run_model(batch)
            for item, output in zip(batch.items, outputs):
                if item.item_idx not in results:
                    results[item.item_idx] = {}
                    if progress_unit == "seq":
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
            outputs.append(self._process_results(window_items, window_outputs))
        return outputs

    def _correct_unsorted(
        self,
        loader: data.InferenceLoader,
        progress_desc: str,
        progress_total: int,
        progress_unit: str = "seq",
        show_progress: bool = False,
    ) -> Iterator[data.InferenceData]:
        prev_item_idx = 0
        window_items = []
        window_outputs = []
        pbar = self._pbar(progress_desc, progress_total, progress_unit, show_progress)
        for batch in loader:
            outputs = self._run_model(batch)
            for item, output in zip(batch.items, outputs):
                if item.item_idx == prev_item_idx:
                    window_items.append(item)
                    window_outputs.append(output)
                    continue
                yield self._process_results(window_items, window_outputs)
                if progress_unit == "seq":
                    pbar.update(1)
                else:
                    pbar.update(sum(item.window_bytes() for item in window_items))
                prev_item_idx = item.item_idx
                window_items = [item]
                window_outputs = [output]
        # dont forget to yield final item
        yield self._process_results(window_items, window_outputs)

    def to(self, device: Union[str, int]) -> "TextCorrector":
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
            self.logger.info("setting precision to bfp16 instead of fp16, because fp16 is not supported on CPU yet")
            mixed_precision_dtype = torch.bfloat16

        self._mixed_precision_dtype = mixed_precision_dtype

    @ property
    def mixed_precision_enabled(self) -> bool:
        return self._mixed_precision_dtype != torch.float32
