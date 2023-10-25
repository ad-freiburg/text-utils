import argparse
import math
import copy
import functools
import sys
import random
import os
import hashlib
import shutil
import time
import zipfile
from typing import Dict, Optional, Tuple, Any, List, Callable, Union
from text_utils.api.utils import get_peft_config

import torch
from torch.backends import cuda, cudnn
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import nn
from torch.optim import lr_scheduler
from torch.backends import cudnn, cuda  # noqa
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import (
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig,
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.tensorboard.writer import SummaryWriter
from peft import PeftConfig
import yaml

from text_utils.modules.loss import loss_from_config
from text_utils.modules.scheduler import (
    lr_scheduler_from_config,
    max_length_scheduler_from_config
)
from text_utils.modules.optimizer import optimizer_from_config
from text_utils import (
    distributed,
    data,
    configuration,
    io,
    tokenization,
    logging,
    api,
    tensorboard
)


def clamp(v: float, minimum: int, maximum: int) -> int:
    return max(min(math.floor(v), maximum), minimum)


ShardingPolicy = Callable[[nn.Module, bool, int], bool]


class Trainer:
    @classmethod
    def parser(cls, name: str, description: str) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(name, description)
        parser.add_argument(
            "-e",
            "--experiment",
            type=str,
            required=True,
            help="Path to directory where experiment will be saved. If experiment already exists, \
training will resume from latest checkpoint."
        )
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            default=None,
            help="Path to config file, only required for a new training run."
        )
        parser.add_argument(
            "-p",
            "--platform",
            type=str,
            choices=["local", "slurm"],
            default="local",
            required=True,
            help="Platform used for training."
        )
        parser.add_argument(
            "--profile",
            type=str,
            default=None,
            help="Run cProfile profile on main process and output stats to this file "
            "(only respected if platform=local)"
        )
        return parser

    def __init__(
        self,
        cfg: Dict[str, Any],
        directories: Dict[str, str],
        info: distributed.DistributedInfo
    ):
        self.cfg = cfg
        self.directories = directories
        self.info = info

        # globals used throughout training
        self.epoch_items = 0
        self.total_items = 0
        self.total_step = 0
        self.epoch_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.best_benchmark: Optional[float] = None
        self.logger = logging.get_logger("TRAIN")

        if self.info.is_main_process:
            log_file = os.path.join(self.directories["experiment"], "logs.txt")
            logging.add_file_log(self.logger, log_file)

        device_props = api.device_info(self.info.device)
        self.logger.info(
            f"[GPU:{self.info.rank}:{self.info.local_rank}] {device_props}"
        )
        cpu_props = api.cpu_info()
        self.logger.info(
            f"[CPU:{self.info.rank}:{self.info.local_rank}] {cpu_props}, "
            f"{len(os.sched_getaffinity(0))} cores available"
        )

        torch.manual_seed(self.cfg["seed"])
        torch.cuda.manual_seed(self.cfg["seed"])

        torch.use_deterministic_algorithms(False)
        cuda.matmul.allow_tf32 = True
        cudnn.benchmark = True

        self.input_tokenizer = tokenization.Tokenizer.from_config(
            self.cfg["input_tokenizer"]
        )
        if "output_tokenizer" in self.cfg:
            self.output_tokenizer = tokenization.Tokenizer.from_config(
                self.cfg["output_tokenizer"]
            )
        else:
            self.output_tokenizer = None

        model, sharding_policy = self._model_from_config(self.cfg)

        peft = self.cfg["train"].get("peft", None)
        if peft is not None:
            if self.info.is_main_process:
                self.logger.info(
                    "preparing model for parametering efficient fine tuning with config:\n"
                    f"{yaml.safe_dump(peft)}"
                )
            model = self._prepare_peft(
                model,
                get_peft_config(peft),
                peft.get("use_8bit", False)
            )

        precision = self.cfg["train"].get("precision", "fp32")
        if precision == "fp32":
            self.precision_dtype = torch.float32
        elif precision == "fp16":
            self.precision_dtype = torch.float16
        elif precision == "bfp16":
            self.precision_dtype = torch.bfloat16
        else:
            raise ValueError(
                f"unknown precision {precision}, "
                f"must be fp32, fp16 or bfp16"
            )

        compile = self.cfg["train"].get("compile", False)
        dist_cfg = self.cfg["train"].get("distributed", {})
        dist_type = dist_cfg["type"]
        assert dist_type in {"DDP", "FSDP"}, \
            f"distributed training type must be either DDP or FSDP, but got {dist_type}"

        if dist_type == "DDP":
            self.model = DDP(
                model.to(self.info.device),
                static_graph=compile
            )
        else:
            offload_params = dist_cfg.get("offload", False)
            prefetch = dist_cfg.get("prefetch", True)
            strategy = ShardingStrategy[dist_cfg.get("strategy", "NO_SHARD")]
            if strategy != ShardingStrategy.NO_SHARD:
                shard_size = dist_cfg.get("shard_size", None)
                if shard_size is not None:
                    if self.info.is_main_process:
                        self.logger.info(
                            f"sharding based on number of parameters with "
                            f"a minimum of {shard_size:,}"
                        )
                    sharding_policy = functools.partial(
                        size_based_auto_wrap_policy,
                        min_num_params=shard_size,
                        force_leaf_modules=None,
                        exclude_wrap_modules=None
                    )
                elif sharding_policy is None:
                    if self.info.is_main_process:
                        self.logger.info(
                            f"sharding strategy is {strategy.name}, but got "
                            f"no sharding policy, disabling sharding"
                        )
                    strategy = ShardingStrategy.NO_SHARD
            else:
                sharding_policy = None
                offload_params = False

            offload_state_dict = self.info.world_size > 1

            self.model = FSDP(
                model,
                auto_wrap_policy=sharding_policy,
                mixed_precision=MixedPrecision(
                    param_dtype=self.precision_dtype,
                    reduce_dtype=self.precision_dtype,
                    buffer_dtype=self.precision_dtype
                ),
                cpu_offload=CPUOffload(offload_params=offload_params),
                limit_all_gathers=True,
                sharding_strategy=strategy,
                forward_prefetch=prefetch,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE if prefetch else BackwardPrefetch.BACKWARD_POST,
                device_id=self.info.device,
                use_orig_params=compile or (peft is not None),
            )
            FSDP.set_state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(
                    offload_to_cpu=offload_state_dict,
                    rank0_only=offload_state_dict
                ),
                FullOptimStateDictConfig(
                    offload_to_cpu=offload_state_dict,
                    rank0_only=offload_state_dict
                )
            )

        self.model: Union[DDP, FSDP] = torch.compile(self.model, disable=not compile)  # type: ignore

        self.optimizer = optimizer_from_config(
            self.model,
            self.cfg["train"]["optimizer"],
            additional_optimizer_fn=self._additional_optimizer_fn()
        )

        self.clip_grad_norm: Optional[float] = self.cfg["train"].get("clip_grad_norm", None)

        num_epochs = self.cfg["train"]["num_epochs"]
        (
            self.train_loader,
            self.val_loader,
            self.training_items_per_epoch,
            self.training_items,
            self.max_length,
            self.max_length_scheduler,
            self.cleanup
        ) = self._data_from_config(
            self.cfg["train"]["data"],
            self.cfg["val"]["data"],
            self.cfg["input_tokenizer"],
            num_epochs=num_epochs,
            seed=self.cfg["seed"],
            info=self.info
        )

        lower = self.cfg["train"]["data"]["batch_limit"]
        if self.cfg["train"]["data"]["batch_limit_type"] != "batch_size":
            lower = lower // self.cfg["train"]["data"]["max_length"]
        self.log_interval = clamp(
            self.training_items *
            self.cfg["train"].get("log_interval", 0.001),
            lower,
            self.training_items
        )
        self.eval_interval = clamp(
            self.training_items *
            self.cfg["train"].get("eval_interval", 0.1),
            lower,
            self.training_items
        )
        cooldown = self.cfg["val"].get("cooldown", 0)
        if isinstance(cooldown, float):
            cooldown_items = self.training_items * cooldown
        elif isinstance(cooldown, int):
            cooldown_items = cooldown
        else:
            raise ValueError(f"cooldown must be a float between 0 and 1, but got {cooldown}")
        if cooldown_items > 0:
            self.cooldown_items = clamp(
                cooldown_items,
                lower,
                self.training_items
            )
            assert self.cooldown_items < self.eval_interval, \
                f"cooldown items {self.cooldown_items:,} must be smaller " \
                f"than evaluation interval {self.eval_interval:,}"
        else:
            self.cooldown_items = 0

        self.cooldown_scheduler: Optional[lr_scheduler.LambdaLR] = None

        if "lr_scheduler" in self.cfg["train"]:
            self.step_interval = clamp(
                self.training_items *
                self.cfg["train"].get("step_interval", 0.001),
                lower,
                self.training_items
            )
            steps = self.training_items // self.step_interval
            self.lr_scheduler = lr_scheduler_from_config(
                self.optimizer,
                steps,
                self.info.world_size,
                self.cfg["train"]["lr_scheduler"],
                additional_lr_scheduler_fn=self._additional_lr_scheduler_fn()
            )
        else:
            self.step_interval = 0
            self.lr_scheduler = None

        self.log_at = self.log_interval
        self.eval_at = self.eval_interval
        self.step_at = self.step_interval

        self.loss_fn = loss_from_config(
            self.cfg["train"]["loss"],
            additional_loss_fn=self._additional_loss_fn()
        ).to(self.info.device).train()

        if self.info.is_main_process:
            self.summary_writer = SummaryWriter(
                log_dir=self.directories["tensorboard"]
            )

            self.logger.info(f"Using model:\n{self.model}")
            self.logger.info(f"Model parameters: {api.num_parameters(self.model)}")
            num_params = 0
            param_group_infos = []
            for i, param_group in enumerate(self.optimizer.param_groups):
                group_num_params = sum(
                    p.numel()
                    for p in param_group["params"]
                )
                group_cfg = {k: v for k, v in param_group.items() if k != "params"}
                param_group_infos.append(
                    f"{i+1}. group: {group_num_params:,} params, other: {group_cfg}"
                )
                num_params += group_num_params
            param_group_info = "\n".join(param_group_infos)
            self.logger.info(
                f"Optimizer parameter groups:\n{param_group_info}"
            )
            self.logger.info(
                f"Training with {dist_type} and {precision} precision"
            )
            self.logger.info(
                f"Number of training items: {self.training_items_per_epoch:,} per epoch, "
                f"{self.training_items:,} total"
            )
            self.logger.info(
                f"Logging every {self.log_interval:,} items, "
                f"evaluating every {self.eval_interval:,} items"
                + f", stepping every {self.step_interval:,} items" if self.lr_scheduler is not None else ""
            )

            test_sentence = "This is a test sentence."
            self.logger.info(
                f"Testing input tokenizer:\n{self.input_tokenizer.tokenize(test_sentence).token_ids}"
            )
            if self.output_tokenizer is not None:
                self.logger.info(
                    f"Testing output tokenizer:\n{self.output_tokenizer.tokenize(test_sentence).token_ids}"
                )

            self.logger.info(
                f"Type 'tensorboard --logdir {self.directories['tensorboard']}' "
                f"to view the training process in Tensorboard"
            )
        else:
            self.summary_writer = None

        # resume training from last checkpoint if it exists
        last_checkpoint = os.path.join(
            self.directories["checkpoints"],
            "checkpoint_last.pt"
        )
        load_checkpoint = self.cfg["train"].get("load_checkpoint")
        if os.path.exists(last_checkpoint):
            self._load_checkpoint(last_checkpoint)
            if self.info.is_main_process:
                self.logger.info(
                    f"Resuming training from checkpoint {last_checkpoint}\n"
                    f"Starting at epoch {self.epoch + 1} at global step {self.total_step:,} "
                    f"(total items = {self.total_items:,}, epoch step {self.epoch_step:,}) "
                    f"with a best validation loss of {self.best_val_loss:.6f}\n"
                    f"Fast forwarding {self.epoch_items:,} items within epoch."
                )
        elif load_checkpoint is not None:
            checkpoint = io.load_checkpoint(load_checkpoint)
            wrong_keys = distributed.unwrap_model(self.model).load_state_dict(
                checkpoint["model_state_dict"],
                strict=False
            )
            assert len(wrong_keys.unexpected_keys) == 0, \
                f"unexpected keys in checkpoint \"{load_checkpoint}\": {wrong_keys.unexpected_keys}"

            self.logger.info(
                f"initializing model from checkpoint \"{load_checkpoint}\" "
                f"(missing keys: {wrong_keys.missing_keys})"
            )

        self.grad_scaler = ShardedGradScaler(
            enabled=precision != "fp32"
        )

    def _save_checkpoint(
        self,
        path: str,
        val_loss: float,
        full: bool = True,
        **kwargs: Any
    ):
        save = {
            "checkpoint_path": path,
            "model_state_dict": distributed.unwrap_model(self.model).state_dict(),
            "step": self.total_step,
            "epoch": self.epoch,
            "epoch_step": self.epoch_step,
            "epoch_items": self.epoch_items,
            "total_items": self.total_items,
            "val_loss": val_loss,
            **kwargs
        }
        if full:
            save["optimizer_state_dict"] = distributed.get_optimizer_state_dict(
                self.model,
                self.optimizer
            )
            save["loss_fn_state_dict"] = self.loss_fn.state_dict()
            if self.lr_scheduler is not None:
                save["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

        # only save on main process
        if self.info.is_main_process:
            io.save_checkpoint(**save)

        # wait until checkpoint is saved
        dist.barrier()

    def _load_checkpoint(self, path: str):
        checkpoint = io.load_checkpoint(path)
        distributed.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
        optim_state_dict = checkpoint["optimizer_state_dict"]
        if isinstance(self.model, FSDP):
            optim_state_dict = FSDP.optim_state_dict_to_load(
                optim_state_dict,
                self.model,
                self.optimizer,
            )
        self.optimizer.load_state_dict(optim_state_dict)
        if self.lr_scheduler is not None and checkpoint.get("lr_scheduler_state_dict") is not None:
            self.lr_scheduler.load_state_dict(
                checkpoint["lr_scheduler_state_dict"]
            )
        if checkpoint.get("loss_fn_state_dict") is not None:
            self.loss_fn.load_state_dict(
                checkpoint["loss_fn_state_dict"]
            )

        self.epoch = checkpoint["epoch"]
        self.epoch_step = checkpoint["epoch_step"]
        self.total_step = checkpoint["step"]
        self.best_val_loss = checkpoint["val_loss"]
        self.epoch_items = checkpoint["epoch_items"]
        self.total_items = checkpoint["total_items"]

        if self.max_length_scheduler is not None:
            self.max_length = self.max_length_scheduler(self.total_items)
            self.train_loader.set_max_length(self.max_length)

        self.train_loader.set_epoch(self.epoch)
        self.train_loader.set_fast_forward(self.epoch_items)

        # reset eval, log, and step counters
        self.log_at = math.ceil(self.total_items / self.log_interval) * self.log_interval
        self.eval_at = math.ceil(self.total_items / self.eval_interval) * self.eval_interval
        self.step_at = math.ceil(self.total_items / self.step_interval) * self.step_interval

        # wait until everyone loaded the checkpoint
        dist.barrier()

    @classmethod
    def _prepare_peft(
        cls,
        model: nn.Module,
        peft_cfg: PeftConfig,
        use_8bit: bool = False
    ) -> nn.Module:
        raise NotImplementedError

    @classmethod
    def _model_from_config(
        cls,
        cfg: Dict[str, Any]
    ) -> Tuple[nn.Module, Optional[ShardingPolicy]]:
        raise NotImplementedError

    @classmethod
    def _additional_loss_fn(cls) -> Optional[Callable]:
        return None

    @classmethod
    def _additional_optimizer_fn(cls) -> Optional[Callable]:
        return None

    @classmethod
    def _additional_lr_scheduler_fn(cls) -> Optional[Callable]:
        return None

    @classmethod
    def _additional_max_length_scheduler_fn(cls) -> Optional[Callable]:
        return None

    @classmethod
    def _copy_file_to_tmp_dir(cls, path: str, dir: str, info: distributed.DistributedInfo) -> str:
        path = os.path.abspath(path)
        _, file_name = os.path.split(path)
        # make temp path unique by hashing the full input path, because only
        # using the file name could cause issues if some files are named the same
        os.makedirs(dir, exist_ok=True)
        hash = hashlib.sha256(path.encode("utf8")).hexdigest()
        temp_path = os.path.join(dir, f"{hash}_{file_name}")
        # only copy file to temp dir once on local main process
        if info.is_local_main_process:
            shutil.copy2(path, temp_path)
        # wait on the other processes until file is copied,
        # otherwise errors occurr later
        dist.barrier()
        return temp_path

    @classmethod
    def _prepare_data_sources(
        cls,
        sources: List[Dict[str, Any]],
        info: distributed.DistributedInfo
    ) -> Tuple[
        List[Tuple[str, Optional[str]]],
        List[Optional[str]],
        List[Optional[Any]],
        List[Optional[Any]],
        List[str]
    ]:
        src_paths = []
        src_preprocessings = []
        src_postprocessings = []
        src_langs = []
        cleanup_paths = []
        for src in sources:
            src = copy.deepcopy(src)
            src_type = src.pop("type")
            if src_type == "file":
                lang = src.get("language")
                preprocessing = src.get("preprocessing")
                postprocessing = src.get("postprocessing")
                path = src["path"]
                assert os.path.isfile(path), f"{path} is not a file"
                temp_dir = src.get("temp_dir")
                if temp_dir is not None:
                    path = cls._copy_file_to_tmp_dir(path, temp_dir, info)
                    cleanup_paths.append(path)
                src_preprocessings.append(preprocessing)
                src_postprocessings.append(postprocessing)
                src_paths.append((path, None))
                src_langs.append(lang)
            elif src_type == "file_glob":
                lang = src.get("language")
                preprocessing = src.get("preprocessing")
                postprocessing = src.get("postprocessing")
                temp_dir = src.get("temp_dir")
                for path in io.glob_safe(src["glob"]):
                    assert os.path.isfile(path), f"{path} is not a file"
                    if temp_dir is not None:
                        path = cls._copy_file_to_tmp_dir(path, temp_dir, info)
                        cleanup_paths.append(path)
                    src_preprocessings.append(preprocessing)
                    src_postprocessings.append(postprocessing)
                    src_paths.append((path, None))
                    src_langs.append(lang)
            elif src_type == "file_pair":
                lang = src.get("language")
                preprocessing = src.get("preprocessing")
                input_path = src["input_path"]
                target_path = src["target_path"]
                assert os.path.isfile(input_path) and os.path.isfile(target_path), \
                    f"one of {input_path} or {target_path} is not a file"
                temp_dir = src.get("temp_dir")
                if temp_dir is not None:
                    input_path = cls._copy_file_to_tmp_dir(
                        input_path, temp_dir, info
                    )
                    target_path = cls._copy_file_to_tmp_dir(
                        target_path, temp_dir, info
                    )
                    cleanup_paths.extend([input_path, target_path])
                src_preprocessings.append(preprocessing)
                src_paths.append((input_path, target_path))
                src_langs.append(lang)
            else:
                raise ValueError(f"unknown source type {src_type}")
        assert len(src_paths) > 0, "got no data sources"
        return (
            src_paths,
            src_langs,
            src_preprocessings,
            src_postprocessings,
            cleanup_paths
        )

    @classmethod
    def _data_from_config(
        cls,
        train_cfg: Dict[str, Any],
        val_cfg: Union[List[Any], int],
        tokenizer_config: Dict[str, Any],
        num_epochs: int,
        seed: Optional[int],
        info: distributed.DistributedInfo
    ) -> Tuple[
        data.DataLoader,
        data.DataLoader,
        int,
        int,
        int,
        Optional[Callable[[int], int]],
        List[str]
    ]:
        def prepare_data_loader(
            default_language: Optional[str],
            pipeline_cfg: Dict[str, Any],
            sources: List[Dict[str, Any]],
            languages: List[Optional[str]],
            preprocessings: List[Optional[Any]],
            postprocessings: List[Optional[Any]],
            **kwargs: Any,
        ) -> data.DataLoader:
            num_languages_specified = sum(
                lang is not None for lang in languages
            )
            if num_languages_specified > 0 and num_languages_specified < len(languages):
                assert default_language is not None, \
                    "expected default_language to be specified if some, but not all " \
                    "individual data sources specify a language"
                languages = [
                    default_language if lang is None else lang
                    for lang in languages
                ]
            elif num_languages_specified == 0:
                languages = None  # type: ignore

            pipeline_cfg = copy.deepcopy(pipeline_cfg)
            if "preprocessing" not in pipeline_cfg:
                assert all(preproc is not None for preproc in preprocessings), \
                    "expected preprocessing to be specified per data source if not specified " \
                    "for pipeline"
                pipeline_cfg["preprocessing"] = preprocessings
            if "postprocessing" not in pipeline_cfg:
                assert all(postproc is not None for postproc in postprocessings), \
                    "expected postprocessing to be specified per data source if not specified " \
                    "for pipeline"
                pipeline_cfg["postprocessing"] = postprocessings

            return data.DataLoader.from_files(
                sources,
                pipeline_cfg,
                tokenizer_config,
                languages,
                **kwargs
            )

        train_cfg = copy.deepcopy(train_cfg)

        # adapt config to multi gpu usage
        assert "batch_limit" in train_cfg, "batch_limit must be in data config"
        train_cfg["batch_limit"] = max(1, train_cfg["batch_limit"] // info.world_size)

        # pop some configs not used by the dataloader
        max_length = train_cfg.pop("max_length")
        assert max_length is not None, "missing max_length in data config"
        max_length_scheduler_cfg = train_cfg.pop("max_length_scheduler", None)

        (
            *training,
            train_cleanup
        ) = cls._prepare_data_sources(
            train_cfg.pop("sources"),
            info
        )

        default_language = train_cfg.pop("default_language", None)
        pipeline_cfg = train_cfg.pop("pipeline")

        if isinstance(val_cfg, int):
            # if validation is a split of the training set
            train_limit = train_cfg.get("limit", None)
            if train_limit is not None:
                assert train_limit > val_cfg, \
                    f"train limit ({train_limit:,}) cannot be smaller or " \
                    f"equal to val limit ({val_cfg:,})"
            train_loader = prepare_data_loader(
                default_language,
                pipeline_cfg,
                *training,
                skip=val_cfg,
                seed=seed,
                max_length=max_length,
                distributed=(info.rank, info.world_size),
                **train_cfg,
            )
            # for validation always turn off shuffling, turn on sorting, and
            # specify the val limit
            val_loader = prepare_data_loader(
                default_language,
                pipeline_cfg,
                *training,
                limit=val_cfg,
                max_length=max_length,
                distributed=(info.rank, info.world_size),
                shuffle=False,
                sort=True
            )

        elif isinstance(val_cfg, list):
            # if validation is a separate set of data sources
            train_loader = prepare_data_loader(
                default_language,
                pipeline_cfg,
                *training,
                seed=seed,
                max_length=max_length,
                distributed=(info.rank, info.world_size),
                **train_cfg,
            )
            (
                *validation,
                val_cleanup
            ) = cls._prepare_data_sources(
                val_cfg,
                info
            )
            train_cleanup.extend(val_cleanup)
            val_loader = prepare_data_loader(
                default_language,
                pipeline_cfg,
                *validation,
                max_length=max_length,
                distributed=(info.rank, info.world_size),
                shuffle=False,
                sort=True
            )

        else:
            raise ValueError("unsupported validation config")

        # trigger train loader, so that min_items is set
        iter(train_loader)
        training_items_per_epoch = train_loader.min_items
        training_items = training_items_per_epoch * num_epochs

        if max_length_scheduler_cfg is not None:
            max_length_scheduler = max_length_scheduler_from_config(
                training_items,
                max_length,
                max_length_scheduler_cfg,
                additional_max_length_scheduler_fn=cls._additional_max_length_scheduler_fn()
            )
        else:
            max_length_scheduler = None

        return (
            train_loader,
            val_loader,
            training_items_per_epoch,
            training_items,
            max_length,
            max_length_scheduler,
            train_cleanup
        )

    @classmethod
    def _setup_experiment(cls, work_dir: str, exp_dir: str, config_path: str, cfg: Dict[str, Any]):
        config_name = os.path.split(config_path)[-1]
        os.makedirs(exp_dir, exist_ok=True)
        # save the resolved config to the experiment directory
        with open(os.path.join(exp_dir, config_name), "w", encoding="utf8") as f:
            f.write(yaml.safe_dump(cfg))
        # make a backup of the raw, unresolved configs in the config directory as zip
        with zipfile.ZipFile(os.path.join(exp_dir, "configs.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
            root = os.path.dirname(config_path)
            for config_dir, _, files in os.walk(root):
                for file in files:
                    rel_sub_dir = os.path.relpath(config_dir, root)
                    if not file.endswith(".yaml"):
                        continue
                    zf.write(os.path.join(config_dir, file),
                             os.path.join(rel_sub_dir, file))
        with open(os.path.join(exp_dir, "info.yaml"), "w", encoding="utf8") as f:
            f.write(
                yaml.safe_dump({
                    "config_name": config_name,
                    "git": {
                        "branch": api.git_branch(work_dir),
                        "commit": api.git_commit(work_dir)
                    }
                })
            )
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "tensorboard"), exist_ok=True)

    @classmethod
    def _train_local_distributed(
        cls,
        rank: int,
        world_size: int,
        port: int,
        cfg: Dict[str, Any],
        directories: Dict[str, str],
        profile: Optional[str] = None
    ):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)

        dist.init_process_group(
            backend=dist.Backend.NCCL,
            init_method="env://",
            rank=rank,
            world_size=world_size
        )

        info = distributed.DistributedInfo(
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            local_world_size=world_size
        )
        torch.cuda.set_device(info.device)
        cuda.matmul.allow_tf32 = True
        cudnn.allow_tf32 = True

        assert dist.is_initialized(), "failed to initialize process group"

        if info.is_main_process and profile is not None:
            import cProfile
            cProfile.runctx(
                "cls(cfg, directories, info).run()",
                globals(),
                locals(),
                filename=profile
            )
        else:
            cls(cfg, directories, info).run()
        dist.destroy_process_group()

    @classmethod
    def train_slurm(cls, work_dir: str, experiment_dir: str, config_path: str):
        assert torch.cuda.device_count() > 0, "need at least one GPU for training, but found none"
        assert dist.is_available(), "distributed package must be available for training"
        assert dist.is_nccl_available(), "nccl backend for distributed training must be available"
        logger = logging.get_logger("SLURM_INITIALIZATION")
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} GPU{'s' * (num_gpus > 1)} "
                    f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")

        assert (
            "MASTER_ADDR" in os.environ
            and "MASTER_PORT" in os.environ
            and "WORLD_SIZE" in os.environ
        ), "could not find at least one of MASTER_ADDR, MASTER_PORT and WORLD_SIZE env variables"
        master_addr = os.environ["MASTER_ADDR"]
        master_port = int(os.environ["MASTER_PORT"])
        world_size = int(os.environ["WORLD_SIZE"])

        assert "SLURM_PROCID" in os.environ, "distributed training across multiple nodes is only supported with SLURM"
        rank = int(os.environ["SLURM_PROCID"])
        local_world_size = int(os.environ.get(
            "SLURM_NTASKS_PER_NODE", os.environ["SLURM_NTASKS"]))
        local_rank = rank % local_world_size
        logger.info(
            f"Running on Slurm Cluster: master_addr={master_addr}, master_port={master_port}, "
            f"rank={rank}, local_rank={local_rank}, world_size={world_size}, local_world_size={local_world_size}"
        )

        dist.init_process_group(
            backend=dist.Backend.NCCL,
            init_method="env://",
            rank=rank,
            world_size=world_size
        )

        info = distributed.DistributedInfo(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size
        )
        torch.cuda.set_device(info.device)
        cuda.matmul.allow_tf32 = True
        cudnn.allow_tf32 = True

        assert dist.is_initialized(), "failed to initialize process group"

        resuming = os.path.exists(experiment_dir) and os.path.exists(
            os.path.join(experiment_dir, "checkpoints", "checkpoint_last.pt")
        )
        if not resuming:
            assert config_path is not None, "specify config if not resuming an existing experiment"
            cfg = configuration.load_config(config_path)
            if info.is_main_process:
                cls._setup_experiment(
                    work_dir,
                    experiment_dir,
                    config_path,
                    cfg
                )
                logger.info(
                    f"Starting experiment at {experiment_dir} with config:\n{yaml.safe_dump(cfg)}"
                )
        else:
            cfg = configuration.load_config_from_experiment(experiment_dir)
            if info.is_main_process:
                logger.info(
                    f"Resuming from {experiment_dir} with config:\n{yaml.safe_dump(cfg)}"
                )

        directories = {
            "experiment": experiment_dir,
            "checkpoints": os.path.join(experiment_dir, "checkpoints"),
            "tensorboard": os.path.join(experiment_dir, "tensorboard")
        }

        cls(cfg, directories, info).run()
        dist.destroy_process_group()

    @classmethod
    def train_local(cls, work_dir: str, experiment_dir: str, config_path: str, profile: Optional[str] = None):
        logger = logging.get_logger("LOCAL_INITIALIZATION")
        num_gpus = torch.cuda.device_count()
        assert num_gpus > 0, "need at least one GPU for local training"
        # start local distributed training
        port = int(os.environ.get("MASTER_PORT", random.randint(10000, 60000)))
        resuming = os.path.exists(experiment_dir) and os.path.exists(
            os.path.join(experiment_dir, "checkpoints", "checkpoint_last.pt")
        )
        if not resuming:
            cfg = configuration.load_config(config_path)
            assert config_path is not None, "specify config if not resuming an existing experiment"
            cls._setup_experiment(work_dir, experiment_dir, config_path, cfg)
            logger.info(
                f"Starting experiment at {experiment_dir} with config:\n{yaml.safe_dump(cfg)}"
            )
        else:
            cfg = configuration.load_config_from_experiment(experiment_dir)
            logger.info(
                f"Resuming from {experiment_dir} with config:\n{yaml.safe_dump(cfg)}"
            )
        directories = {
            "experiment": experiment_dir,
            "checkpoints": os.path.join(experiment_dir, "checkpoints"),
            "tensorboard": os.path.join(experiment_dir, "tensorboard")
        }
        mp.spawn(
            fn=cls._train_local_distributed,
            nprocs=num_gpus,
            args=(num_gpus, port, cfg, directories, profile),
            join=True
        )

    def _prepare_batch(
        self,
        batch: data.DataBatch,
        train: bool = True,
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        assert len(batch) > 0, "got empty batch"
        token_ids_np, pad_mask_np, lengths, info, labels_np, _ = batch.tensors()
        inputs = {
            "token_ids": torch.from_numpy(token_ids_np).to(
                non_blocking=True,
                device=self.info.device
            ),
            "lengths": lengths,
            "padding_mask": torch.from_numpy(pad_mask_np).to(
                non_blocking=True,
                device=self.info.device
            ),
            **api.to(info, self.info.device)
        }
        labels = torch.from_numpy(labels_np).to(
            non_blocking=True,
            dtype=torch.long,
            device=self.info.device
        )
        return inputs, labels

    def _train_one_epoch(self):
        begin_of_epoch = time.perf_counter()
        start = time.perf_counter()

        mean_loss = tensorboard.DistAverageTracker(
            "train_loss",
            self.info.device,
            fmt=".2e"
        )
        mean_fwdbwd_pass = tensorboard.DistAverageTracker(
            "train_forward_backward_pass",
            self.info.device
        )
        mean_batch_load = tensorboard.DistAverageTracker(
            "train_batch_load",
            self.info.device
        )
        mean_step_time = tensorboard.DistAverageTracker(
            "train_step_time",
            self.info.device
        )
        mean_batch_preparation = tensorboard.DistAverageTracker(
            "train_batch_preparation",
            self.info.device
        )
        mean_bsz = tensorboard.DistAverageTracker(
            "train_batch_size",
            self.info.device,
        )
        mean_seq_length = tensorboard.DistAverageTracker(
            "train_sequence_length",
            self.info.device
        )
        mean_seq_length_ratio = tensorboard.DistAverageTracker(
            "train_sequence_length_ratio",
            self.info.device
        )

        metric_cfg = self.cfg["train"].get("metrics")
        if metric_cfg is not None:
            metrics = tensorboard.metrics_from_config(
                metric_cfg,
                self.input_tokenizer,
                self.output_tokenizer,
                prefix="train"
            )
        else:
            metrics = []

        dist_items = torch.zeros(1, dtype=torch.long, device=self.info.device)
        start_items = self.epoch_items
        self.model = self.model.train()

        train_iter = iter(self.train_loader)
        while True:
            start_batch = time.perf_counter()
            batch = next(train_iter, None)
            end_batch = time.perf_counter()
            if batch is None:
                self.logger.info(
                    f"[rank {self.info.rank}] finished epoch {self.epoch + 1}"
                )
                break
            elif len(batch) == 0:
                raise RuntimeError(
                    "got empty batch, this should not happen during training"
                )

            start_preparation = time.perf_counter()
            inputs, labels = self._prepare_batch(batch, train=True)
            end_preparation = time.perf_counter()

            self.optimizer.zero_grad(set_to_none=True)

            start_fwdbwd = time.perf_counter()
            with torch.autocast(
                "cuda",
                dtype=self.precision_dtype,
                enabled=self.precision_dtype != torch.float32
            ):
                outputs, loss_dict = self.model(**inputs)
                loss = self.loss_fn(outputs, labels)
                loss = loss + sum(loss_dict.values())

            mean_loss.add(loss.item())
            self.grad_scaler.scale(loss).backward()
            end_fwdbwd = time.perf_counter()

            if self.clip_grad_norm is not None:
                self.grad_scaler.unscale_(self.optimizer)
                if isinstance(self.model, FSDP):
                    self.model.clip_grad_norm_(self.clip_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_grad_norm
                    )

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            self.total_step += 1
            self.epoch_step += 1
            dist_items[0] = len(batch)
            dist.all_reduce(dist_items, dist.ReduceOp.SUM)
            batch_items = dist_items[0].item()
            self.epoch_items += batch_items
            self.total_items += batch_items

            if self.total_items >= self.step_at:
                lr_scheduler = self.cooldown_scheduler or self.lr_scheduler
                if lr_scheduler is not None:
                    lr_scheduler.step()
                self.step_at += self.step_interval

            if self.max_length_scheduler is not None:
                max_length = self.max_length_scheduler(self.total_items)
                if max_length != self.max_length:
                    self.max_length = max_length
                    self.train_loader.set_max_length(max_length)
                    self.val_loader.set_max_length(max_length)

            mean_step_time.add((time.perf_counter() - start_batch) * 1000)
            mean_fwdbwd_pass.add((end_fwdbwd - start_fwdbwd) * 1000)
            mean_bsz.add(batch_items)
            min_length = sys.maxsize
            max_length = 0
            for length in inputs["lengths"]:
                mean_seq_length.add(length)
                if length < min_length:
                    min_length = length
                if length > max_length:
                    max_length = length
            mean_batch_load.add((end_batch - start_batch) * 1000)
            mean_batch_preparation.add(
                (end_preparation - start_preparation) * 1000
            )
            mean_seq_length_ratio.add(max_length / max(1, min_length))

            if self.total_items >= self.log_at:
                mean_loss.sync()
                mean_bsz.sync()
                mean_step_time.sync()
                mean_fwdbwd_pass.sync()
                mean_batch_load.sync()
                mean_seq_length.sync()
                mean_seq_length_ratio.sync()
                mean_batch_preparation.sync()
                end = time.perf_counter()

                if self.info.is_main_process:
                    # log training progress only on main process
                    assert self.summary_writer is not None

                    progress = 100 * self.total_items / self.training_items
                    self.summary_writer.add_scalar(
                        "train_progress",
                        progress,
                        self.total_step
                    )
                    self.logger.info(
                        f"[step {self.total_step}] "
                        f"train_progress: {progress:.2f}%, "
                        f"{self.total_items:,} / {self.training_items:,} items on this rank"
                    )

                    lr_scheduler = self.cooldown_scheduler or self.lr_scheduler
                    if lr_scheduler is not None:
                        for i, lr in enumerate(lr_scheduler.get_last_lr()):
                            self.summary_writer.add_scalar(
                                f"train_lr_{i}", lr, self.total_step
                            )
                            self.logger.info(
                                f"[step {self.total_step}] train_lr_{i}: {lr:.8f}"
                            )

                    mean_loss.log_tensorboard(self.summary_writer, self.total_step)
                    mean_loss.log_info(self.logger, self.total_step)

                    mean_bsz.log_tensorboard(self.summary_writer, self.total_step)
                    mean_bsz.log_info(self.logger, self.total_step)

                    mean_fwdbwd_pass.log_tensorboard(
                        self.summary_writer, self.total_step
                    )
                    mean_fwdbwd_pass.log_info(self.logger, self.total_step)

                    mean_batch_load.log_tensorboard(
                        self.summary_writer, self.total_step)
                    mean_batch_load.log_info(self.logger, self.total_step)

                    mean_step_time.log_tensorboard(
                        self.summary_writer, self.total_step)
                    mean_step_time.log_info(self.logger, self.total_step)

                    mean_batch_preparation.log_tensorboard(
                        self.summary_writer, self.total_step
                    )
                    mean_batch_preparation.log_info(self.logger, self.total_step)

                    mean_seq_length.log_tensorboard(
                        self.summary_writer, self.total_step)
                    mean_seq_length.log_info(self.logger, self.total_step)

                    mean_seq_length_ratio.log_tensorboard(
                        self.summary_writer, self.total_step
                    )
                    mean_seq_length_ratio.log_info(self.logger, self.total_step)

                    items = batch.items()
                    for metric in metrics:
                        metric.set_values(items, outputs)
                        metric.log_tensorboard(
                            self.summary_writer,
                            self.total_step
                        )
                        metric.log_info(self.logger, self.total_step)

                    self.logger.info(
                        f"[step {self.total_step}] train_time for ~{self.log_interval:,} items: "
                        f"{(end - start) / 60:.2f} minutes"
                    )
                    eta_msg = logging.eta_minutes_message(
                        (end - begin_of_epoch) / 60,
                        self.epoch_items - start_items,
                        self.training_items_per_epoch - start_items
                    )
                    self.logger.info(
                        f"[step {self.total_step}] [epoch {self.epoch + 1}] {eta_msg}"
                    )

                if self.info.is_local_main_process:
                    self.logger.info(
                        f"[step {self.total_step}] [rank {self.info.rank}] nvidia-smi:\n"
                        f"{api.nvidia_smi()}"
                    )

                start = end
                mean_loss.reset()
                mean_bsz.reset()
                mean_step_time.reset()
                mean_fwdbwd_pass.reset()
                mean_batch_load.reset()
                mean_seq_length.reset()
                mean_seq_length_ratio.reset()
                mean_batch_preparation.reset()
                self.log_at += self.log_interval

            if self.cooldown_items > 0 and self.total_items >= self.eval_at - self.cooldown_items:
                self._start_cooldown()

            if self.total_items >= self.eval_at:
                # evaluation is done distributed
                self._evaluate_and_checkpoint()
                # benchmarking needs to be properly implemented by subclasses
                self._benchmark_and_checkpoint()

                if self.cooldown_items > 0:
                    if self.info.is_main_process:
                        self.logger.info(
                            f"[step {self.total_step}] resetting to "
                            f"to checkpoint before cooldown"
                        )
                    # stop cooldown
                    self._stop_cooldown()

                    # trigger train loader again
                    train_iter = iter(self.train_loader)

                    # reset the statistics
                    mean_loss.reset()
                    mean_bsz.reset()
                    mean_step_time.reset()
                    mean_fwdbwd_pass.reset()
                    mean_batch_load.reset()
                    mean_seq_length.reset()
                    mean_seq_length_ratio.reset()
                    mean_batch_preparation.reset()
                    start = time.perf_counter()

                self.eval_at += self.eval_interval

    def _evaluate_and_checkpoint(self):
        mean_loss = tensorboard.DistAverageTracker(
            "val_loss",
            self.info.device,
            fmt=".2e"
        )

        self.model = self.model.eval()
        self.loss_fn = self.loss_fn.eval()

        metric_cfg = self.cfg["val"].get("metrics")
        if metric_cfg is not None:
            metrics = tensorboard.metrics_from_config(
                metric_cfg,
                self.input_tokenizer,
                self.output_tokenizer,
                prefix="val"
            )
        else:
            metrics = []

        start = time.perf_counter()
        for batch_num, batch in enumerate(self.val_loader):
            inputs, labels = self._prepare_batch(batch, train=False)

            with torch.autocast(
                "cuda",
                dtype=self.precision_dtype,
                enabled=self.precision_dtype != torch.float32
            ), torch.no_grad():
                outputs, loss_dict = self.model(**inputs)
                loss = self.loss_fn(outputs, labels)
                loss = loss + sum(loss_dict.values())

            mean_loss.add(loss.item())

            if batch_num == 0 and self.info.is_main_process:
                items = batch.items()
                for metric in metrics:
                    metric.set_values(items, outputs)
                    metric.log_tensorboard(
                        self.summary_writer,
                        self.total_step
                    )
                    metric.log_info(self.logger, self.total_step)

        end = time.perf_counter()
        mean_loss.sync()

        # only log on main process
        if self.info.is_main_process:
            assert self.summary_writer is not None

            mean_loss.log_tensorboard(self.summary_writer, self.total_step)
            mean_loss.log_info(self.logger, self.total_step)

            self.logger.info(
                f"[step {self.total_step}] validation took {(end - start) / 60:.2f} minutes"
            )

        ckpt_path = os.path.join(
            self.directories["checkpoints"],
            "checkpoint_last.pt"
        )
        val_loss = mean_loss.value
        self._save_checkpoint(ckpt_path, val_loss)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_ckpt_path = os.path.join(
                self.directories["checkpoints"],
                "checkpoint_best.pt"
            )
            self._save_checkpoint(best_ckpt_path, val_loss, full=False)

        self.model = self.model.train()
        self.loss_fn = self.loss_fn.train()

    def _benchmark_and_checkpoint(self):
        pass

    def _start_cooldown(self):
        # already started
        if self.cooldown_scheduler is not None:
            return
        # cooldown scheduler linearly decays lr from
        # current value to 0
        if self.lr_scheduler is not None:
            factor = self.lr_scheduler.get_last_lr()[0] / self.cfg["train"]["optimizer"]["lr"]
        else:
            factor = 1.0
        steps = max(1, min(self.cooldown_items, self.eval_at - self.total_items) // self.step_interval)
        self.cooldown_scheduler = lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: (1 - (min(step, steps) / steps)) * factor
        )
        path = os.path.join(
            self.directories["checkpoints"],
            "cooldown_checkpoint.pt"
        )
        self._save_checkpoint(path, self.best_val_loss)

    def _stop_cooldown(self):
        # already stopped
        if self.cooldown_scheduler is None:
            return
        self.cooldown_scheduler = None
        path = os.path.join(
            self.directories["checkpoints"],
            "cooldown_checkpoint.pt"
        )
        # load cooldown checkpoint, but pay special attention
        # to best val loss, because it is reset in self._load_checkpoint
        val_loss = self.best_val_loss
        self._load_checkpoint(path)
        self.best_val_loss = val_loss

        if self.info.is_main_process:
            os.remove(path)

    def run(self):
        try:
            while self.epoch < self.cfg["train"]["num_epochs"]:
                self.train_loader.set_epoch(self.epoch)
                self.train_loader.set_fast_forward(self.epoch_items)

                self._train_one_epoch()

                self.epoch += 1
                self.epoch_step = 0
                self.epoch_items = 0

        except KeyboardInterrupt:
            if self.info.is_main_process:
                self.logger.info(
                    "got termination signal, saving on main process before exiting"
                )

        finally:
            start = time.perf_counter()
            ckpt_path = os.path.join(
                self.directories["checkpoints"],
                "checkpoint_last.pt"
            )
            self._save_checkpoint(ckpt_path, self.best_val_loss)
            end = time.perf_counter()
            if self.info.is_main_process:
                self.logger.info(
                    f"final checkpointing took {end - start:.2f}s"
                )
                if len(self.cleanup) > 0:
                    self.logger.info(
                        f"deleting temporary data sources on local main process with rank {self.info.rank}"
                    )
                    for path in self.cleanup:
                        os.remove(path)
