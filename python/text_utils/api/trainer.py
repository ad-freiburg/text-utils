import argparse
import copy
import functools
import hashlib
import math
import os
import random
import shutil
import sys
import time
import zipfile
from logging import INFO
from typing import Any, Callable

import torch
import yaml
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import nn
from torch.backends import cuda, cudnn  # noqa
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from torch.utils.tensorboard.writer import SummaryWriter

from text_utils import api, configuration, data, distributed, io, logging, tensorboard
from text_utils.api.utils import get_gradient_clipper
from text_utils.modules.loss import loss_from_config
from text_utils.modules.optimizer import optimizer_from_config
from text_utils.modules.scheduler import (
    lr_scheduler_from_config,
    max_length_scheduler_from_config,
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
training will resume from latest checkpoint.",
        )
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            default=None,
            help="Path to config file, only required for a new training run.",
        )
        parser.add_argument(
            "-p",
            "--platform",
            type=str,
            choices=["local", "slurm"],
            default="local",
            required=True,
            help="Platform used for training.",
        )
        parser.add_argument(
            "--profile",
            action="store_true",
            help="Run cProfile profile on main process and output stats to 'profile.pstat' "
            "in experiment directory (only works for platform=local)",
        )
        return parser

    def __init__(
        self,
        cfg: dict[str, Any],
        directories: dict[str, str],
        info: distributed.DistributedInfo,
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
        self.best_benchmark: float | None = None
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

        model = self._model_from_config(self.cfg)

        peft = self.cfg["train"].get("peft", None)
        if peft is not None:
            if self.info.is_main_process:
                self.logger.info(
                    "preparing model for parametering efficient fine tuning with config:\n"
                    f"{yaml.safe_dump(peft)}"
                )
            model = self._prepare_peft(model, peft)

        mixed_precision = self.cfg["train"].get("mixed_precision", None)
        if mixed_precision == "fp16":
            self.mixed_precision = torch.float16
        elif mixed_precision == "bfp16":
            self.mixed_precision = torch.bfloat16
        elif mixed_precision is not None:
            raise ValueError(f"unknown mixed precision {mixed_precision}")
        else:
            self.mixed_precision = None

        precision = self.cfg["train"].get("precision", None)
        if precision == "fp16":
            model = model.to(torch.float16)
        elif precision == "bfp16":
            model = model.to(torch.bfloat16)
        elif precision == "fp32":
            model = model.to(torch.float32)
        elif precision is not None:
            raise ValueError(f"unknown precision {precision}")

        compile = self.cfg["train"].get("compile", False)
        dist_cfg = self.cfg["train"].get("distributed", {})
        dist_type = dist_cfg["type"]
        assert dist_type in {
            "DDP",
            "FSDP",
        }, f"distributed training type must be either DDP or FSDP, but got {dist_type}"

        if self.info.is_single_gpu:
            self.model = model.to(self.info.device)
            dist_type = "single GPU"
        elif dist_type == "DDP":
            self.model = DDP(
                model.to(self.info.device),
                static_graph=compile,
                gradient_as_bucket_view=True,
            )
        else:
            sharding_policy = self._sharding_policy(model)
            offload_params = dist_cfg.get("offload", False)
            prefetch = dist_cfg.get("prefetch", True)
            strategy = ShardingStrategy[dist_cfg.get("strategy", "NO_SHARD")]
            if strategy != ShardingStrategy.NO_SHARD:
                shard_size = dist_cfg.get("shard_size", None)
                if sharding_policy is not None:
                    if self.info.is_main_process:
                        self.logger.info("sharding based on custom policy")
                elif shard_size is not None:
                    if self.info.is_main_process:
                        self.logger.info(
                            f"sharding based on number of parameters with "
                            f"a minimum of {shard_size:,}"
                        )
                    sharding_policy = functools.partial(
                        size_based_auto_wrap_policy,
                        min_num_params=shard_size,
                        force_leaf_modules=None,
                        exclude_wrap_modules=None,
                    )
                else:
                    raise ValueError(
                        "sharding strategy is set, but no custom sharding policy "
                        "or shard size is specified"
                    )
            else:
                sharding_policy = None
                offload_params = False

            self.model = FSDP(
                model,
                auto_wrap_policy=sharding_policy,
                mixed_precision=MixedPrecision(
                    param_dtype=self.mixed_precision, reduce_dtype=self.mixed_precision
                ),
                cpu_offload=CPUOffload(offload_params=offload_params),
                limit_all_gathers=True,
                sharding_strategy=strategy,
                forward_prefetch=prefetch,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE
                if prefetch
                else BackwardPrefetch.BACKWARD_POST,
                device_id=self.info.device,
            )
            # set mixed precision to none here for FSDP to avoid autocasting
            # later, because FSDP handles mixed precision itself
            self.mixed_precision = None

            FSDP.set_state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
            )

        self.model: nn.Module | DDP | FSDP = torch.compile(
            self.model, fullgraph=True, disable=not compile
        )  # type: ignore

        self.optimizer = optimizer_from_config(
            self.model,  # type: ignore
            self.cfg["train"]["optimizer"],
            additional_optimizer_fn=self._additional_optimizer_fn(),
        )

        gradient_clipping_cfg = self.cfg["train"].get("gradient_clipping", None)
        if gradient_clipping_cfg:
            self.gradient_clipper = get_gradient_clipper(gradient_clipping_cfg)
        else:
            self.gradient_clipper = None

        gradient_accumulation_cfg = self.cfg["train"].get("gradient_accumulation", {})
        self.gradient_accumulation_steps = gradient_accumulation_cfg.get("steps", 1)
        self.gradient_accumulation_reduction = gradient_accumulation_cfg.get(
            "reduction", "mean"
        )
        assert self.gradient_accumulation_reduction in {"mean", "sum"}, (
            "gradient accumulation reduction must be either mean or sum, "
            f"but got {self.gradient_accumulation_reduction}"
        )

        self.grad_scaler = ShardedGradScaler(
            enabled=self.mixed_precision == torch.float16,
        )

        num_epochs = self.cfg["train"]["num_epochs"]
        (
            self.train_loader,
            self.val_loader,
            self.training_items_per_epoch,
            self.training_items,
            self.max_length,
            self.max_length_scheduler,
            self.cleanup,
        ) = self._data_from_config(
            self.cfg["train"]["data"],
            self.cfg["val"]["data"],
            num_epochs=num_epochs,
            seed=self.cfg["seed"],
            info=self.info,
        )

        lower = self.cfg["train"]["data"]["batch_limit"]
        if self.cfg["train"]["data"]["batch_limit_type"] != "batch_size":
            lower = lower // self.cfg["train"]["data"]["max_length"]
        lower = lower * self.gradient_accumulation_steps

        self.log_interval = clamp(
            self.training_items * self.cfg["train"].get("log_interval", 0.001),
            lower,
            self.training_items,
        )
        self.eval_interval = clamp(
            self.training_items * self.cfg["train"].get("eval_interval", 0.1),
            lower,
            self.training_items,
        )

        cooldown = self.cfg["val"].get("cooldown", 0)
        if isinstance(cooldown, float):
            cooldown_items = self.training_items * cooldown
        elif isinstance(cooldown, int):
            cooldown_items = cooldown
        else:
            raise ValueError(
                f"cooldown must be a float between 0 and 1 or an int, but got {cooldown}"
            )
        if cooldown_items > 0:
            self.cooldown_items = clamp(cooldown_items, lower, self.training_items)
            assert self.cooldown_items < self.eval_interval, (
                f"cooldown items {self.cooldown_items:,} must be smaller "
                f"than evaluation interval {self.eval_interval:,}"
            )
        else:
            self.cooldown_items = 0

        self.cooldown_scheduler: lr_scheduler.LambdaLR | None = None

        lr_scheduler_cfg = self.cfg["train"].get("lr_scheduler", None)
        if lr_scheduler_cfg is not None:
            self.step_interval = clamp(
                self.training_items * lr_scheduler_cfg.pop("step_interval", 0.001),
                lower,
                self.training_items,
            )
            steps = self.training_items // self.step_interval
            self.lr_scheduler = lr_scheduler_from_config(
                self.optimizer,
                steps,
                self.info.world_size,
                lr_scheduler_cfg,
                additional_lr_scheduler_fn=self._additional_lr_scheduler_fn(),
            )
        else:
            self.step_interval = 0
            self.lr_scheduler = None

        self.log_at = self.log_interval
        self.eval_at = self.eval_interval
        self.step_at = self.step_interval

        self.loss_fn = (
            loss_from_config(
                self.cfg["train"]["loss"], additional_loss_fn=self._additional_loss_fn()
            )
            .to(self.info.device)
            .train()
        )

        if self.info.is_main_process:
            self.summary_writer = SummaryWriter(log_dir=self.directories["tensorboard"])

            self.logger.info(f"Using model:\n{self.model}")
            self.logger.info(f"Model parameters: {api.num_parameters(self.model)}")
            num_params = 0
            param_group_infos = []
            for i, param_group in enumerate(self.optimizer.param_groups):
                group_num_params = sum(p.numel() for p in param_group["params"])
                group_cfg = {k: v for k, v in param_group.items() if k != "params"}
                param_group_infos.append(
                    f"{i+1}. group: {group_num_params:,} params, other: {group_cfg}"
                )
                num_params += group_num_params
            param_group_info = "\n".join(param_group_infos)
            self.logger.info(f"Optimizer parameter groups:\n{param_group_info}")
            self.logger.info(
                f"Training with {dist_type} and {mixed_precision or 'no'} mixed precision"
            )
            self.logger.info(
                f"Number of training items: {self.training_items_per_epoch:,} per epoch, "
                f"{self.training_items:,} total"
            )
            self.logger.info(
                f"Logging every {self.log_interval:,} items, "
                f"evaluating every {self.eval_interval:,} items"
                + f", stepping every {self.step_interval:,} items"
                if self.lr_scheduler is not None
                else ""
            )

            self.logger.info(
                f"Type 'tensorboard --logdir {self.directories['tensorboard']}' "
                f"to view the training process in Tensorboard"
            )
        else:
            self.summary_writer = None

        # resume training from last checkpoint if it exists
        last_checkpoint = os.path.join(
            self.directories["checkpoints"], "checkpoint_last.pt"
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
                checkpoint["model_state_dict"], strict=False
            )
            assert (
                len(wrong_keys.unexpected_keys) == 0
            ), f'unexpected keys in checkpoint "{load_checkpoint}": {wrong_keys.unexpected_keys}'

            self.logger.info(
                f'initializing model from checkpoint "{load_checkpoint}" '
                f"(missing keys: {wrong_keys.missing_keys})"
            )

    def _save_checkpoint(
        self, path: str, val_loss: float, full: bool = True, **kwargs: Any
    ):
        # need to call this on all processes because model and optimizer
        # might be distributed across processes
        save = {
            "checkpoint_path": path,
            "model_state_dict": distributed.unwrap_model(self.model).state_dict(),
            "step": self.total_step,
            "epoch": self.epoch,
            "epoch_step": self.epoch_step,
            "epoch_items": self.epoch_items,
            "total_items": self.total_items,
            "val_loss": val_loss,
            **kwargs,
        }
        if full:
            save["optimizer_state_dict"] = distributed.get_optimizer_state_dict(
                self.model, self.optimizer
            )
            save["grad_scaler_state_dict"] = self.grad_scaler.state_dict()
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
        distributed.unwrap_model(self.model).load_state_dict(
            checkpoint["model_state_dict"]
        )
        optim_state_dict = checkpoint["optimizer_state_dict"]
        if isinstance(self.model, FSDP):
            optim_state_dict = FSDP.optim_state_dict_to_load(
                self.model,
                self.optimizer,
                optim_state_dict,
            )
        self.optimizer.load_state_dict(optim_state_dict)
        self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        if (
            self.lr_scheduler is not None
            and checkpoint.get("lr_scheduler_state_dict") is not None
        ):
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        if checkpoint.get("loss_fn_state_dict") is not None:
            self.loss_fn.load_state_dict(checkpoint["loss_fn_state_dict"])

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
        self.log_at = (
            math.ceil(self.total_items / self.log_interval) * self.log_interval
        )
        self.eval_at = (
            math.ceil(self.total_items / self.eval_interval) * self.eval_interval
        )
        self.step_at = (
            math.ceil(self.total_items / self.step_interval) * self.step_interval
        )

        # wait until everyone loaded the checkpoint
        dist.barrier()

    @classmethod
    def _prepare_peft(
        cls,
        model: nn.Module,
        cfg: dict[str, Any],
    ) -> nn.Module:
        raise NotImplementedError

    @classmethod
    def _model_from_config(cls, cfg: dict[str, Any]) -> nn.Module:
        raise NotImplementedError

    @classmethod
    def _sharding_policy(
        cls,
        model: nn.Module,
    ) -> ShardingPolicy | None:
        return None

    @classmethod
    def _additional_loss_fn(cls) -> Callable | None:
        return None

    @classmethod
    def _additional_optimizer_fn(cls) -> Callable | None:
        return None

    @classmethod
    def _additional_lr_scheduler_fn(cls) -> Callable | None:
        return None

    @classmethod
    def _additional_max_length_scheduler_fn(cls) -> Callable | None:
        return None

    @classmethod
    def _metric_from_config(
        cls, cfg: dict[str, Any], prefix: str
    ) -> tensorboard.TensorboardMetric:
        raise NotImplementedError("metric from config not implemented")

    @classmethod
    def _copy_file_to_tmp_dir(
        cls, path: str, dir: str, info: distributed.DistributedInfo
    ) -> str:
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
        sources: list[dict[str, Any]],
        info: distributed.DistributedInfo,
    ) -> tuple[list[str], list[str]]:
        src_paths = []
        cleanup_paths = []
        for src in sources:
            src = copy.deepcopy(src)
            src_type = src.pop("type")
            if src_type == "file":
                path = src["path"]
                assert os.path.isfile(path), f"{path} is not a file"
                temp_dir = src.get("temp_dir")
                if temp_dir is not None:
                    path = cls._copy_file_to_tmp_dir(path, temp_dir, info)
                    cleanup_paths.append(path)
                src_paths.append(path)

            elif src_type == "file_glob":
                temp_dir = src.get("temp_dir")
                for path in io.glob_safe(src["glob"]):
                    assert os.path.isfile(path), f"{path} is not a file"
                    if temp_dir is not None:
                        path = cls._copy_file_to_tmp_dir(path, temp_dir, info)
                        cleanup_paths.append(path)
                    src_paths.append(path)

            else:
                raise ValueError(f"Unknown source type {src_type}")

        assert len(src_paths) > 0, "Got no data sources"
        return src_paths, cleanup_paths

    @classmethod
    def _data_from_config(
        cls,
        train_cfg: dict[str, Any],
        val_cfg: list[Any] | int,
        num_epochs: int,
        seed: int,
        info: distributed.DistributedInfo,
    ) -> tuple[
        data.TrainLoader,
        data.TrainLoader,
        int,
        int,
        int,
        Callable[[int], int] | None,
        list[str],
    ]:
        train_cfg = copy.deepcopy(train_cfg)

        # adapt config to multi gpu usage
        assert "batch_limit" in train_cfg, "batch_limit must be in data config"
        train_cfg["batch_limit"] = max(1, train_cfg["batch_limit"] // info.world_size)

        # pop some configs not used by the dataloader
        max_length = train_cfg.pop("max_length")
        assert max_length is not None, "missing max_length in data config"
        max_length_scheduler_cfg = train_cfg.pop("max_length_scheduler", None)

        train_files, cleanup = cls._prepare_data_sources(train_cfg.pop("sources"), info)

        pipeline_cfg = train_cfg.pop("pipeline")

        # copy over some options from the training config
        val_options = {
            field: train_cfg[field]
            for field in [
                "batch_limit",
                "batch_limit_type",
                "num_threads",
                "buffer_size",
            ]
            if field in train_cfg
        }

        # for validation always turn off shuffling and turn on sorting
        if isinstance(val_cfg, int):
            # if validation is a split of the training set
            train_limit = train_cfg.get("limit", None)
            if train_limit is not None:
                assert train_limit > val_cfg, (
                    f"train limit ({train_limit:,}) cannot be smaller or "
                    f"equal to val limit ({val_cfg:,})"
                )

            train_loader = data.TrainLoader.from_files(
                train_files,
                pipeline_cfg,
                skip=val_cfg,
                seed=seed,
                max_length=max_length,
                distributed=(info.rank, info.world_size),
                **train_cfg,
            )
            # specify the val limit
            val_loader = data.TrainLoader.from_files(
                train_files,
                pipeline_cfg,
                limit=val_cfg,
                seed=seed,
                max_length=max_length,
                distributed=(info.rank, info.world_size),
                **val_options,
            )

        else:
            # if validation is a separate set of data sources
            train_loader = data.TrainLoader.from_files(
                train_files,
                pipeline_cfg,
                seed=seed,
                max_length=max_length,
                distributed=(info.rank, info.world_size),
                **train_cfg,
            )
            val_files, val_cleanup = cls._prepare_data_sources(val_cfg, info)
            cleanup.extend(val_cleanup)
            val_loader = data.TrainLoader.from_files(
                val_files,
                pipeline_cfg,
                seed=seed,
                max_length=max_length,
                distributed=(info.rank, info.world_size),
                **val_options,
            )

        # trigger train loader, so that min_items is set
        iter(train_loader)
        training_items_per_epoch = train_loader.min_items
        training_items = training_items_per_epoch * num_epochs

        if max_length_scheduler_cfg is not None:
            max_length_scheduler = max_length_scheduler_from_config(
                training_items,
                max_length,
                max_length_scheduler_cfg,
                additional_max_length_scheduler_fn=cls._additional_max_length_scheduler_fn(),
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
            cleanup,
        )

    @classmethod
    def _setup_experiment(
        cls, work_dir: str, exp_dir: str, config_path: str, cfg: dict[str, Any]
    ):
        config_name = os.path.split(config_path)[-1]
        os.makedirs(exp_dir, exist_ok=True)
        # save the resolved config to the experiment directory
        with open(os.path.join(exp_dir, config_name), "w", encoding="utf8") as f:
            f.write(yaml.safe_dump(cfg))
        # make a backup of the raw, unresolved configs in the config directory as zip
        with zipfile.ZipFile(
            os.path.join(exp_dir, "configs.zip"), "w", zipfile.ZIP_DEFLATED
        ) as zf:
            root = os.path.dirname(config_path)
            for config_dir, _, files in os.walk(root):
                for file in files:
                    rel_sub_dir = os.path.relpath(config_dir, root)
                    if not file.endswith(".yaml"):
                        continue
                    zf.write(
                        os.path.join(config_dir, file), os.path.join(rel_sub_dir, file)
                    )
        with open(os.path.join(exp_dir, "info.yaml"), "w", encoding="utf8") as f:
            f.write(
                yaml.safe_dump(
                    {
                        "config_name": config_name,
                        "git": {
                            "branch": api.git_branch(work_dir),
                            "commit": api.git_commit(work_dir),
                        },
                    }
                )
            )
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "tensorboard"), exist_ok=True)

    @classmethod
    def _train_local_distributed(
        cls,
        rank: int,
        world_size: int,
        port: int,
        cfg: dict[str, Any],
        directories: dict[str, str],
        profile: bool,
    ):
        logging.setup_logging(INFO)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)

        dist.init_process_group(
            backend=dist.Backend.NCCL,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )

        info = distributed.DistributedInfo(
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            local_world_size=world_size,
        )
        torch.cuda.set_device(info.device)
        cuda.matmul.allow_tf32 = True
        cudnn.allow_tf32 = True

        assert dist.is_initialized(), "failed to initialize process group"

        if info.is_main_process and profile:
            import cProfile

            torch.cuda.memory._record_memory_history()
            cProfile.runctx(
                "cls(cfg, directories, info).run()",
                globals(),
                locals(),
                filename=os.path.join(directories["experiment"], "profile.pstat"),
            )
            torch.cuda.memory._dump_snapshot(
                os.path.join(directories["experiment"], "memory_profile.pickle")
            )
        else:
            cls(cfg, directories, info).run()

        dist.destroy_process_group()

    @classmethod
    def train_slurm(cls, work_dir: str, experiment_dir: str, config_path: str):
        assert (
            torch.cuda.device_count() > 0
        ), "need at least one GPU for training, but found none"
        assert dist.is_available(), "distributed package must be available for training"
        assert (
            dist.is_nccl_available()
        ), "nccl backend for distributed training must be available"
        logging.setup_logging(INFO)
        logger = logging.get_logger("SLURM_INITIALIZATION")
        num_gpus = torch.cuda.device_count()
        logger.info(
            f"Found {num_gpus} GPU{'s' * (num_gpus > 1)} "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})"
        )

        assert (
            "MASTER_ADDR" in os.environ
            and "MASTER_PORT" in os.environ
            and "WORLD_SIZE" in os.environ
        ), "could not find at least one of MASTER_ADDR, MASTER_PORT and WORLD_SIZE env variables"
        master_addr = os.environ["MASTER_ADDR"]
        master_port = int(os.environ["MASTER_PORT"])
        world_size = int(os.environ["WORLD_SIZE"])

        assert (
            "SLURM_PROCID" in os.environ
        ), "distributed training across multiple nodes is only supported with SLURM"
        rank = int(os.environ["SLURM_PROCID"])
        local_world_size = int(
            os.environ.get("SLURM_NTASKS_PER_NODE", os.environ["SLURM_NTASKS"])
        )
        local_rank = rank % local_world_size
        logger.info(
            f"Running on Slurm Cluster: master_addr={master_addr}, master_port={master_port}, "
            f"rank={rank}, local_rank={local_rank}, world_size={world_size}, local_world_size={local_world_size}"
        )

        dist.init_process_group(
            backend=dist.Backend.NCCL,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )

        info = distributed.DistributedInfo(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
        )
        torch.cuda.set_device(info.device)
        cuda.matmul.allow_tf32 = True
        cudnn.allow_tf32 = True

        assert dist.is_initialized(), "failed to initialize process group"

        resuming = os.path.exists(experiment_dir) and os.path.exists(
            os.path.join(experiment_dir, "checkpoints", "checkpoint_last.pt")
        )
        if not resuming:
            assert (
                config_path is not None
            ), "specify config if not resuming an existing experiment"
            cfg = configuration.load_config(config_path)
            if info.is_main_process:
                cls._setup_experiment(work_dir, experiment_dir, config_path, cfg)
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
            "tensorboard": os.path.join(experiment_dir, "tensorboard"),
        }

        cls(cfg, directories, info).run()
        dist.destroy_process_group()

    @classmethod
    def train_local(
        cls, work_dir: str, experiment_dir: str, config_path: str, profile: bool = False
    ):
        logging.setup_logging(INFO)
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
            assert (
                config_path is not None
            ), "specify config if not resuming an existing experiment"
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
            "tensorboard": os.path.join(experiment_dir, "tensorboard"),
        }
        mp.spawn(
            fn=cls._train_local_distributed,
            args=(num_gpus, port, cfg, directories, profile),
            nprocs=num_gpus,
        )

    def _prepare_batch(
        self, batch: data.TrainBatch
    ) -> tuple[dict[str, Any], torch.Tensor]:
        raise NotImplementedError("prepare batch not implemented")

    def _train_one_epoch(self):
        begin_of_epoch = time.perf_counter()
        start = time.perf_counter()

        mean_loss = tensorboard.DistAverageTracker(
            "train_loss", self.info.device, fmt=".2e"
        )
        mean_grad_norm = tensorboard.DistAverageTracker(
            "train_grad_norm",
            self.info.device,
        )
        mean_batch_load = tensorboard.DistAverageTracker(
            "train_batch_load_time", self.info.device
        )
        mean_step_time = tensorboard.DistAverageTracker(
            "train_step_time", self.info.device
        )
        mean_batch_preparation = tensorboard.DistAverageTracker(
            "train_batch_preparation_time", self.info.device
        )
        mean_batch_size = tensorboard.DistAverageTracker(
            "train_batch_size",
            self.info.device,
        )
        mean_item_size = tensorboard.DistAverageTracker(
            "train_item_size", self.info.device
        )
        mean_item_size_ratio = tensorboard.DistAverageTracker(
            "train_item_size_ratio", self.info.device
        )
        mean_peak_gpu_memory = tensorboard.DistAverageTracker(
            "train_peak_gpu_memory", self.info.device
        )
        total_batch_size = torch.zeros(1, dtype=torch.long, device=self.info.device)
        min_num_batches = torch.zeros(1, dtype=torch.long, device=self.info.device)

        metrics = []
        for metric_cfg in self.cfg["train"].get("metrics", []):
            metric = self._metric_from_config(metric_cfg, "train")
            metrics.append(metric)

        start_items = self.epoch_items
        self.model = self.model.train()

        def step(
            batch: data.TrainBatch,
            rank_batch_size: int,
            inputs: dict[str, Any],
            labels: torch.Tensor,
        ) -> tuple[torch.Tensor, float]:
            outputs, loss_dict = self.model(**inputs)
            loss = self.loss_fn(outputs, labels)
            loss = loss + sum(loss_dict.values())
            if self.gradient_accumulation_reduction == "mean":
                loss = loss * len(batch) / rank_batch_size
            if loss.isnan():
                # nans typically from ce loss with all labels
                # ignored, not from training issues
                loss.fill_(0.0)
            self.grad_scaler.scale(loss).backward()
            return outputs.detach(), loss.item()

        train_iter = iter(self.train_loader)
        while True:
            # reset stuff for training step
            start_step = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)

            start_batch = time.perf_counter()
            min_size = sys.maxsize
            max_size = 0
            batches = []
            for i in range(self.gradient_accumulation_steps):
                batch = next(train_iter, None)
                if batch is None:
                    break
                elif len(batch) == 0:  # type: ignore
                    raise RuntimeError(
                        "got empty batch, this should not happen during training"
                    )

                for size in batch.sizes():
                    mean_item_size.add(size)
                    if size < min_size:
                        min_size = size
                    if size > max_size:
                        max_size = size

                batches.append(batch)

            end_batch = time.perf_counter()
            min_num_batches[0] = len(batches)
            dist.all_reduce(min_num_batches, dist.ReduceOp.MIN)
            batches = batches[: min_num_batches.item()]

            if len(batches) == 0:
                self.logger.info(
                    f"[rank {self.info.rank}] finished epoch {self.epoch + 1}"
                )
                break

            rank_batch_size = sum(len(batch) for batch in batches)
            total_batch_size[0] = rank_batch_size
            mean_batch_load.add((end_batch - start_batch) * 1000)
            mean_item_size_ratio.add(max_size / max(1, min_size))

            first_outputs = None
            losses = []
            for i, batch in enumerate(batches):
                start_preparation = time.perf_counter()
                inputs, labels = self._prepare_batch(batch)
                end_preparation = time.perf_counter()
                mean_batch_preparation.add((end_preparation - start_preparation) * 1000)

                with (
                    torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision,
                        enabled=self.mixed_precision is not None,
                    ),
                    torch.autograd.set_detect_anomaly(
                        os.environ.get("TORCH_SET_DETECT_ANOMALY", "") != ""
                    ),
                ):
                    if i < len(batches) - 1 and self.info.is_distributed:
                        with self.model.no_sync():
                            outputs, loss = step(batch, rank_batch_size, inputs, labels)
                    else:
                        # synchronize gradients for the last batch
                        outputs, loss = step(batch, rank_batch_size, inputs, labels)

                losses.append(loss)
                if first_outputs is None:
                    first_outputs = outputs

            grad_norm = 0.0
            for p in self.model.parameters():
                if p.requires_grad and p.grad is not None:
                    grad_norm += torch.linalg.vector_norm(p.grad).item() ** 2

            if isinstance(self.model, FSDP):
                grad_norm_all = torch.full(1, grad_norm, device=self.info.device)
                dist.all_reduce(grad_norm_all, dist.ReduceOp.SUM)
                grad_norm = grad_norm_all.item()

            grad_norm = grad_norm**0.5
            if self.info.is_main_process:
                mean_grad_norm.add(grad_norm)

            if self.gradient_clipper is not None:
                self.gradient_clipper.add_norm(grad_norm)
                clip_norm = self.gradient_clipper.get_norm()

                self.grad_scaler.unscale_(self.optimizer)
                if isinstance(self.model, FSDP):
                    self.model.clip_grad_norm_(clip_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            self.total_step += 1
            self.epoch_step += 1

            dist.all_reduce(total_batch_size, dist.ReduceOp.SUM)
            batch_items = total_batch_size.item()
            self.epoch_items += batch_items
            self.total_items += batch_items
            mean_batch_size.add(batch_items)

            mean_loss.add(sum(losses))
            mean_peak_gpu_memory.add(
                torch.cuda.max_memory_allocated(self.info.device) / 1024**3
            )

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

            mean_step_time.add((time.perf_counter() - start_step) * 1000)

            if self.total_items >= self.log_at:
                mean_loss.sync()
                mean_grad_norm.sync()
                mean_batch_size.sync()
                mean_step_time.sync()
                mean_batch_load.sync()
                mean_item_size.sync()
                mean_item_size_ratio.sync()
                mean_batch_preparation.sync()
                mean_peak_gpu_memory.sync()
                end = time.perf_counter()

                if self.info.is_main_process:
                    # log training progress only on main process
                    assert self.summary_writer is not None

                    progress = 100 * self.total_items / self.training_items
                    self.summary_writer.add_scalar(
                        "train_progress", progress, self.total_step
                    )
                    self.logger.info(
                        f"[step {self.total_step}] "
                        f"train_progress: {progress:.2f}%, "
                        f"{self.total_items:,} / {self.training_items:,} items"
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

                    mean_grad_norm.log_tensorboard(self.summary_writer, self.total_step)
                    mean_grad_norm.log_info(self.logger, self.total_step)

                    mean_batch_size.log_tensorboard(
                        self.summary_writer, self.total_step
                    )
                    mean_batch_size.log_info(self.logger, self.total_step)

                    mean_batch_load.log_tensorboard(
                        self.summary_writer, self.total_step
                    )
                    mean_batch_load.log_info(self.logger, self.total_step)

                    mean_step_time.log_tensorboard(self.summary_writer, self.total_step)
                    mean_step_time.log_info(self.logger, self.total_step)

                    mean_batch_preparation.log_tensorboard(
                        self.summary_writer, self.total_step
                    )
                    mean_batch_preparation.log_info(self.logger, self.total_step)

                    mean_item_size.log_tensorboard(self.summary_writer, self.total_step)
                    mean_item_size.log_info(self.logger, self.total_step)

                    mean_item_size_ratio.log_tensorboard(
                        self.summary_writer, self.total_step
                    )
                    mean_item_size_ratio.log_info(self.logger, self.total_step)

                    mean_peak_gpu_memory.log_tensorboard(
                        self.summary_writer, self.total_step
                    )
                    mean_peak_gpu_memory.log_info(self.logger, self.total_step)

                    items = batches[0].items()
                    for metric in metrics:
                        metric.set_values(items, first_outputs)
                        metric.log_tensorboard(self.summary_writer, self.total_step)
                        metric.log_info(self.logger, self.total_step)

                    self.logger.info(
                        f"[step {self.total_step}] train_time for ~{self.log_interval:,} items: "
                        f"{(end - start) / 60:.2f} minutes"
                    )
                    eta_msg = logging.eta_minutes_message(
                        (end - begin_of_epoch) / 60,
                        self.epoch_items - start_items,
                        self.training_items_per_epoch - start_items,
                    )
                    self.logger.info(
                        f"[step {self.total_step}] [epoch {self.epoch + 1}] {eta_msg}"
                    )

                start = end
                mean_loss.reset()
                mean_grad_norm.reset()
                mean_batch_size.reset()
                mean_step_time.reset()
                mean_batch_load.reset()
                mean_item_size.reset()
                mean_item_size_ratio.reset()
                mean_batch_preparation.reset()
                mean_peak_gpu_memory.reset()
                torch.cuda.reset_peak_memory_stats(self.info.device)
                self.log_at += self.log_interval

            if (
                self.cooldown_items > 0
                and self.total_items >= self.eval_at - self.cooldown_items
            ):
                self._start_cooldown()

            if self.total_items >= self.eval_at:
                # evaluation is done distributed
                self._evaluate_and_checkpoint()

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
                    mean_grad_norm.reset()
                    mean_batch_size.reset()
                    mean_step_time.reset()
                    mean_batch_load.reset()
                    mean_item_size.reset()
                    mean_item_size_ratio.reset()
                    mean_batch_preparation.reset()
                    mean_peak_gpu_memory.reset()
                    torch.cuda.reset_peak_memory_stats(self.info.device)
                    start = time.perf_counter()

                self.eval_at += self.eval_interval

    def _evaluate_and_checkpoint(self):
        mean_loss = tensorboard.DistAverageTracker(
            "val_loss", self.info.device, fmt=".2e"
        )

        self.model = self.model.eval()
        self.loss_fn = self.loss_fn.eval()

        metrics = []
        for metric_cfg in self.cfg["val"].get("metrics", []):
            metric = self._metric_from_config(metric_cfg, "val")
            metrics.append(metric)

        start = time.perf_counter()
        val_iter = iter(self.val_loader)
        has_batch = torch.zeros(1, dtype=torch.long, device=self.info.device)
        logged = False
        while True:
            batch = next(val_iter, None)

            has_batch[0] = int(batch is not None)
            dist.all_reduce(has_batch, op=dist.ReduceOp.MIN)

            if has_batch[0] == 0:
                break

            inputs, labels = self._prepare_batch(batch)

            with (
                torch.autocast(
                    "cuda",
                    dtype=self.mixed_precision,
                    enabled=self.mixed_precision is not None,
                ),
                torch.inference_mode(),
            ):
                outputs, loss_dict = self.model(**inputs)
                loss = self.loss_fn(outputs, labels)

            loss = loss + sum(loss_dict.values())
            if not loss.isnan():
                mean_loss.add(loss.item())

            if not logged and self.info.is_main_process:
                for items, outputs in zip(batch.items(), outputs):
                    for metric in metrics:
                        metric.set_values(items, outputs)
                        metric.log_tensorboard(self.summary_writer, self.total_step)
                        metric.log_info(self.logger, self.total_step)
                logged = True

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

        ckpt_path = os.path.join(self.directories["checkpoints"], "checkpoint_last.pt")
        val_loss = mean_loss.value
        self._save_checkpoint(ckpt_path, val_loss)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_ckpt_path = os.path.join(
                self.directories["checkpoints"], "checkpoint_best.pt"
            )
            self._save_checkpoint(best_ckpt_path, val_loss, full=False)

        self.model = self.model.train()
        self.loss_fn = self.loss_fn.train()

    def _start_cooldown(self):
        # already started
        if self.cooldown_scheduler is not None:
            return
        # cooldown scheduler linearly decays lr from
        # current value to 0
        if self.lr_scheduler is not None:
            factor = (
                self.lr_scheduler.get_last_lr()[0]
                / self.cfg["train"]["optimizer"]["lr"]
            )
        else:
            factor = 1.0
        steps = max(
            1,
            min(self.cooldown_items, self.eval_at - self.total_items)
            // self.step_interval,
        )
        self.cooldown_scheduler = lr_scheduler.LambdaLR(
            self.optimizer, lambda step: (1 - (min(step, steps) / steps)) * factor
        )
        path = os.path.join(self.directories["checkpoints"], "cooldown_checkpoint.pt")
        self._save_checkpoint(path, self.best_val_loss)

    def _stop_cooldown(self):
        # already stopped
        if self.cooldown_scheduler is None:
            return
        self.cooldown_scheduler = None
        path = os.path.join(self.directories["checkpoints"], "cooldown_checkpoint.pt")
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
                self.directories["checkpoints"], "checkpoint_last.pt"
            )
            self._save_checkpoint(ckpt_path, self.best_val_loss)
            end = time.perf_counter()
            if self.info.is_main_process:
                self.logger.info(f"final checkpointing took {end - start:.2f}s")
                if len(self.cleanup) > 0:
                    self.logger.info(
                        f"deleting temporary data sources on local main process with rank {self.info.rank}"
                    )
                    for path in self.cleanup:
                        os.remove(path)
