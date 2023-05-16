import argparse
import copy
import sys
import random
import os
import hashlib
import shutil
import time
import zipfile
from typing import Dict, Optional, Tuple, Any, List, Callable

import torch
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import nn
from torch.backends import cudnn, cuda  # noqa
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import yaml

from text_correction_utils.modules.loss import loss_from_config
from text_correction_utils.modules.scheduler import (
    lr_scheduler_from_config,
    max_length_scheduler_from_config
)
from text_correction_utils.modules.optimizer import optimizer_from_config
from text_correction_utils import (
    distributed,
    data,
    configuration,
    io,
    tokenization,
    logging,
    api,
    tensorboard
)


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

    def __init__(self, cfg: Dict[str, Any], directories: Dict[str, str], info: distributed.DistributedInfo):
        self.cfg = cfg
        self.directories = directories
        self.info = info

        # globals used throughout training
        self.total_step = 0
        self.epoch_step = 0
        self.epoch_items = 0
        self.total_items = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
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

        self.model = self._model_from_config(
            self.cfg
        ).to(self.info.device).train()

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
            self.cfg["input_tokenizer"],
            num_epochs=num_epochs,
            seed=self.cfg["seed"],
            info=self.info
        )

        self.optimizer = optimizer_from_config(
            self.model,
            self.cfg["train"]["optimizer"],
            additional_optimizer_fn=self._additional_optimizer_fn
        )
        if self.info.is_main_process:
            num_params = 0
            param_group_infos = []
            for i, param_group in enumerate(self.optimizer.param_groups):
                group_num_params = sum(
                    p.numel()
                    for p in param_group["params"]
                )
                other = {k: v for k, v in param_group.items() if k != "params"}
                param_group_infos.append(
                    f"{i+1}. group: {group_num_params:,} params, other: {other}"
                )
                num_params += group_num_params
            param_group_info = "\n".join(param_group_infos)
            self.logger.info(
                f"Optimizer parameter groups:\n{param_group_info}"
            )
            model_params = api.num_parameters(self.model)
            assert model_params["trainable"] == num_params, \
                f"number of trainable parameters in model {model_params['trainable']:,}, " \
                f"and optimized parameters {num_params:,} do not match"

        self.clip_grad_norm = self.cfg["train"].get("clip_grad_norm")

        def clamp(v: float, minimum: int, maximum: int) -> int:
            return max(min(round(v), maximum), minimum)

        self.log_interval = clamp(
            self.training_items_per_epoch * self.cfg["train"].get("log_interval", 0.001),
            1,
            self.training_items_per_epoch
        )
        self.eval_interval = clamp(
            self.training_items_per_epoch * self.cfg["train"].get("eval_interval", 0.1),
            1,
            self.training_items_per_epoch
        )

        if "lr_scheduler" in self.cfg["train"]:
            self.step_interval = clamp(
                self.training_items_per_epoch * self.cfg["train"].get("step_interval", 0.001),
                1,
                self.training_items_per_epoch
            )
            steps = self.training_items // self.step_interval
            self.lr_scheduler = lr_scheduler_from_config(
                self.optimizer,
                steps,
                cfg["train"]["lr_scheduler"],
                additional_lr_scheduler_fn=self._additional_lr_scheduler_fn
            )
        else:
            self.step_interval = 0
            self.lr_scheduler = None

        self.loss_fn = loss_from_config(
            self.cfg["train"]["loss"],
            additional_loss_fn=self._additional_loss_fn
        ).to(self.info.device).train()

        self.grad_scaler = amp.GradScaler(
            enabled=self.cfg["train"].get("mixed_precision", False)
        )
        mixed_precision_dtype = self.cfg["train"].get(
            "mixed_precision_dtype", "fp16"
        )
        if mixed_precision_dtype == "fp16":
            self.mixed_prec_dtype = torch.float16
        elif mixed_precision_dtype == "bfp16":
            self.mixed_prec_dtype = torch.bfloat16
        else:
            raise ValueError(
                f"unknown mixed precision type {mixed_precision_dtype}, "
                f"must be fp16 or bfp16"
            )

        if self.info.is_main_process:
            self.summary_writer = SummaryWriter(
                log_dir=self.directories["tensorboard"]
            )

            self.logger.info(f"Using model:\n{self.model}")
            self.logger.info(f"Model parameters: {api.num_parameters(self.model)}")
            self.logger.info(
                f"Number of training items: {self.training_items_per_epoch:,} per epoch, "
                f"{self.training_items:,} total"
            )
            self.logger.info(
                f"Logging every {self.log_interval:,} items, "
                f"evaluating every {self.eval_interval:,} items"
            )

            test_sentence = "This is a test sentence."
            self.logger.info(
                f"Testing input tokenizer:\n{self.input_tokenizer.tokenize(test_sentence).token_ids}")
            if self.output_tokenizer is not None:
                self.logger.info(
                    f"Testing output tokenizer:\n{self.output_tokenizer.tokenize(test_sentence).token_ids}")

            self.logger.info(f"Type 'tensorboard --logdir {self.directories['tensorboard']}' "
                             f"to view the training process in Tensorboard")
        else:
            self.summary_writer = None

        # resume training from last checkpoint if it exists
        last_checkpoint = os.path.join(
            self.directories["checkpoints"],
            "checkpoint_last.pt"
        )
        load_checkpoint = cfg["train"].get("load_checkpoint")
        if os.path.exists(last_checkpoint):
            checkpoint = io.load_checkpoint(last_checkpoint)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.lr_scheduler is not None and checkpoint.get("lr_scheduler_state_dict") is not None:
                self.lr_scheduler.load_state_dict(
                    checkpoint["lr_scheduler_state_dict"]
                )
            if checkpoint.get("grad_scaler_state_dict") is not None:
                self.grad_scaler.load_state_dict(
                    checkpoint["grad_scaler_state_dict"]
                )
            if checkpoint.get("loss_fn_state_dict") is not None:
                self.loss_fn.load_state_dict(checkpoint["loss_fn_state_dict"])

            self.total_step = checkpoint["step"]
            self.epoch = checkpoint["epoch"]
            self.best_val_loss = checkpoint["val_loss"]
            self.epoch_step = checkpoint["epoch_step"]
            self.epoch_items = checkpoint["epoch_items"]
            self.total_items = checkpoint["total_items"]

            if self.max_length_scheduler is not None:
                self.max_length = self.max_length_scheduler(self.total_items)
                self.train_loader.set_max_length(self.max_length)

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
            wrong_keys = self.model.load_state_dict(
                checkpoint["model_state_dict"],
                strict=False
            )
            assert len(wrong_keys.unexpected_keys) == 0, \
                f"unexpected keys in checkpoint \"{load_checkpoint}\": {wrong_keys.unexpected_keys}"

            self.logger.info(
                f"initializing model from checkpoint \"{load_checkpoint}\" "
                f"(missing keys: {wrong_keys.missing_keys})"
            )

        self.model = DDP(self.model)

    @classmethod
    def _model_from_config(cls, cfg: Dict[str, Any]) -> nn.Module:
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
                org_path = src["original_path"]
                proc_path = src["processed_path"]
                assert os.path.isfile(org_path) and os.path.isfile(proc_path), \
                    f"one of {org_path} or {proc_path} is not a file"
                temp_dir = src.get("temp_dir")
                if temp_dir is not None:
                    org_path = cls._copy_file_to_tmp_dir(
                        org_path, temp_dir, info)
                    proc_path = cls._copy_file_to_tmp_dir(
                        proc_path, temp_dir, info)
                    cleanup_paths.extend([org_path, proc_path])
                src_preprocessings.append(preprocessing)
                src_paths.append((org_path, proc_path))
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
        cfg: Dict[str, Any],
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
        cfg = copy.deepcopy(cfg)
        val_cfg = cfg.pop("val")
        if isinstance(val_cfg, int):
            val_limit = val_cfg
        else:
            raise ValueError(
                "val data must be an integer"
            )

        (
            train_sources,
            train_languages,
            train_preprocessings,
            train_postprocessings,
            train_cleanup
        ) = cls._prepare_data_sources(
            cfg.pop("sources"),
            info
        )

        num_languages_specified = sum(
            lang is not None for lang in train_languages
        )
        default_language = cfg.pop("default_language", None)
        if num_languages_specified > 0 and num_languages_specified < len(train_languages):
            assert default_language is not None, \
                "expected default_language to be specified if some, but not all " \
                "individual data sources specify a language"
            train_languages = [
                default_language if lang is None else lang
                for lang in train_languages
            ]
        elif num_languages_specified == 0:
            train_languages = None

        pipeline_cfg = cfg.pop("pipeline")
        if "preprocessing" not in pipeline_cfg:
            assert all(preproc is not None for preproc in train_preprocessings), \
                "expected preprocessing to be specified per data source if not specified " \
                "for pipeline"
            pipeline_cfg["preprocessing"] = train_preprocessings
        if "postprocessing" not in pipeline_cfg:
            assert all(postproc is not None for postproc in train_postprocessings), \
                "expected postprocessing to be specified per data source if not specified " \
                "for pipeline"
            pipeline_cfg["postprocessing"] = train_postprocessings

        # adapt config to multi gpu usage
        assert "batch_limit" in cfg, "batch_limit must be in data config"
        cfg["batch_limit"] = max(1, cfg["batch_limit"] // info.world_size)

        train_limit = cfg.get("limit")
        if train_limit is not None:
            assert train_limit > val_limit, \
                "train limit must be bigger than val limit"

        max_length = cfg.pop("max_length")
        assert max_length is not None, "missing max_length in data config"
        max_length_scheduler_cfg = cfg.pop("max_length_scheduler", None)

        train_loader = data.DataLoader.from_files(
            train_sources,
            pipeline_cfg,
            tokenizer_config,
            train_languages,
            seed=seed,
            skip=val_limit,
            max_length=max_length,
            distributed=(info.rank, info.world_size),
            **cfg
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
                additional_max_length_scheduler_fn=cls._additional_max_length_scheduler_fn
            )
        else:
            max_length_scheduler = None

        # for validation always turn off shuffling, turn on sorting, and
        # specify a val limit
        cfg["shuffle"] = False
        cfg["sort"] = True
        cfg["limit"] = val_limit
        val_loader = data.DataLoader.from_files(
            train_sources,
            pipeline_cfg,
            tokenizer_config,
            train_languages,
            max_length=max_length,
            seed=seed,
            distributed=None,
            **cfg
        )
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
            f.write(yaml.dump(cfg))
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
                yaml.dump({
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
        local_world_size = int(os.environ.get("SLURM_NTASKS_PER_NODE", os.environ["SLURM_NTASKS"]))
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

        assert dist.is_initialized(), "failed to initialize process group"

        resuming = os.path.exists(experiment_dir) and os.path.exists(
            os.path.join(experiment_dir, "checkpoints", "checkpoint_last.pt")
        )
        if not resuming:
            assert config_path is not None, "specify config if not resuming an existing experiment"
            cfg = configuration.load_config(config_path)
            if info.is_main_process:
                cls._setup_experiment(
                    work_dir, experiment_dir, config_path, cfg)
                logger.info(
                    f"Starting experiment at {experiment_dir} with config:\n{yaml.dump(cfg)}")
        else:
            cfg = configuration.load_config_from_experiment(experiment_dir)
            if info.is_main_process:
                logger.info(
                    f"Resuming from {experiment_dir} with config:\n{yaml.dump(cfg)}")

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
                f"Starting experiment at {experiment_dir} with config:\n{yaml.dump(cfg)}")
        else:
            cfg = configuration.load_config_from_experiment(experiment_dir)
            logger.info(
                f"Resuming from {experiment_dir} with config:\n{yaml.dump(cfg)}")
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
        batch: data.DataBatch
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

        mean_loss = tensorboard.AverageTracker("train_loss", fmt=".2e")
        mean_forward_pass = tensorboard.AverageTracker("train_forward_pass")
        mean_batch_load = tensorboard.AverageTracker("train_batch_load")
        mean_step_time = tensorboard.AverageTracker("train_step_time")
        mean_batch_preparation = tensorboard.AverageTracker(
            "train_batch_preparation"
        )
        mean_bsz = tensorboard.AverageTracker("train_batch_size")
        mean_seq_length = tensorboard.AverageTracker("train_sequence_length")
        mean_seq_length_ratio = tensorboard.AverageTracker(
            "train_sequence_length_ratio"
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

        log_at = self.total_items + self.log_interval
        eval_at = self.total_items + self.eval_interval
        step_at = self.total_items + self.step_interval

        start_items = self.epoch_items

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
            inputs, labels = self._prepare_batch(batch)
            end_preparation = time.perf_counter()

            self.optimizer.zero_grad(set_to_none=True)

            start_forward = time.perf_counter()
            with amp.autocast(
                enabled=self.grad_scaler.is_enabled(),
                dtype=self.mixed_prec_dtype
            ):
                outputs, loss_dict = self.model(**inputs)
                loss = self.loss_fn(outputs, labels)
                loss = loss + sum(loss_dict.values())
            end_forward = time.perf_counter()

            self.grad_scaler.scale(loss).backward()
            if self.clip_grad_norm is not None:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_grad_norm
                )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            self.total_step += 1
            self.epoch_step += 1
            self.epoch_items += len(batch)
            self.total_items += len(batch)

            if self.lr_scheduler is not None and self.total_items >= step_at:
                self.lr_scheduler.step()
                step_at += self.step_interval

            if self.max_length_scheduler is not None:
                max_length = self.max_length_scheduler(self.total_items)
                if max_length != self.max_length:
                    self.max_length = max_length
                    self.train_loader.set_max_length(max_length)
                    self.val_loader.set_max_length(max_length)

            if self.info.is_main_process:
                mean_step_time.add((time.perf_counter() - start_batch) * 1000)
                mean_loss.add(loss.detach())
                mean_forward_pass.add((end_forward - start_forward) * 1000)
                # approximation since we expect every rank to roughly
                # have the same batch size
                batch_size = len(batch) * self.info.world_size
                mean_bsz.add(batch_size)
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

            if self.info.is_main_process and self.total_items >= eval_at:
                self._evaluate_and_checkpoint()
                eval_at += self.eval_interval

            if self.info.is_main_process and self.total_items >= log_at:
                # log training progress only on main process
                progress = 100 * self.total_items / self.training_items
                self.summary_writer.add_scalar(
                    "train_progress",
                    progress,
                    self.total_step
                )
                self.logger.info(
                    f"[step {self.total_step}] "
                    f"train_progress: {progress:.2f}%, {self.total_items:,} / {self.training_items:,} items"
                )

                if self.lr_scheduler is not None:
                    for i, lr in enumerate(self.lr_scheduler.get_last_lr()):
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

                mean_forward_pass.log_tensorboard(
                    self.summary_writer, self.total_step
                )
                mean_forward_pass.log_info(self.logger, self.total_step)

                mean_batch_load.log_tensorboard(self.summary_writer, self.total_step)
                mean_batch_load.log_info(self.logger, self.total_step)

                mean_step_time.log_tensorboard(self.summary_writer, self.total_step)
                mean_step_time.log_info(self.logger, self.total_step)

                mean_batch_preparation.log_tensorboard(
                    self.summary_writer, self.total_step
                )
                mean_batch_preparation.log_info(self.logger, self.total_step)

                mean_seq_length.log_tensorboard(self.summary_writer, self.total_step)
                mean_seq_length.log_info(self.logger, self.total_step)

                mean_seq_length_ratio.log_tensorboard(
                    self.summary_writer, self.total_step
                )
                mean_seq_length_ratio.log_info(self.logger, self.total_step)

                items = batch.items
                for metric in metrics:
                    metric.set_values(items, outputs)
                    metric.log_tensorboard(self.summary_writer, self.total_step)
                    metric.log_info(self.logger, self.total_step)

                self.summary_writer.add_histogram(
                    "train_batch_size_hist",
                    torch.as_tensor(mean_bsz.values),
                    self.total_step
                )

                self.summary_writer.add_histogram(
                    "train_batch_load_hist",
                    torch.as_tensor(mean_batch_load.values),
                    self.total_step
                )

                self.summary_writer.add_histogram(
                    "train_step_hist",
                    torch.as_tensor(mean_step_time.values),
                    self.total_step
                )

                self.summary_writer.add_histogram(
                    "train_batch_sequence_length_hist",
                    torch.as_tensor(mean_seq_length.values),
                    self.total_step
                )

                self.summary_writer.add_histogram(
                    "train_sequence_length_ratio_hist",
                    torch.as_tensor(mean_seq_length_ratio.values),
                    self.total_step
                )

                end = time.perf_counter()
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

                mean_loss.reset()
                mean_bsz.reset()
                mean_step_time.reset()
                mean_forward_pass.reset()
                mean_batch_load.reset()
                mean_seq_length.reset()
                mean_seq_length_ratio.reset()
                mean_batch_preparation.reset()
                start = end

            if self.total_items >= log_at:
                self.logger.info(
                    f"[step {self.total_step}] [GPU:{self.info.rank}:{self.info.local_rank}] nvidia-smi:\n"
                    f"{api.nvidia_smi()}"
                )
                log_at += self.log_interval

    def _evaluate_and_checkpoint(self):
        assert self.info.is_main_process, "evaluation should be only done on main process"

        mean_loss = tensorboard.AverageTracker("val_loss", fmt=".2e")

        self.model = self.model.eval()
        self.loss_fn = self.loss_fn.eval()
        metric_cfg = self.cfg["train"].get("metrics")
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
            inputs, labels = self._prepare_batch(batch=batch)

            with torch.inference_mode(), amp.autocast(
                enabled=self.grad_scaler.is_enabled(),
                dtype=self.mixed_prec_dtype
            ):
                outputs, loss_dict = self.model(**inputs)
                loss = self.loss_fn(outputs, labels)
                loss = loss + sum(loss_dict.values())

                mean_loss.add(loss.item())

            if batch_num == 0:
                items = batch.items
                for metric in metrics:
                    metric.set_values(items, outputs)
                    metric.log_tensorboard(self.summary_writer, self.total_step)
                    metric.log_info(self.logger, self.total_step)

        end = time.perf_counter()

        mean_loss.log_tensorboard(self.summary_writer, self.total_step)
        mean_loss.log_info(self.logger, self.total_step)

        self.logger.info(
            f"[step {self.total_step}] validation took {(end - start) / 60:.2f} minutes"
        )
        val_loss = mean_loss.value
        ckpt_path = os.path.join(
            self.directories["checkpoints"],
            "checkpoint_last.pt"
        )
        io.save_checkpoint(
            checkpoint_path=ckpt_path,
            model=distributed.unwrap_ddp(self.model),
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            loss_fn=self.loss_fn,
            grad_scaler=self.grad_scaler,
            step=self.total_step,
            epoch=self.epoch,
            epoch_step=self.epoch_step,
            epoch_items=self.epoch_items,
            total_items=self.total_items,
            val_loss=val_loss
        )

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_ckpt_path = os.path.join(
                self.directories["checkpoints"],
                "checkpoint_best.pt"
            )
            shutil.copy2(ckpt_path, best_ckpt_path)

        self.model = self.model.train()
        self.loss_fn = self.loss_fn.train()

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
                    "got termination signal, evaluating and saving on main process before exiting"
                )

        finally:
            if self.info.is_main_process:
                start = time.perf_counter()
                self._evaluate_and_checkpoint()
                end = time.perf_counter()
                self.logger.info(
                    f"final evaluation and checkpointing took {end - start:.2f}s")
            if len(self.cleanup) > 0 and self.info.is_local_main_process:
                self.logger.info(
                    f"deleting temporary data sources on local main process with rank {self.info.rank}")
                for path in self.cleanup:
                    os.remove(path)
