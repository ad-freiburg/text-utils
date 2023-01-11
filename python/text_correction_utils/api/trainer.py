import argparse
import copy
import random
import collections
import os
import shutil
import tempfile
import time
from typing import Dict, Optional, Tuple, Any, List
import zipfile

import torch
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import nn
from torch.backends import cudnn  # noqa
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import yaml

from text_correction_utils.modules.loss import loss_from_config
from text_correction_utils.modules.lr_scheduler import lr_scheduler_from_config
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
        return parser

    def __init__(self, cfg: Dict[str, Any], directories: Dict[str, str], info: distributed.DistributedInfo):
        self.cfg = cfg
        self.directories = directories
        self.info = info

        # globals used throughout training
        self.step = 0
        self.epoch_step = 0
        self.epoch_items = 0
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

        torch.manual_seed(self.cfg["seed"])
        torch.cuda.manual_seed(self.cfg["seed"])

        torch.use_deterministic_algorithms(False)

        self.input_tokenizer = tokenization.tokenizer_from_config(self.cfg["input_tokenizer"])
        if "output_tokenizer" in self.cfg:
            self.output_tokenizer = tokenization.tokenizer_from_config(self.cfg["output_tokenizer"])
        else:
            self.output_tokenizer = None

        self.model = self._model_from_config(self.cfg).to(self.info.device).train()

        self.train_loader, self.val_loader = self._data_from_config(
            self.cfg["train"]["data"],
            self.cfg["input_tokenizer"],
            seed=self.cfg["seed"],
            info=self.info
        )
        num_batches = 200
        avg_batch_size = tensorboard.AverageTracker("batch_size")
        if self.info.is_main_process:
            self.logger.info(
                f"Estimating train loader length from average batch size of first {num_batches} batches "
                f"and minimum train loader items."
            )
        for idx, batch in enumerate(self.train_loader):
            if idx >= num_batches:
                break
            avg_batch_size.add(len(batch))

        self.training_steps_per_epoch = int(self.train_loader.min_items // avg_batch_size.value)
        self.training_steps = self.cfg["train"]["num_epochs"] * self.training_steps_per_epoch
        self.logger.info(f"Got an average batch size of {avg_batch_size.value:.2f} after {num_batches:,} batches. "
                         f"The train loader contains at least {self.train_loader.min_items} items, so the estimated "
                         f"number of training steps over {self.cfg['train']['num_epochs']} epochs "
                         f"is {self.training_steps:,} ({self.training_steps_per_epoch:,} per epoch).")
        self.optimizer = optimizer_from_config(
            self.model,
            self.cfg["train"]["optimizer"]
        )
        self.clip_grad_norm = self.cfg["train"].get("clip_grad_norm")

        if "lr_scheduler" in self.cfg["train"]:
            self.lr_scheduler = lr_scheduler_from_config(
                self.optimizer,
                self.training_steps,
                cfg["train"]["lr_scheduler"]
            )
        else:
            self.lr_scheduler = None

        self.loss_fn = loss_from_config(self.training_steps, self.cfg["train"]["loss"]).train()
        self.grad_scaler = amp.GradScaler(enabled=self.cfg["train"].get("mixed_precision", False))
        mixed_precision_dtype = self.cfg["train"].get("mixed_precision_dtype", "fp16")
        if mixed_precision_dtype == "fp16":
            self.mixed_prec_dtype = torch.float16
        elif mixed_precision_dtype == "bfp16":
            self.mixed_prec_dtype = torch.bfloat16
        else:
            raise ValueError(f"unknown mixed precision type {mixed_precision_dtype}, must fp16 or bfp16")

        eval_interval = self.cfg["train"].get("eval_interval", 0.1)
        log_interval = self.cfg["train"].get("log_interval", 0.01)

        def clamp(v: int, low: int, up: int) -> int:
            return min(max(low, v), up)

        if isinstance(eval_interval, float):
            eval_interval = int(eval_interval * self.training_steps_per_epoch)
        if isinstance(log_interval, float):
            log_interval = int(log_interval * self.training_steps_per_epoch)

        self.eval_interval = clamp(eval_interval, 1, self.training_steps_per_epoch)
        self.log_interval = clamp(log_interval, 1, self.training_steps_per_epoch)

        if self.info.is_main_process:
            self.summary_writer = SummaryWriter(log_dir=self.directories["tensorboard"])

            self.logger.info(f"Using model:\n{self.model}")
            self.logger.info(f"Model parameters: {api.num_parameters(self.model)}")

            test_sentence = "This is a test sentence."
            self.logger.info(f"Testing input tokenizer:\n{self.input_tokenizer.tokenize(test_sentence).token_ids}")
            if self.output_tokenizer is not None:
                self.logger.info(
                    f"Testing output tokenizer:\n{self.output_tokenizer.tokenize(test_sentence).token_ids}")

            self.logger.info(f"Type 'tensorboard --logdir {self.directories['tensorboard']}' "
                             f"to view the training process in Tensorboard")
            self.logger.info(f"Evaluating every {eval_interval:,} steps, logging every {log_interval:,} steps")
        else:
            self.summary_writer = None

        # resume training from last checkpoint if it exists
        last_checkpoint = os.path.join(self.directories["checkpoints"], "checkpoint_last.pt")
        if os.path.exists(last_checkpoint):
            checkpoint = io.load_checkpoint(last_checkpoint)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.lr_scheduler is not None and checkpoint.get("lr_scheduler_state_dict") is not None:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            if checkpoint.get("grad_scaler_state_dict") is not None:
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
            if checkpoint.get("loss_fn_state_dict") is not None:
                self.loss_fn.load_state_dict(checkpoint["loss_fn_state_dict"])

            self.step = checkpoint["step"]
            self.epoch = checkpoint["epoch"]
            self.best_val_loss = checkpoint["val_loss"]
            self.epoch_step = checkpoint["epoch_step"]
            self.epoch_items = checkpoint["epoch_items"]

            if self.info.is_main_process:
                self.logger.info(
                    f"Resuming training from checkpoint {last_checkpoint}\n"
                    f"Starting at epoch {self.epoch + 1} at global step {self.step: , } (epoch step {self.epoch_step}) "
                    f"with a best validation loss of {self.best_val_loss:.6f}\n"
                    f"Fast forwarding {self.epoch_items:,} items."
                )

        self.model = DDP(self.model)

    @classmethod
    def _model_from_config(cls, cfg: Dict[str, Any]) -> nn.Module:
        raise NotImplementedError

    @classmethod
    def _copy_file_to_tmp_dir(cls, path: str, dir: str) -> str:
        _, file_name = os.path.split(path)
        # make temp path unique by hashing the full input path, because only
        # using the file name could cause issues if some files are named the same
        os.makedirs(dir, exist_ok=True)
        temp_path = os.path.join(dir, f"{hash(path) % 10 ** 8}_{file_name}")
        shutil.copy2(path, temp_path)
        return temp_path

    @classmethod
    def _parse_data_sources(cls, sources: List[Dict[str, Any]]) -> Tuple[List[str], Optional[List[str]]]:
        src_paths = []
        src_langs = []
        for src in sources:
            src = copy.deepcopy(src)
            src_type = src.pop("type")
            if src_type == "file":
                lang = src.get("language")
                path = src["path"]
                assert os.path.isfile(path), f"{path} is not a file"
                temp_dir = src.get("temp_dir")
                if temp_dir is not None:
                    path = cls._copy_file_to_tmp_dir(path, temp_dir)
                src_paths.append(path)
                src_langs.append(lang)
            elif src_type == "file_glob":
                lang = src.get("language")
                temp_dir = src.get("temp_dir")
                for path in io.glob_safe(src["glob"]):
                    assert os.path.isfile(path), f"{path} is not a file"
                    if temp_dir is not None:
                        path = cls._copy_file_to_tmp_dir(path, temp_dir)
                    src_paths.append(path)
                    src_langs.append(lang)
            else:
                raise ValueError(f"unknown source type {src_type}")
        assert len(src_paths) > 0, "got no data sources"
        if all(lang is None for lang in src_langs):
            return src_paths, None
        else:
            return src_paths, [
                lang if lang is not None
                else tokenization.LanguageTokens.UNK
                for lang in src_langs
            ]

    @classmethod
    def _data_from_config(
        cls,
        cfg: Dict[str, Any],
        input_tokenizer_cfg: Dict[str, Any],
        seed: Optional[int],
        info: distributed.DistributedInfo
    ) -> Tuple[data.DataLoader, data.DataLoader]:
        cfg = copy.deepcopy(cfg)
        val_cfg = cfg.pop("val")
        if isinstance(val_cfg, int):
            val_limit = val_cfg
            val_sources = val_languages = None
        elif isinstance(val_cfg, list):
            val_limit = None
            val_sources, val_languages = cls._parse_data_sources(val_cfg)
        else:
            raise ValueError("val data must either be an integer or a list of data sources")

        train_sources, train_languages = cls._parse_data_sources(cfg.pop("sources"))

        pipeline = cfg.pop("pipeline")
        pipeline_config = data.PreprocessingPipelineConfig(
            preprocessing=pipeline.get("preprocessing", []),
            labeling=pipeline["labeling"],
        )
        tokenizer_config = tokenization.tokenizer_config(input_tokenizer_cfg)
        train_loader = data.DataLoader.from_files(
            train_sources,
            pipeline_config,
            tokenizer_config,
            train_languages,
            seed=seed,
            skip=val_limit if val_limit is not None else 0,
            distributed=(info.rank, info.world_size),
            **cfg
        )

        if val_limit is not None:
            val_loader = data.DataLoader.from_files(
                train_sources,
                pipeline_config,
                tokenizer_config,
                train_languages,
                seed=seed,
                limit=val_limit,
                distributed=None,
                **cfg
            )
        else:
            val_loader = data.DataLoader.from_files(
                val_sources,
                pipeline_config,
                tokenizer_config,
                val_languages,
                seed=seed,
                distributed=None,
                **cfg
            )
        return train_loader, val_loader

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
                    zf.write(os.path.join(config_dir, file), os.path.join(rel_sub_dir, file))
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
    def _config_from_experiment(cls, dir: str) -> Dict[str, Any]:
        info = configuration.load_config(os.path.join(dir, "info.yaml"))
        with zipfile.ZipFile(os.path.join(dir, "configs.zip"), "r", zipfile.ZIP_DEFLATED) as inz:
            with tempfile.TemporaryDirectory() as tmp_dir:
                inz.extractall(tmp_dir)
                return configuration.load_config(os.path.join(tmp_dir, info["config_name"]))

    @classmethod
    def _train_local_distributed(
        cls,
        rank: int,
        world_size: int,
        port: int,
        cfg: Dict[str, Any],
        directories: Dict[str, str],
    ):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)

        dist.init_process_group(
            backend=dist.Backend.NCCL,
            init_method="env://",
            rank=rank,
            world_size=world_size
        )

        assert dist.is_initialized(), "failed to initialize process group"

        info = distributed.DistributedInfo(
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            local_world_size=world_size
        )

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
        local_rank = int(rank % torch.cuda.device_count())
        local_world_size = torch.cuda.device_count()
        if rank == 0:
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

        assert dist.is_initialized(), "failed to initialize process group"

        info = distributed.DistributedInfo(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size
        )

        resuming = os.path.exists(experiment_dir) and os.path.exists(
            os.path.join(experiment_dir, "checkpoints", "checkpoint_last.pt")
        )
        if not resuming:
            assert config_path is not None, "specify config if not resuming an existing experiment"
            cfg = configuration.load_config(config_path)
            if info.is_main_process:
                cls._setup_experiment(work_dir, experiment_dir, config_path, cfg)
                logger.info(f"Starting experiment at {experiment_dir} with config:\n{yaml.dump(cfg)}")
        else:
            cfg = cls._config_from_experiment(experiment_dir)
            if info.is_main_process:
                logger.info(f"Resuming from {experiment_dir} with config:\n{yaml.dump(cfg)}")

        directories = {
            "experiment": experiment_dir,
            "checkpoints": os.path.join(experiment_dir, "checkpoints"),
            "tensorboard": os.path.join(experiment_dir, "tensorboard")
        }

        cls(cfg, directories, info).run()
        dist.destroy_process_group()

    @classmethod
    def train_local(cls, work_dir: str, experiment_dir: str, config_path: str):
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
            logger.info(f"Starting experiment at {experiment_dir} with config:\n{yaml.dump(cfg)}")
        else:
            cfg = cls._config_from_experiment(experiment_dir)
            logger.info(f"Resuming from {experiment_dir} with config:\n{yaml.dump(cfg)}")
        directories = {
            "experiment": experiment_dir,
            "checkpoints": os.path.join(experiment_dir, "checkpoints"),
            "tensorboard": os.path.join(experiment_dir, "tensorboard")
        }
        mp.spawn(
            fn=cls._train_local_distributed,
            nprocs=num_gpus,
            args=(num_gpus, port, cfg, directories),
            join=True
        )

    def _prepare_label(
        self,
        label: Optional[Dict[str, Any]],
        max_length: int
    ) -> Any:
        assert label is not None
        if label["type"] == "classification":
            return label["label"]
        elif label["type"] == "sequence_classification":
            num_prefix_tokens = self.input_tokenizer.num_prefix_tokens()
            return (
                [-1] * num_prefix_tokens
                + label["labels"]
                + [-1] * (max_length - len(label["labels"]) - num_prefix_tokens)
            )
        else:
            raise ValueError(f"unknown labels type {label['type']}")

    def _prepare_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        info_type = info.pop("type")
        if info_type in {"empty", "token_groups"}:
            return info
        else:
            raise ValueError(f"unknown info type {info_type}")

    def _prepare_batch(self, batch: data.ItemBatch) -> Tuple[Dict[str, Any], torch.Tensor]:
        assert len(batch) > 0, "got empty batch"
        if self.cfg["model"]["type"] == "encoder_with_head":
            pad_token_id = self.input_tokenizer.pad_token_id()
            token_ids = []
            token_lengths = [len(item.tokenization.token_ids) for item in batch]
            max_tokens = max(token_lengths)
            first_info = next(iter(batch)).tokenization.info
            if first_info["type"] == "token_groups":
                if "code_point_groups" in first_info:
                    group_name = "code_point_groups"
                else:
                    group_name = "byte_groups"
                max_groups = max(len(item.tokenization.info[group_name]) for item in batch)
            else:
                max_groups = max_tokens
            labels = []
            kwargs = collections.defaultdict(list)
            for i, item in enumerate(batch):
                token_ids.append(item.tokenization.token_ids + [pad_token_id] * (max_tokens - token_lengths[i]))
                for k, v in self._prepare_info(item.tokenization.info).items():
                    kwargs[k].append(v)

                labels.append(self._prepare_label(item.label, max_groups))

            inputs = {
                "token_ids": torch.as_tensor(token_ids, dtype=torch.long, device=self.info.device),
                "lengths": token_lengths,
                **kwargs
            }
            labels = torch.as_tensor(labels, dtype=torch.long, device=self.info.device)

        else:
            raise ValueError(f"unknown model type {self.cfg['model']['type']}")

        return inputs, labels

    def _train_one_epoch(self):
        begin_of_epoch = time.perf_counter()
        start = time.perf_counter()

        mean_loss = tensorboard.AverageTracker("train_loss", fmt=".2e")
        mean_forward_pass = tensorboard.AverageTracker("train_forward_pass")
        mean_batch_load = tensorboard.AverageTracker("train_batch_load")
        mean_batch_preparation = tensorboard.AverageTracker("train_batch_preparation")
        mean_bsz = tensorboard.AverageTracker("train_batch_size")
        mean_seq_length = tensorboard.AverageTracker("train_sequence_length")

        metric_cfg = self.cfg["train"].get("metrics")
        if metric_cfg is not None:
            metrics = tensorboard.metrics_from_config(
                metric_cfg, self.input_tokenizer, self.output_tokenizer, prefix="train"
            )
        else:
            metrics = []

        self.logger.info(f"[rank {self.info.rank}] [epoch {self.epoch + 1}] training_steps: {self.training_steps}")

        train_iter = iter(self.train_loader)
        while True:
            start_batch = time.perf_counter()
            batch = next(train_iter, None)
            end_batch = time.perf_counter()
            if batch is None:
                self.logger.info(f"[rank {self.info.rank}] finished epoch {self.epoch + 1}")
                break

            start_preparation = time.perf_counter()
            inputs, labels = self._prepare_batch(batch)
            end_preparation = time.perf_counter()

            self.optimizer.zero_grad()

            start_forward = time.perf_counter()
            with amp.autocast(enabled=self.grad_scaler.is_enabled(), dtype=self.mixed_prec_dtype):
                outputs, loss_dict = self.model(**inputs)
                loss = self.loss_fn(outputs, labels)
                loss = loss + sum(loss_dict.values())
            end_forward = time.perf_counter()

            self.grad_scaler.scale(loss).backward()
            if self.clip_grad_norm is not None:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            if self.info.is_main_process:
                mean_loss.add(loss.detach())
                mean_forward_pass.add((end_forward - start_forward) * 1000)
                # approximation since we expect every rank to roughly
                # have the same batch size
                batch_size = len(batch) * self.info.world_size
                mean_bsz.add(batch_size)
                for item in batch:
                    mean_seq_length.add(len(item))
                mean_batch_load.add((end_batch - start_batch) * 1000)
                mean_batch_preparation.add((end_preparation - start_preparation) * 1000)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if hasattr(self.loss_fn, "step"):
                self.loss_fn.step()

            self.step += 1
            self.epoch_step += 1
            self.epoch_items += len(batch)
            if self.epoch_step % self.eval_interval == 0 and self.info.is_main_process:
                self._evaluate_and_checkpoint()

            if self.epoch_step % self.log_interval == 0 and self.info.is_main_process:
                if self.lr_scheduler is not None:
                    lr = self.lr_scheduler.get_last_lr()[0]
                    self.summary_writer.add_scalar("train_lr", lr, self.step)
                    self.logger.info(f"[step {self.step}] train_lr: {lr:.8f}")

                mean_loss.log_tensorboard(self.summary_writer, self.step)
                mean_loss.log_info(self.logger, self.step)

                mean_bsz.log_tensorboard(self.summary_writer, self.step)
                mean_bsz.log_info(self.logger, self.step)

                mean_forward_pass.log_tensorboard(self.summary_writer, self.step)
                mean_forward_pass.log_info(self.logger, self.step)

                mean_batch_load.log_tensorboard(self.summary_writer, self.step)
                mean_batch_load.log_info(self.logger, self.step)

                mean_batch_preparation.log_tensorboard(self.summary_writer, self.step)
                mean_batch_preparation.log_info(self.logger, self.step)

                mean_seq_length.log_tensorboard(self.summary_writer, self.step)
                mean_seq_length.log_info(self.logger, self.step)

                for metric in metrics:
                    metric.set_values(batch, outputs)
                    metric.log_tensorboard(self.summary_writer, self.step)
                    metric.log_info(self.logger, self.step)

                self.summary_writer.add_histogram(
                    "train_batch_size_hist",
                    torch.as_tensor(mean_bsz.values),
                    self.step
                )

                self.summary_writer.add_histogram(
                    "train_batch_sequence_length_hist",
                    torch.as_tensor(mean_seq_length.values),
                    self.step
                )

                end = time.perf_counter()
                self.logger.info(
                    f"[step {self.step}] [train_time {self.step - self.log_interval}\u2192{self.step}] "
                    f"{(end - start) / 60:.2f} minutes"
                )
                self.logger.info(
                    f"[step {self.step}] [epoch {self.epoch + 1}] "
                    f"{logging.eta_minutes_message((end - begin_of_epoch) / 60, self.epoch_step, self.training_steps_per_epoch)}"
                )

                mean_loss.reset()
                mean_bsz.reset()
                mean_forward_pass.reset()
                mean_batch_load.reset()
                mean_seq_length.reset()
                start = end

    def _evaluate_and_checkpoint(self):
        assert self.info.is_main_process, "evaluation should be only done on main process"

        mean_loss = tensorboard.AverageTracker("val_loss", fmt=".2e")

        self.model = self.model.eval()
        self.loss_fn = self.loss_fn.eval()
        metric_cfg = self.cfg["train"].get("metrics")
        if metric_cfg is not None:
            metrics = tensorboard.metrics_from_config(
                metric_cfg, self.input_tokenizer, self.output_tokenizer, prefix="val"
            )
        else:
            metrics = []

        start = time.perf_counter()
        for batch_num, batch in enumerate(self.val_loader):
            inputs, labels = self._prepare_batch(batch=batch)

            with amp.autocast(enabled=self.grad_scaler.is_enabled(), dtype=self.mixed_prec_dtype):
                outputs, loss_dict = self.model(**inputs)
                loss = self.loss_fn(outputs, labels)
                loss = loss + sum(loss_dict.values())

            mean_loss.add(loss.detach())
            if batch_num == 0:
                for metric in metrics:
                    metric.set_values(batch, outputs)
                    metric.log_tensorboard(self.summary_writer, self.step)
                    metric.log_info(self.logger, self.step)

        end = time.perf_counter()

        mean_loss.log_tensorboard(self.summary_writer, self.step)
        mean_loss.log_info(self.logger, self.step)

        self.logger.info(f"[step {self.step}] validation took {(end - start) / 60:.2f} minutes")
        val_loss = mean_loss.value
        ckpt_path = os.path.join(self.directories["checkpoints"], "checkpoint_last.pt")
        io.save_checkpoint(
            checkpoint_path=ckpt_path,
            model=distributed.unwrap_ddp(self.model),
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            loss_fn=self.loss_fn,
            grad_scaler=self.grad_scaler,
            step=self.step,
            epoch=self.epoch,
            epoch_step=self.epoch_step,
            epoch_items=self.epoch_items,
            val_loss=val_loss
        )

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_ckpt_path = os.path.join(self.directories["checkpoints"], "checkpoint_best.pt")
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
                self.logger.info("got termination signal, evaluating and saving on main process before exiting")

        finally:
            if self.info.is_main_process:
                start = time.perf_counter()
                self._evaluate_and_checkpoint()
                end = time.perf_counter()
                self.logger.info(f"final evaluation and checkpointing took {end - start:.2f}s")
