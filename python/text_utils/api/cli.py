import argparse
import logging
import sys
import time
import warnings
from io import TextIOWrapper
from typing import Any, Iterator, Type

try:
    import readline  # noqa
except ImportError:
    # readline is e.g. not available on Windows
    pass

import torch

from text_utils.api.processor import TextProcessor
from text_utils.api.server import TextProcessingServer
from text_utils.api.table import generate_report, generate_table
from text_utils.api.utils import ProgressIterator
from text_utils.logging import disable_logging, setup_logging


class TextProcessingCli:
    text_processor_cls: Type[TextProcessor]
    text_processing_server_cls: Type[TextProcessingServer]

    @classmethod
    def parser(cls, name: str, description: str) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(name, description)
        model_group = parser.add_mutually_exclusive_group()
        default_model = cls.text_processor_cls.default_model()
        model_group.add_argument(
            "-m",
            "--model",
            choices=[model.name for model in cls.text_processor_cls.available_models()],
            default=None if default_model is None else default_model.name,
            help=f"Name of the model to use for {cls.text_processor_cls.task}",
        )
        model_group.add_argument(
            "-e",
            "--experiment",
            type=str,
            default=None,
            help="Path to an experiment directory from which the model will be loaded "
            "(use this when you trained your own model and want to use it)",
        )
        parser.add_argument(
            "--last",
            action="store_true",
            help="Use last checkpoint instead of best, only works with experiments",
        )
        input_group = parser.add_mutually_exclusive_group()
        input_group.add_argument(
            "-p", "--process", type=str, default=None, help="Text to process"
        )
        input_group.add_argument(
            "-f",
            "--file",
            type=str,
            default=None,
            help="Path to a text file which will be processed",
        )
        input_group.add_argument(
            "-i",
            "--interactive",
            action="store_true",
            default=None,
            help="Start an interactive session where your command line input is processed",
        )
        parser.add_argument(
            "-o",
            "--out-path",
            type=str,
            default=None,
            help="Path where processed text should be saved to",
        )
        parser.add_argument(
            "-d",
            "--device",
            type=str,
            nargs="+",
            help="Specify one or more devices to use for inference, by default a single GPU is used if available",
        )
        parser.add_argument(
            "-n",
            "--num-threads",
            type=int,
            default=None,
            help="Number of threads used for running the inference pipeline",
        )
        batch_limit_group = parser.add_mutually_exclusive_group()
        batch_limit_group.add_argument(
            "-b",
            "--batch-size",
            type=int,
            default=16,
            help="Determines how many inputs will be processed at the same time, larger values should usually result "
            "in faster processing but require more memory",
        )
        batch_limit_group.add_argument(
            "-t",
            "--batch-max-tokens",
            type=int,
            default=None,
            help="Determines the maximum number of tokens processed at the same time, larger values should usually "
            "result in faster processing but require more memory",
        )
        parser.add_argument(
            "-u",
            "--unsorted",
            action="store_true",
            help="Disable sorting of the inputs before processing (for a large number of inputs or large text files "
            "sorting the sequences beforehand leads to speed ups because it minimizes the amount of padding "
            "needed within a batch of sequences)",
        )
        parser.add_argument(
            "-l",
            "--list",
            action="store_true",
            help="List all available models with short descriptions",
        )
        parser.add_argument(
            "-v",
            "--version",
            action="store_true",
            help=f"Print name and version of the underlying {cls.text_processor_cls.task} library",
        )
        parser.add_argument(
            "--force-download",
            action="store_true",
            help="Download the model again even if it already was downloaded",
        )
        parser.add_argument(
            "--download-dir",
            type=str,
            default=None,
            help="Directory the model will be downloaded to (as zip file)",
        )
        parser.add_argument(
            "--cache-dir",
            type=str,
            default=None,
            help="Directory the downloaded model will be extracted to",
        )
        parser.add_argument(
            "--server",
            type=str,
            default=None,
            help=f"Path to a yaml config file to run a {cls.text_processor_cls.task} server",
        )
        parser.add_argument(
            "--report",
            action="store_true",
            help="Print a runtime report (ignoring startup time) at the end of the processing",
        )
        parser.add_argument(
            "--progress",
            action="store_true",
            help="Show a progress bar while processing",
        )
        parser.add_argument(
            "--log-level",
            type=str,
            choices=list(logging._nameToLevel),
            default=None,
            help="Sets the logging level for the underlying loggers",
        )
        parser.add_argument(
            "--profile",
            type=str,
            default=None,
            help="Run CLI with cProfile profiler on and output stats to this file",
        )
        return parser

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def version(self) -> str:
        raise NotImplementedError

    def _run_with_profiling(self, file: str) -> None:
        import cProfile

        cProfile.runctx("self.run()", globals(), locals(), file)

    def process_iter(
        self, processor: TextProcessor, iter: Iterator[str]
    ) -> Iterator[Any]:
        raise NotImplementedError

    def setup(self) -> TextProcessor:
        device = self.args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.args.experiment:
            cor = self.text_processor_cls.from_experiment(
                experiment_dir=self.args.experiment,
                device=device,
                last=self.args.last,
            )
        else:
            cor = self.text_processor_cls.from_pretrained(
                model=self.args.model,
                device=device,
                download_dir=self.args.download_dir,
                cache_dir=self.args.cache_dir,
                force_download=self.args.force_download,
            )

        return cor

    @staticmethod
    def input_size(item: str) -> int:
        return len(item.encode("utf8"))

    def run(self) -> None:
        # ignore warnings in CLI, so the user doesn't get confused
        warnings.filterwarnings("ignore")

        if self.args.profile is not None:
            file = self.args.profile
            self.args.profile = None
            return self._run_with_profiling(file)
        elif self.args.version:
            print(self.version())
            return
        elif self.args.list:
            table = generate_table(
                headers=[["Model", "Description", "Tags"]],
                data=[
                    [
                        model.name,
                        model.description,
                        ", ".join(str(tag) for tag in model.tags),
                    ]
                    for model in self.text_processor_cls.available_models()
                ],
                alignments=["left", "left", "left"],
                max_column_width=80,
            )
            print(table)
            return
        elif self.args.server is not None:
            setup_logging(self.args.log_level or logging.INFO)
            self.text_processing_server_cls.from_config(self.args.server).run()
            return

        if self.args.log_level is not None:
            # set template for logging
            setup_logging(self.args.log_level.upper())
        else:
            disable_logging()

        self.cor = self.setup()

        for d in self.cor.devices:
            if d.type == "cuda":
                torch.cuda.reset_peak_memory_stats(d)

        start = time.perf_counter()
        if self.args.process is not None:
            self.args.progress = False
            for output in self.process_iter(self.cor, iter([self.args.process])):
                print(output)

        elif self.args.file is not None:
            if self.args.out_path is None:
                out = sys.stdout
                assert isinstance(out, TextIOWrapper)
            else:
                assert isinstance(self.args.out_path, str)
                out = open(self.args.out_path, "w")

            input_it = (line.rstrip("\r\n") for line in open(self.args.file))
            sized_it = ProgressIterator(input_it, self.input_size)
            for output in self.process_iter(self.cor, sized_it):
                out.write(output + "\n")

            out.close()

            if self.args.report:
                for d in self.cor.devices:
                    if d.type == "cuda":
                        torch.cuda.synchronize(d)
                end = time.perf_counter()

                report = generate_report(
                    self.cor.task,
                    self.cor.name,
                    self.cor.model,
                    sized_it.num_items,
                    sized_it.total_size,
                    end - start,
                    self.args.batch_size,
                    not self.args.unsorted,
                    self.cor.devices,
                    next(self.cor.model.parameters()).dtype,
                    self.args.batch_max_tokens,
                )
                print(report)

        elif self.args.interactive:
            self.args.progress = False
            while True:
                ipt = input(">> ")
                for output in self.process_iter(self.cor, iter([ipt])):
                    print(output, flush=self.args.unsorted)

        else:
            if sys.stdin.isatty():
                return

            try:
                # correct lines from stdin as they come
                input_it = (line.rstrip("\r\n") for line in sys.stdin)
                sized_it = ProgressIterator(input_it, self.input_size)
                for output in self.process_iter(self.cor, sized_it):
                    print(output, flush=self.args.unsorted)

                if self.args.report:
                    for d in self.cor.devices:
                        if d.type == "cuda":
                            torch.cuda.synchronize(d)

                    report = generate_report(
                        self.cor.task,
                        self.cor.name,
                        self.cor.model,
                        sized_it.num_items,
                        sized_it.total_size,
                        time.perf_counter() - start,
                        self.args.batch_size,
                        not self.args.unsorted,
                        self.cor.devices,
                        next(self.cor.model.parameters()).dtype,
                        self.args.batch_max_tokens,
                    )
                    print(report)

            except BrokenPipeError:
                pass
