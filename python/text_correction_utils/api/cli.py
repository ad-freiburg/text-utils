import argparse
import sys
import time
import logging
from typing import Any, Iterator, Tuple, Generator

import torch

from text_correction_utils.api.corrector import TextCorrector
from text_correction_utils.api.server import TextCorrectionServer
from text_correction_utils.api.table import generate_report


class TextCorrectionCli:
    text_corrector_cls: TextCorrector
    text_correction_server_cls: TextCorrectionServer

    @classmethod
    def parser(
        cls,
        name: str,
        description: str
    ) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(name, description)
        model_group = parser.add_mutually_exclusive_group()
        model_group.add_argument(
            "-m",
            "--model",
            choices=[model.name for model in cls.text_corrector_cls.available_models()],
            default=cls.text_corrector_cls.available_models()[0].name,
            help=f"Name of the model to use for {cls.text_corrector_cls.task}"
        )
        model_group.add_argument(
            "-e",
            "--experiment",
            type=str,
            default=None,
            help="Path to an experiment directory from which the model will be loaded "
                 "(use this when you trained your own model and want to use it)"
        )
        input_group = parser.add_mutually_exclusive_group()
        input_group.add_argument(
            "-c",
            "--correct",
            type=str,
            default=None,
            help="Text to correct"
        )
        input_group.add_argument(
            "-f",
            "--file",
            type=str,
            default=None,
            help="Path to a text file which will be corrected line by line"
        )
        parser.add_argument(
            "-o",
            "--out-path",
            type=str,
            default=None,
            help="Path where corrected text should be saved to"
        )
        input_group.add_argument(
            "-i",
            "--interactive",
            action="store_true",
            default=None,
            help="Start an interactive session where your command line input is corrected"
        )
        parser.add_argument(
            "--cpu",
            action="store_true",
            help="Force to run the model on CPU, by default a GPU is used if available"
        )
        parser.add_argument(
            "--num-threads",
            type=int,
            default=None,
            help="Number of threads used for running the inference pipeline"
        )
        batch_limit_group = parser.add_mutually_exclusive_group()
        batch_limit_group.add_argument(
            "-b",
            "--batch-size",
            type=int,
            default=16,
            help="Determines how many inputs will be corrected at the same time, larger values should usually result in "
                 "faster correcting but require more memory"
        )
        batch_limit_group.add_argument(
            "-t",
            "--batch-max-tokens",
            type=int,
            default=None,
            help="Determines the maximum number of tokens corrected at the same time, larger values should usually result in "
                 "faster correcting but require more memory"
        )
        parser.add_argument(
            "-u",
            "--unsorted",
            action="store_true",
            help="Disable sorting of the inputs before correcting (for a large number of inputs or large text files "
                 "sorting the sequences beforehand leads to speed ups because it minimizes the amount of padding needed "
                 "within a batch of sequences)"
        )
        parser.add_argument(
            "-l",
            "--list",
            action="store_true",
            help="List all available models with short descriptions"
        )
        parser.add_argument(
            "-p",
            "--pipe",
            action="store_true",
            help=f"Pass this flag when using {cls.text_corrector_cls.task} in a pipe because input and output is then treated as an "
                 "iterator"
        )
        parser.add_argument(
            "--precision",
            choices=["fp32", "fp16", "bfp16"],
            default="fp32",
            help="Choose the precision for inference, fp16 or bfp16 can result in faster runtimes when running on a "
                 "new GPU that supports lower precision, but it can be slower on older GPUs"
        )
        parser.add_argument(
            "-v",
            "--version",
            action="store_true",
            help=f"Print name and version of the underlying {cls.text_corrector_cls.task} library"
        )
        parser.add_argument(
            "--force-download",
            action="store_true",
            help="Download the model again even if it already was downloaded"
        )
        parser.add_argument(
            "--download-dir",
            type=str,
            default=None,
            help="Directory the model will be downloaded to (as zip file)"
        )
        parser.add_argument(
            "--cache-dir",
            type=str,
            default=None,
            help="Directory the downloaded model will be extracted to"
        )
        parser.add_argument(
            "--server",
            type=str,
            default=None,
            help=f"Path to a JSON config file to run a {cls.text_corrector_cls.task} server"
        )
        parser.add_argument(
            "--report",
            type=str,
            default=None,
            help="Save a runtime report (ignoring startup time) formatted as markdown table to a file, append new line "
                 "if the file already exists"
        )
        parser.add_argument(
            "--log-level",
            type=str,
            choices=["none", "info", "debug"],
            default="none",
            help="Sets the logging level for the underlying loggers"
        )
        return parser

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def version(self) -> str:
        raise NotImplementedError

    def parse_input(self, ipt: str) -> Any:
        raise NotImplementedError

    def format_output(self, pred: Any) -> str:
        raise NotImplementedError

    def correct_iter(self, corrector: TextCorrector, iter: Iterator[str]) -> Generator[str, None, Tuple[int, int]]:
        raise NotImplementedError

    def correct_file(self, corrector: TextCorrector, path: str) -> Generator[str, None, Tuple[int, int]]:
        raise NotImplementedError

    def correct_single(self, corrector: TextCorrector, s: str) -> str:
        return self.correct_iter(corrector, iter([s]))

    def run(self, args: argparse.Namespace):
        if args.version:
            print(self.version())
            return
        elif args.list:
            model_names = []
            model_descriptions = []
            for model in self.text_corrector_cls.available_models():
                model_names.append(model.name)
                model_descriptions.append(model.description)
            max_model_name_len = max(len(n) for n in model_names)
            print("\n".join(
                f"{name.ljust(max_model_name_len)}: {desc}"
                for name, desc in zip(model_names, model_descriptions)
            ))
            return
        elif args.server:
            self.text_correction_server_cls.from_config(args.server).run()
            return

        if args.log_level == "info":
            logging.basicConfig(level=logging.INFO)
        elif args.log_level == "debug":
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.disable(logging.CRITICAL)

        device = "cpu" if args.cpu else "cuda"
        if args.experiment:
            cor = self.text_corrector_cls.from_experiment(
                experiment_dir=args.experiment,
                device=device
            )
        else:
            cor = self.text_corrector_cls.from_pretrained(
                model=args.model,
                device=device,
                download_dir=args.download_dir,
                cache_dir=args.cache_dir,
                force_download=args.force_download
            )

        cor.set_precision(args.precision)
        is_cuda = cor.device.type == "cuda"

        if is_cuda:
            torch.cuda.reset_peak_memory_stats(cor.device)

        start = time.perf_counter()
        if args.correct is not None:
            print(self.correct_single(cor, args.correct))

        elif args.file is not None:
            corrected, num_bytes = self.correct_file(args.file)
            if args.out_path is None:
                for line in corrected:
                    print(line)
            if args.report:
                if is_cuda:
                    torch.cuda.synchronize(cor.device)
                end = time.perf_counter()

                with open(args.file, "r", encoding="utf8") as inf:
                    lines = [line.strip() for line in inf]

                generate_report(
                    cor.task,
                    cor.name,
                    cor.model,
                    lines,
                    end - start,
                    cor._mixed_precision_dtype,
                    args.batch_size,
                    not args.unsorted,
                    cor.device,
                    file_path=args.report
                )

        elif args.interactive:
            while True:
                try:
                    line = input()
                    corrected = self.correct_single(line)
                    print(self.format_output(corrected))
                except KeyboardInterrupt:
                    pass

        else:
            if sys.stdin.isatty():
                return

            try:
                if args.pipe:
                    # correct lines from stdin as they come
                    for corrected in self.correct_iter(sys.stdin):
                        print(self.format_output(corrected))
                else:
                    # read stdin completely, then potentially sort and correct
                    lines = [line.strip() for line in sys.stdin]
                    for corrected in self.correct_iter(iter(lines)):
                        print(self.format_output(corrected))

                    if args.report:
                        if is_cuda:
                            torch.cuda.synchronize(cor.device)
                        generate_report(
                            cor.task,
                            cor.name,
                            cor.model,
                            lines,
                            time.perf_counter() - start,
                            cor._mixed_precision_dtype,
                            args.batch_size,
                            not args.unsorted,
                            cor.device,
                            file_path=args.report
                        )

            except BrokenPipeError:
                pass
