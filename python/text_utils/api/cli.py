import argparse
from io import TextIOWrapper
import sys
import time
import logging
import warnings
from typing import Iterator, Iterable, Union, Optional, Type
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

from text_utils import data, text


class TextProcessingCli:
    text_processor_cls: Type[TextProcessor]
    text_processing_server_cls: Type[TextProcessingServer]

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
            choices=[
                model.name for model in
                cls.text_processor_cls.available_models()
            ],
            default=cls.text_processor_cls.default_model().name,
            help=f"Name of the model to use for {cls.text_processor_cls.task}"
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
            "-p",
            "--process",
            type=str,
            default=None,
            help="Text to process"
        )
        input_group.add_argument(
            "-f",
            "--file",
            type=str,
            default=None,
            help="Path to a text file which will be processed"
        )
        input_group.add_argument(
            "-i",
            "--interactive",
            action="store_true",
            default=None,
            help="Start an interactive session where your command line input is processed"
        )
        parser.add_argument(
            "-if",
            "--input-format",
            choices=cls.text_processor_cls.supported_input_formats(),
            default=cls.text_processor_cls.supported_input_formats()[0],
            help="Format of the text input"
        )
        parser.add_argument(
            "-of",
            "--output-format",
            choices=cls.text_processor_cls.supported_output_formats(),
            default=cls.text_processor_cls.supported_output_formats()[0],
            help="Format of the text output"
        )
        parser.add_argument(
            "-o",
            "--out-path",
            type=str,
            default=None,
            help="Path where processed text should be saved to"
        )
        parser.add_argument(
            "-d",
            "--device",
            type=str,
            nargs="+",
            help="Specify one or more devices to use for inference, by default a single GPU is used if available"
        )
        parser.add_argument(
            "-n",
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
            help="Determines how many inputs will be processed at the same time, larger values should usually result "
            "in faster processing but require more memory"
        )
        batch_limit_group.add_argument(
            "-t",
            "--batch-max-tokens",
            type=int,
            default=None,
            help="Determines the maximum number of tokens processed at the same time, larger values should usually "
            "result in faster processing but require more memory"
        )
        parser.add_argument(
            "-u",
            "--unsorted",
            action="store_true",
            help="Disable sorting of the inputs before processing (for a large number of inputs or large text files "
            "sorting the sequences beforehand leads to speed ups because it minimizes the amount of padding "
            "needed within a batch of sequences)"
        )
        parser.add_argument(
            "-l",
            "--list",
            action="store_true",
            help="List all available models with short descriptions"
        )
        parser.add_argument(
            "--precision",
            choices=["auto", "fp32", "fp16", "bfp16"],
            default="auto",
            help="Choose the precision for inference, fp16 or bfp16 can result in faster runtimes when running on a "
            "new GPU that supports lower precision, but it can be slower on older GPUs. Auto will set the precision to "
            "the precision used for training if it is available, otherwise it will use fp32."
        )
        parser.add_argument(
            "-v",
            "--version",
            action="store_true",
            help=f"Print name and version of the underlying {cls.text_processor_cls.task} library"
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
            help=f"Path to a yaml config file to run a {cls.text_processor_cls.task} server"
        )
        parser.add_argument(
            "--report",
            action="store_true",
            help="Print a runtime report (ignoring startup time) at the end of the processing"
        )
        parser.add_argument(
            "--progress",
            action="store_true",
            help="Show a progress bar while processing"
        )
        parser.add_argument(
            "--log-level",
            type=str,
            choices=["none", "info", "debug"],
            default="none",
            help="Sets the logging level for the underlying loggers"
        )
        parser.add_argument(
            "--lang",
            type=str,
            default=None,
            help="Specify the language of the input, only allowed if the chosen model supports multiple languages. "
            "This language setting is ignored if the input format already specifies a language for each input."
        )
        parser.add_argument(
            "--profile",
            type=str,
            default=None,
            help="Run CLI with cProfile profiler on and output stats to this file"
        )
        return parser

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def version(self) -> str:
        raise NotImplementedError

    def parse_input(self, ipt: str, lang: Optional[str]) -> data.InferenceData:
        item = data.InferenceData.from_str(ipt, self.args.input_format)
        if item.language is None and lang is not None:
            item.language = lang
        return item

    def format_output(self, item: data.InferenceData) -> Iterable[str]:
        return [item.to_str(self.args.output_format)]

    def _run_with_profiling(self, file: str) -> None:
        import cProfile
        cProfile.runctx("self.run()", globals(), locals(), file)

    def process_iter(
        self,
        text_processor: TextProcessor,
        iter: Iterator[data.InferenceData]
    ) -> Iterator[data.InferenceData]:
        raise NotImplementedError

    def process_file(
        self,
        text_processor: TextProcessor,
        path: str,
        lang: Optional[str],
        out_file: Union[str, TextIOWrapper]
    ) -> None:
        raise NotImplementedError

    def setup(self) -> TextProcessor:
        device = self.args.device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        if self.args.experiment:
            cor = self.text_processor_cls.from_experiment(
                experiment_dir=self.args.experiment,
                device=device
            )
        else:
            cor = self.text_processor_cls.from_pretrained(
                model=self.args.model,
                device=device,
                download_dir=self.args.download_dir,
                cache_dir=self.args.cache_dir,
                force_download=self.args.force_download
            )

        if self.args.lang is not None:
            supported_languages = cor.supported_languages()
            assert supported_languages is not None, \
                f"language {self.args.lang} specified but model does not " \
                f"support multiple languages"
            assert self.args.lang in supported_languages, \
                f"the model supports the languages {supported_languages}, " \
                f"but {self.args.lang} was specified"

        if self.args.precision != "auto":
            cor.set_precision(self.args.precision)

        return cor

    @staticmethod
    def inference_data_size(item: data.InferenceData) -> int:
        return len(item.text.encode("utf8"))

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
                    [model.name, model.description, ", ".join(
                        str(tag) for tag in model.tags)]
                    for model in self.text_processor_cls.available_models()
                ],
                alignments=["left", "left", "left"],
                max_column_width=80
            )
            print(table)
            return
        elif self.args.server is not None:
            self.text_processing_server_cls.from_config(self.args.server).run()
            return

        if self.args.log_level == "info":
            logging.basicConfig(level=logging.INFO)
        elif self.args.log_level == "debug":
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.disable(logging.CRITICAL)

        self.cor = self.setup()

        for d in self.cor.devices:
            if d.type == "cuda":
                torch.cuda.reset_peak_memory_stats(d)

        start = time.perf_counter()
        if self.args.process is not None:
            self.args.progress = False
            ipt = self.parse_input(self.args.process, self.args.lang)
            opt = next(self.process_iter(self.cor, iter([ipt])))
            for line in self.format_output(opt):
                print(line)

        elif self.args.file is not None:
            if self.args.out_path is None:
                out = sys.stdout
                assert isinstance(out, TextIOWrapper)
            else:
                assert isinstance(self.args.out_path, str)
                out = self.args.out_path

            self.process_file(self.cor, self.args.file, self.args.lang, out)

            if self.args.report:
                for d in self.cor.devices:
                    if d.type == "cuda":
                        torch.cuda.synchronize(d)
                end = time.perf_counter()

                num_lines, num_bytes = text.file_size(self.args.file)

                report = generate_report(
                    self.cor.task,
                    self.cor.name,
                    self.cor.model,
                    num_lines,
                    num_bytes,
                    end - start,
                    self.cor._precision_dtype,
                    self.args.batch_size,
                    not self.args.unsorted,
                    self.cor.devices,
                    batch_max_tokens=self.args.batch_max_tokens,
                )
                print(report)

        elif self.args.interactive:
            self.args.progress = False
            while True:
                ipt = self.parse_input(input(">> "), self.args.lang)
                opt = next(self.process_iter(self.cor, iter([ipt])))
                for line in self.format_output(opt):
                    print(line)

        else:
            if sys.stdin.isatty():
                return

            try:
                if self.args.unsorted:
                    # correct lines from stdin as they come
                    input_it = (
                        self.parse_input(line.rstrip("\r\n"), self.args.lang)
                        for line in sys.stdin
                    )
                    sized_it = ProgressIterator(
                        input_it,
                        self.inference_data_size
                    )
                    outputs = self.process_iter(self.cor, sized_it)
                    for opt in outputs:
                        for line in self.format_output(opt):
                            print(line)
                else:
                    # read stdin completely, then potentially sort and correct
                    inputs = [
                        self.parse_input(line.rstrip("\r\n"), self.args.lang)
                        for line in sys.stdin
                    ]
                    sized_it = ProgressIterator(
                        iter(inputs),
                        self.inference_data_size
                    )
                    outputs = self.process_iter(self.cor, sized_it)
                    for opt in outputs:
                        for line in self.format_output(opt):
                            print(line)

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
                        self.cor._precision_dtype,
                        self.args.batch_size,
                        not self.args.unsorted,
                        self.cor.devices,
                        batch_max_tokens=self.args.batch_max_tokens,
                    )
                    print(report)

            except BrokenPipeError:
                pass
