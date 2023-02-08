import argparse
from io import TextIOWrapper
import sys
import time
import logging
from typing import Iterator, Union, Optional, Type

import torch

from text_correction_utils.api.corrector import TextCorrector
from text_correction_utils.api.server import TextCorrectionServer
from text_correction_utils.api.table import generate_report, generate_table
from text_correction_utils.api.utils import ProgressIterator

from text_correction_utils import data, text


class TextCorrectionCli:
    text_corrector_cls: Type[TextCorrector]
    text_correction_server_cls: Type[TextCorrectionServer]

    @classmethod
    def _default_model(cls) -> str:
        available_models = cls.text_corrector_cls.available_models()
        for info in available_models:
            if "default" in info.tags:
                return info.name
        return available_models[0].name

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
            default=cls._default_model(),
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
        input_group.add_argument(
            "-i",
            "--interactive",
            action="store_true",
            default=None,
            help="Start an interactive session where your command line input is corrected"
        )
        parser.add_argument(
            "-if",
            "--input-format",
            choices=cls.text_corrector_cls.supported_input_formats(),
            default=cls.text_corrector_cls.supported_input_formats()[0],
            help="Format of the text input"
        )
        parser.add_argument(
            "-of",
            "--output-format",
            choices=cls.text_corrector_cls.supported_output_formats(),
            default=cls.text_corrector_cls.supported_output_formats()[0],
            help="Format of the text output"
        )
        parser.add_argument(
            "-o",
            "--out-path",
            type=str,
            default=None,
            help="Path where corrected text should be saved to"
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
            help=f"Path to a yaml config file to run a {cls.text_corrector_cls.task} server"
        )
        parser.add_argument(
            "--report",
            action="store_true",
            help="Print a runtime report (ignoring startup time) at the end of the correction procedure"
        )
        parser.add_argument(
            "--progress",
            action="store_true",
            help="Show a progress bar while correcting"
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
            help="Specify the language of the input, only allowed if the chosen model supports multiple languages. \
            This language setting is ignored if the input format already specifies a language for each input."
        )
        parser.add_argument(
            "--profile",
            type=str,
            default=None,
            help="Profile the cli run and save the results to this file"
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

    def correct_iter(
        self,
        corrector: TextCorrector,
        iter: Iterator[data.InferenceData]
    ) -> Iterator[data.InferenceData]:
        raise NotImplementedError

    def correct_file(
        self,
        corrector: TextCorrector,
        path: str,
        lang: Optional[str],
        out_file: Union[str, TextIOWrapper]
    ):
        raise NotImplementedError

    def _run_with_profiling(self, file: str) -> None:
        import cProfile
        cProfile.runctx("self.run()", globals(), locals(), file)

    @staticmethod
    def inference_data_size(item: data.InferenceData) -> int:
        return len(item.text.encode("utf8"))

    def run(self) -> None:
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
                    [model.name, model.description, ", ".join(str(tag) for tag in model.tags)]
                    for model in self.text_corrector_cls.available_models()
                ],
                alignments=["left", "left", "left"],
                max_column_width=80
            )
            print(table)
            return
        elif self.args.server is not None:
            self.text_correction_server_cls.from_config(self.args.server).run()
            return

        if self.args.log_level == "info":
            logging.basicConfig(level=logging.INFO)
        elif self.args.log_level == "debug":
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.disable(logging.CRITICAL)

        device = "cpu" if self.args.cpu else "cuda"
        if self.args.experiment:
            cor = self.text_corrector_cls.from_experiment(
                experiment_dir=self.args.experiment,
                device=device
            )
        else:
            cor = self.text_corrector_cls.from_pretrained(
                model=self.args.model,
                device=device,
                download_dir=self.args.download_dir,
                cache_dir=self.args.cache_dir,
                force_download=self.args.force_download
            )

        if self.args.lang is not None:
            supported_languages = cor.supported_languages()
            assert supported_languages is not None, f"language {self.args.lang} specified but model does not \
support multiple languages"
            assert self.args.lang in supported_languages, f"the model supports the languages {supported_languages}, \
but {self.args.lang} was specified"

        if self.args.precision != "auto":
            cor.set_precision(self.args.precision)

        is_cuda = cor.device.type == "cuda"

        if is_cuda:
            torch.cuda.reset_peak_memory_stats(cor.device)

        start = time.perf_counter()
        if self.args.correct is not None:
            ipt = self.parse_input(self.args.correct, self.args.lang)
            opt = next(self.correct_iter(cor, iter([ipt])))
            print(opt.to_str(self.args.output_format))

        elif self.args.file is not None:
            if self.args.out_path is None:
                out = sys.stdout
                assert isinstance(out, TextIOWrapper)
            else:
                assert isinstance(self.args.out_path, str)
                out = self.args.out_path

            self.correct_file(cor, self.args.file, self.args.lang, out)

            if self.args.report:
                if is_cuda:
                    torch.cuda.synchronize(cor.device)
                end = time.perf_counter()

                num_lines, num_bytes = text.file_size(self.args.file)

                report = generate_report(
                    cor.task,
                    cor.name,
                    cor.model,
                    num_lines,
                    num_bytes,
                    end - start,
                    cor._mixed_precision_dtype,
                    self.args.batch_size,
                    not self.args.unsorted,
                    cor.device,
                )
                print(report)

        elif self.args.interactive:
            while True:
                try:
                    ipt = self.parse_input(input(), self.args.lang)
                    opt = next(self.correct_iter(cor, iter([ipt])))
                    print(opt.to_str(self.args.output_format))
                except KeyboardInterrupt:
                    pass

        else:
            if sys.stdin.isatty():
                return

            try:
                if self.args.unsorted:
                    # correct lines from stdin as they come
                    input_it = (self.parse_input(line.strip(), self.args.lang) for line in sys.stdin)
                    sized_it = ProgressIterator(input_it, self.inference_data_size)
                    outputs = self.correct_iter(cor, sized_it)
                    for opt in outputs:
                        print(opt.to_str(self.args.output_format))
                else:
                    # read stdin completely, then potentially sort and correct
                    inputs = [self.parse_input(line.strip(), self.args.lang) for line in sys.stdin]
                    sized_it = ProgressIterator(iter(inputs), self.inference_data_size)
                    outputs = self.correct_iter(cor, sized_it)
                    for opt in outputs:
                        print(opt.to_str(self.args.output_format))

                if self.args.report:
                    if is_cuda:
                        torch.cuda.synchronize(cor.device)

                    report = generate_report(
                        cor.task,
                        cor.name,
                        cor.model,
                        sized_it.num_items,
                        sized_it.total_size,
                        time.perf_counter() - start,
                        cor._mixed_precision_dtype,
                        self.args.batch_size,
                        not self.args.unsorted,
                        cor.device,
                    )
                    print(report)

            except BrokenPipeError:
                pass
