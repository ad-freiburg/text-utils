import argparse
import logging
import os

from text_utils import tokenization
from text_utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--files",
        type=str,
        nargs="+",
        required=True,
        help="Files to train BPE on",
    )
    parser.add_argument(
        "--vocab-size", type=int, required=True, help="Vocabulary size for BPE"
    )
    parser.add_argument(
        "--num-special-tokens", type=int, required=True, help="Number of special tokens"
    )
    parser.add_argument("-o", "--out", type=str, required=True, help="Output file")
    parser.add_argument(
        "--max-lines-per-file", type=int, default=None, help="Max lines per file"
    )
    parser.add_argument(
        "--normalization",
        choices=["nfc", "nfd", "nfkc", "nfkd"],
        default="nfkc",
        help="Normalization to apply to the input",
    )
    parser.add_argument(
        "-n", "--num-threads", type=int, default=None, help="Number of threads"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    return parser.parse_args()


def train_bpe(args: argparse.Namespace):
    if args.num_threads is None:
        args.num_threads = min(len(os.sched_getaffinity(0)), 4)

    setup_logging(logging.CRITICAL if args.no_progress else logging.INFO)

    tokenization.train_bpe(
        args.files,
        args.vocab_size,
        args.num_special_tokens,
        args.out,
        args.max_lines_per_file,
        args.normalization,
        args.num_threads,
        not args.no_progress,
    )


def main():
    train_bpe(parse_args())
