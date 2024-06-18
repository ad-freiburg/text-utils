import time
import argparse
import os


from text_utils import continuations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=str, required=True)
    parser.add_argument("-o", "--output-dir", type=str, required=True)
    parser.add_argument("-sfx", "--common-suffix", type=str, default=None)
    return parser.parse_args()


def create(args: argparse.Namespace):
    os.makedirs(args.output_dir, exist_ok=True)

    start = time.perf_counter()
    continuations.MmapContinuationIndex.build_from_file(
        args.input_file,
        args.output_dir,
        args.common_suffix
    )
    end = time.perf_counter()
    print(f"Continuation index built in {end - start:.2f} seconds")

    start = time.perf_counter()
    continuations.MmapContinuationIndex.load(
        args.input_file,
        args.output_dir,
        args.common_suffix
    )
    end = time.perf_counter()
    print(f"Continuation index loaded in {end - start:.2f} seconds")


def main():
    create(parse_args())
