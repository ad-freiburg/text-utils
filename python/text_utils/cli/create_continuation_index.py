import time
import argparse
import os


from text_utils import continuations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=str, required=True)
    parser.add_argument("-o", "--output-file", type=str, required=True)
    return parser.parse_args()


def create(args: argparse.Namespace):
    dir = os.path.dirname(args.output_file)
    if dir != "":
        os.makedirs(dir, exist_ok=True)

    start = time.perf_counter()
    continuations.Continuations.build_from_file(
        args.input_file,
        args.output_file
    )
    end = time.perf_counter()
    print(f"Continuations trie built in {end - start:.2f} seconds")

    start = time.perf_counter()
    # empty continuations for testing
    conts = []
    continuations.Continuations.load_with_continuations(
        args.output_file,
        conts
    )
    end = time.perf_counter()
    print(f"Continuations trie loaded in {end - start:.2f} seconds")


def main():
    create(parse_args())
