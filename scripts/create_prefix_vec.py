import time
import argparse
import os


from text_correction_utils import prefix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def create(args: argparse.Namespace):
    start = time.perf_counter()
    trie = prefix.Vec.from_file(args.file)
    end = time.perf_counter()
    print(f"Prefix vec built in {end - start:.2f} seconds")
    start = time.perf_counter()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    trie.save(args.out)
    end = time.perf_counter()
    print(f"Prefix vec saved in {end - start:.2f} seconds")
    start = time.perf_counter()
    prefix.Vec.load(args.out)
    end = time.perf_counter()
    print(f"Prefix vec loaded in {end - start:.2f} seconds")


if __name__ == "__main__":
    create(parse_args())
