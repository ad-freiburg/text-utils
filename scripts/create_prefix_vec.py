import time
import argparse
import os


from text_utils import prefix, configuration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--tokenizer-cfg", type=str, default=None)
    return parser.parse_args()


def create(args: argparse.Namespace):
    start = time.perf_counter()
    if args.tokenizer_cfg is not None:
        tokenizer_cfg = configuration.load_config(args.tokenizer_cfg)
    else:
        tokenizer_cfg = None
    trie = prefix.Vec.from_file(args.file, tokenizer_cfg)
    end = time.perf_counter()
    print(f"Prefix vec built in {end - start:.2f} seconds")
    start = time.perf_counter()
    dir = os.path.dirname(args.out)
    if dir != "":
        os.makedirs(dir, exist_ok=True)
    trie.save(args.out)
    end = time.perf_counter()
    print(f"Prefix vec saved in {end - start:.2f} seconds")
    start = time.perf_counter()
    prefix.Vec.load(args.out)
    end = time.perf_counter()
    print(f"Prefix vec loaded in {end - start:.2f} seconds")


if __name__ == "__main__":
    create(parse_args())
