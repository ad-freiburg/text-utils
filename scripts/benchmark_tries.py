import time
import argparse
import random

from tqdm import tqdm
from text_correction_utils import prefix_tree
import marisa_trie


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def benchmark(args: argparse.Namespace):
    pairs = []
    with open(args.file, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            split = line.split("\t")
            try:
                pairs.append((split[0], split[1]))
            except IndexError:
                print(f"error in line {i}:\n{line}\n{split}")
                exit(1)

    if args.limit is not None:
        pairs = pairs[:args.limit]

    sub_pairs = pairs[:32_000]
    random.shuffle(sub_pairs)

    def rand_sub_of_length(s: str, n: int) -> str:
        idx = random.randint(0, max(0, len(s) - n))
        return s[idx: idx + n]

    continuations = [
        rand_sub_of_length(name, 4)
        for _, name in sub_pairs
    ]
    n = 1_000
    random.shuffle(pairs)

    start = time.perf_counter()
    trie = prefix_tree.Tree()
    for name, identifier in tqdm(pairs, "creating prefix tree", leave=False):
        trie.insert(name, int(identifier[1:]))
    end = time.perf_counter()
    print(f"Rust prefix tree: Built in {end - start:.2f} seconds")

    start = time.perf_counter()
    for name, _ in tqdm(pairs, "getting values", leave=False):
        trie.get(name)
    end = time.perf_counter()
    print(f"Rust prefix tree: Get value in {1000 * (end - start):.2f}ms")

    start = time.perf_counter()
    for name, _ in tqdm(pairs, "checking prefixes", leave=False):
        trie.contains_prefix(name)
    end = time.perf_counter()
    print(f"Rust prefix tree: Check for prefix in {1000 * (end - start):.2f}ms")

    start = time.perf_counter()
    for name, _ in tqdm(pairs[:n], "checking continuations", leave=False):
        for cont in continuations:
            trie.contains_prefix(name + cont)
    end = time.perf_counter()
    runtime = 1000 * (end - start)
    print(f"Rust prefix tree: Check for continuations without subtree indexing in {runtime:.2f}ms "
          f"({runtime / min(len(pairs), n):.2f}ms on avg)")

    start = time.perf_counter()
    for name, _ in tqdm(pairs[:n], "checking continuations", leave=False):
        trie.contains_continuations(name, continuations)
    end = time.perf_counter()
    runtime = 1000 * (end - start)
    print(f"Rust prefix tree: Check for continuations with subtree indexing in {runtime:.2f}ms "
          f"({runtime / min(len(pairs), n):.2f}ms on avg)")

    start = time.perf_counter()
    trie = marisa_trie.BytesTrie(
        ((name, identifier.encode())
         for name, identifier in pairs)
    )
    end = time.perf_counter()
    print(f"Marisa trie: Built in {end - start:.2f} seconds")
    start = time.perf_counter()
    for name, _ in pairs:
        trie.get(name)[0]
    end = time.perf_counter()
    print(f"Marisa trie: Get value in {1000 * (end - start):.2f}ms")

    start = time.perf_counter()
    for name, _ in pairs:
        trie.keys(name)
    end = time.perf_counter()
    print(f"Marisa trie: Check for prefix in {1000 * (end - start):.2f}ms")


if __name__ == "__main__":
    benchmark(parse_args())
