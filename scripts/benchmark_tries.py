import time
import argparse

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
    with open(args.file) as f:
        for line in f:
            line = line.strip()
            split = line.split("\t")
            pairs.append((split[0], split[1]))

    if args.limit is not None:
        pairs = pairs[:args.limit]

    print("Building prefix tree...")
    start = time.perf_counter()
    trie = prefix_tree.PrefixTree()
    for property, name in tqdm(pairs, "creating prefix tree", leave=False):
        trie.insert(name, property)
    end = time.perf_counter()
    print(f"Prefix tree built in {end - start:.2f} seconds")
    print(f"Prefix tree contains {trie.size()} nodes/values")

    print("Benchmarking prefix tree...")
    start = time.perf_counter()
    for property, name in pairs:
        trie.get(name)
    end = time.perf_counter()
    print(f"Get value in {1000 * (end - start):.2f}ms")

    start = time.perf_counter()
    for property, name in pairs:
        trie.contains_prefix(name)
    end = time.perf_counter()
    print(f"Check for prefix in {1000 * (end - start):.2f}ms")

    print("Building marisa trie...")
    start = time.perf_counter()
    trie = marisa_trie.BytesTrie(
        ((name, property.encode())
         for property, name in pairs)
    )
    end = time.perf_counter()
    print(f"Marisa trie built in {end - start:.2f} seconds")

    print("Benchmarking marisa trie...")
    start = time.perf_counter()
    for property, name in pairs:
        trie.get(name)[0]
    end = time.perf_counter()
    print(f"Get value in {1000 * (end - start):.2f}ms")

    start = time.perf_counter()
    for property, name in pairs:
        trie.keys(name)
    end = time.perf_counter()
    print(f"Check for prefix in {1000 * (end - start):.2f}ms")


if __name__ == "__main__":
    benchmark(parse_args())
