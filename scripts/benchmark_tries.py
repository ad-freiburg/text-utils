import time
import argparse
import random

from tqdm import tqdm
from text_correction_utils import prefix
import marisa_trie


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--impl", choices=["vec", "tree"], default="vec")
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def benchmark(args: argparse.Namespace):
    pairs = []
    with open(args.file, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            split = line.split("\t")
            try:
                pairs.append((split[0].encode("utf8"), split[1]))
            except IndexError:
                print(f"error in line {i}:\n{line}\n{split}")
                exit(1)
            if i >= 32_000:
                break

    random.shuffle(pairs)

    def rand_sub_of_length(s: str, n: int) -> str:
        idx = random.randint(0, max(0, len(s) - n))
        return s[idx: idx + n]

    continuations = [
        rand_sub_of_length(name, 4)
        for name, _ in pairs
    ]
    n = 1_000
    random.shuffle(pairs)

    cl = prefix.Vec if args.impl == "vec" else prefix.Tree

    start = time.perf_counter()
    trie = cl.from_file(args.file)
    end = time.perf_counter()
    print(f"Rust prefix tree: Built in {end - start:.2f} seconds")
    if args.impl == "vec":
        trie.save(args.file + ".bin")
        start = time.perf_counter()
        trie = cl.load(args.file + ".bin")
        end = time.perf_counter()
        print(f"Rust prefix tree: Loaded in {end - start:.2f} seconds")
    trie.set_continuations(continuations)

    if args.interactive:
        while True:
            name = input("Enter prefix: ")
            if name == "exit":
                break
            print(f"Is prefix: {trie.contains(name)}")
            print(f"Has value: {trie.get(name)}")
            conts = trie.get_continuations(name)
            print(f"{len(conts):,} continuations: {conts[:50]}")
        return

    # start = time.perf_counter()
    # for name, _ in tqdm(pairs, "getting values", leave=False):
    #     trie.get(name)
    # end = time.perf_counter()
    # print(f"Rust prefix tree: Get value in {1000 * (end - start):.2f}ms")
    #
    # start = time.perf_counter()
    # for name, _ in tqdm(pairs, "checking prefixes", leave=False):
    #     trie.contains(name)
    # end = time.perf_counter()
    # print(f"Rust prefix tree: Check for prefix in {1000 * (end - start):.2f}ms")

    # start = time.perf_counter()
    # for name, _ in tqdm(pairs[:n], "checking continuations", leave=False):
    #     for cont in continuations:
    #         trie.contains(name + cont)
    # end = time.perf_counter()
    # runtime = 1000 * (end - start)
    # print(f"Rust prefix tree: Check for continuations without subtree indexing in {runtime:.2f}ms "
    #       f"({runtime / min(len(pairs), n):.2f}ms on avg)")

    start = time.perf_counter()
    for name, _ in tqdm(pairs[:n], "checking continuations", leave=False):
        trie.contains_continuations(name)
    end = time.perf_counter()
    runtime = 1000 * (end - start)
    print(f"Rust prefix tree: Check for continuations with subtree indexing in {runtime:.2f}ms "
          f"({runtime / min(len(pairs), n):.2f}ms on avg)")

    # start = time.perf_counter()
    # trie = marisa_trie.BytesTrie(
    #     ((name, identifier.encode())
    #      for name, identifier in pairs)
    # )
    # end = time.perf_counter()
    # print(f"Marisa trie: Built in {end - start:.2f} seconds")
    # start = time.perf_counter()
    # for name, _ in pairs:
    #     trie.get(name)[0]
    # end = time.perf_counter()
    # print(f"Marisa trie: Get value in {1000 * (end - start):.2f}ms")
    #
    # start = time.perf_counter()
    # for name, _ in pairs:
    #     trie.keys(name)
    # end = time.perf_counter()
    # print(f"Marisa trie: Check for prefix in {1000 * (end - start):.2f}ms")


if __name__ == "__main__":
    benchmark(parse_args())
