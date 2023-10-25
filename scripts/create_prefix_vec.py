from tempfile import NamedTemporaryFile
import time
import argparse
import os


from text_utils import prefix, tokenization, configuration


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
        tokenizer = tokenization.Tokenizer.from_config(tokenizer_cfg)
        num_pfx = tokenizer.num_prefix_tokens()
        num_sfx = tokenizer.num_suffix_tokens()
        with open(args.file, "r", encoding="utf8") as inf, \
                NamedTemporaryFile("w", encoding="utf8") as of:
            print(f"Preparing file with tokenizer into {of.name}")
            for line in inf:
                line = line.strip().split("\t")
                assert len(line) >= 3
                for i in range(2, len(line)):
                    token_ids = tokenizer.tokenize(line[i]).token_ids
                    line[i] = tokenizer.de_tokenize(
                        token_ids[num_pfx:len(token_ids) - num_sfx],
                        False
                    ).strip()
                of.write("\t".join(line) + "\n")
            trie = prefix.Vec.from_file(of.name)
    else:
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
