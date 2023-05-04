import argparse
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", type=str, required=True, nargs="+", help="Files to process")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file")
    parser.add_argument("--min-chars", type=int, default=10)
    parser.add_argument("--min-words", type=int, default=2)
    return parser.parse_args()


def run(args: argparse.Namespace):
    total = 0
    valid = 0
    with open(args.output, "w", encoding="utf8") as of:
        for file in tqdm(args.files, desc="Processing files", leave=False):
            with open(file, "r", encoding="utf8") as inf:
                for line in inf:
                    line = line.strip()
                    if line.startswith("<doc id=") or line.startswith("</doc>") or line == "":
                        continue
                    total += 1
                    has_chars = len(line) >= args.min_chars
                    has_words = len(line.split()) >= args.min_words
                    if not has_chars and not has_words:
                        continue
                    of.write(line + "\n")
                    valid += 1
    # print percentage of valid lines
    print(f"valid lines: {valid}/{total} ({100 * valid / total:.2f}%)")


if __name__ == "__main__":
    run(parse_args())
