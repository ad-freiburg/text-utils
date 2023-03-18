import argparse
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", type=str, required=True, nargs="+", help="Files to process")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file")
    return parser.parse_args()


def run(args: argparse.Namespace):
    with open(args.output, "w", encoding="utf8") as of:
        for file in tqdm(args.files, desc="Processing files", leave=False):
            with open(file, "r", encoding="utf8") as inf:
                for line in inf:
                    line = line.strip()
                    if line.startswith("<doc id=") or line.startswith("</doc>") or line == "":
                        continue
                    of.write(line + "\n")


if __name__ == "__main__":
    run(parse_args())
