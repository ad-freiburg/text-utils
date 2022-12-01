import argparse
from typing import Set, Union

from text_correction_utils import metrics
from text_correction_utils.io import load_text_file


def parse_args():
    parser = argparse.ArgumentParser(
        "Evaluation script for most common text correction tasks."
    )
    parser.add_argument(
        "benchmark_type",
        choices=[
            "sed_sequence",
            "sed_words",
            "sec",
            "wc"
        ],
        help="The benchmark type determines the metrics that will be used to evaluate the given files."
    )
    parser.add_argument("in_file", type=str, help="Path to the input file containing misspelled text.")
    parser.add_argument("gt_file", type=str, help="Path to the groundtruth file containing the target outputs.")
    parser.add_argument(
        "pred_file",
        type=str,
        help="Path to the predicted file as outputted by a spell checking model."
    )
    lowercase_group = parser.add_argument_group()
    lowercase_group.add_argument(
        "--lowercase",
        action="store_true",
        help="Whether to lowercase the model predictions before evaluation. Useful "
             "for spelling correction benchmarks that have lowercased inputs and groundtruths, "
             "such as bea322/bea4660."
    )
    lowercase_group.add_argument(
        "--lowercase-file",
        type=str,
        default=None,
        help="Path to a file containing a 0 or 1 for each line in the benchmark, indicating whether the corresponding "
             "predictions should be lowercased or not. Useful e.g. when evaluating on the neuspell "
             "benchmark because half of its sequences are from bea322/bea4660 and they have lowercased inputs "
             "and groundtruths."
    )
    return parser.parse_args()


def evaluate(
    corrupted_file: str,
    groundtruth_file: str,
    predicted_file: str,
    metric_names: Set[str],
    lowercase: Union[bool, str]
) -> None:
    groundtruths = load_text_file(groundtruth_file)
    predictions = load_text_file(predicted_file)
    corrupted = load_text_file(corrupted_file)
    if isinstance(lowercase, str):
        lowercase_lines = [bool(int(lower)) for lower in load_text_file(lowercase)]
    elif isinstance(lowercase, bool):
        lowercase_lines = [bool(int(lowercase)) for _ in range(len(predictions))]
    else:
        raise TypeError(f"expected lowercase to be a string or a bool, but got {type(lowercase)}")

    assert len(predictions) == len(groundtruths) == len(corrupted) == len(lowercase_lines), \
        "expected the same number of lines in the groundtruth, prediction, corrupted, and lowercase files"

    predictions = [
        p.lower() if lower else p
        for p, lower in zip(predictions, lowercase_lines)
    ]

    for name in sorted(metric_names):
        if name == "bin_f1":
            binary_predictions = [bool(int(p)) for prediction in predictions for p in prediction.split()]
            binary_labels = [bool(int(lab)) for label in groundtruths for lab in label.split()]
            f1, prec, rec = metrics.binary_f1(binary_predictions, binary_labels)
            print(f"F1: {100 * f1:.2f} (Precision: {100 * prec:.2f}%, Recall: {100 * rec:.2f}%)")

        elif name == "word_acc":
            word_predictions = [p for prediction in predictions for p in prediction.split()]
            word_groundtruths = [lab for label in groundtruths for lab in label.split()]

            accuracy = metrics.accuracy(word_predictions, word_groundtruths)
            print(f"Word accuracy: {100 * accuracy:.2f}%")

        elif name == "seq_acc":
            accuracy = metrics.accuracy(predictions, groundtruths)
            print(f"Sequence accuracy: {100 * accuracy:.2f}%")

        elif name == "mned":
            mned = metrics.mean_normalized_sequence_edit_distance(predictions, groundtruths)
            print(f"Mean normalized edit distance: {mned:.4f}")

        elif name == "sec_f1":
            for seq_avg in [False, True]:
                (f1, prec, rec) = metrics.spelling_correction_f1(
                    corrupted,
                    predictions,
                    groundtruths,
                    sequence_averaged=seq_avg
                )
                print(f"{'Sequence averaged s' if seq_avg else 'S'}pelling correction F1: {100 * f1:.2f} "
                      f"(Precision: {100 * prec:.2f}%, Recall: {100 * rec:.2f}%)")

        elif name == "wc_f1":
            for seq_avg in [False, True]:
                (f1, prec, rec) = metrics.whitespace_correction_f1(
                    corrupted,
                    predictions,
                    groundtruths,
                    sequence_averaged=seq_avg
                )
                print(f"{'Sequence averaged w' if seq_avg else 'W'}hitespace correction F1: {100 * f1:.2f} "
                      f"(Precision: {100 * prec:.2f}%, Recall: {100 * rec:.2f}%)")

        else:
            raise RuntimeError(f"unknown metric {name}")


def run(args: argparse.Namespace) -> None:
    if args.benchmark_type == "sed_sequence":
        message = "Evaluating spelling error detection (sequence level)"
        metric_names = {"bin_f1", "seq_acc"}
    elif args.benchmark_type == "sed_words":
        message = "Evaluating spelling error detection (word level)"
        metric_names = {"bin_f1", "word_acc"}
    elif args.benchmark_type == "sec":
        message = "Evaluating spelling error correction"
        metric_names = {"sec_f1", "mned"}
    elif args.benchmark_type == "wc":
        message = "Evaluating whitespace correction"
        metric_names = {"wc_f1", "seq_acc"}
    else:
        raise RuntimeError("should not happen")
    print(message)
    print("-" * len(message))
    try:
        evaluate(
            corrupted_file=args.in_file,
            groundtruth_file=args.gt_file,
            predicted_file=args.pred_file,
            metric_names=metric_names,
            lowercase=args.lowercase and args.benchmark_type == "sec"  # lowercase only respected for sec benchmarks
        )
    except Exception as e:
        print(f"An exception was thrown during evaluation: '{e}'.\n"
              f"Please make sure that you passed the input, groundtruth and prediction file in the correct order "
              f"and that they have the correct format for the benchmark type you want to evaluate.")


def main():
    run(parse_args())
