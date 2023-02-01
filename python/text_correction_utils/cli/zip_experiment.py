import argparse
import os
import glob
import zipfile

import torch

from text_correction_utils import io


def zip_experiment(args: argparse.Namespace) -> None:
    if not args.out_file.endswith(".zip"):
        args.out_file += ".zip"

    out_dir = os.path.dirname(args.out_file)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(args.out_file, "w") as zip_file:
        checkpoint_best = io.glob_safe(os.path.join(args.experiment, "checkpoints", "checkpoint_best.pt"))
        assert len(checkpoint_best) == 1
        checkpoint = io.load_checkpoint(checkpoint_best[0])
        only_model_checkpoint = {"model_state_dict": checkpoint["model_state_dict"]}
        only_model_checkpoint_path = os.path.join(args.experiment, "checkpoints", "model_only_checkpoint_best.pt")
        torch.save(only_model_checkpoint, only_model_checkpoint_path)

        experiment_dir = os.path.join(args.experiment, "..")

        yamls = glob.glob(os.path.join(args.experiment, "*.yaml"))
        for yaml in yamls:
            zip_file.write(
                yaml,
                os.path.relpath(yaml, experiment_dir)
            )

        # best checkpoint
        zip_file.write(
            only_model_checkpoint_path,
            os.path.relpath(checkpoint_best[0], experiment_dir)
        )

        # delete only model checkpoint again
        os.remove(only_model_checkpoint_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Script for zipping text correction experiments for easy distribution. "
        "A experiment should be a directory that contains a 'checkpoints' subdirectory with a "
        "'checkpoint_best.pt' file in it and yaml configuration files. The best checkpoint "
        "will be loaded and trimmed to only contain the 'model_state_dict' key to save space."
    )
    parser.add_argument("-e", "--experiment", type=str, required=True)
    parser.add_argument("-o", "--out-file", type=str, required=True)
    return parser.parse_args()


def main():
    zip_experiment(parse_args())
