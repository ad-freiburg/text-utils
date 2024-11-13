import glob
import json
import os.path
from typing import Any

import torch


def glob_safe(pattern: str, error_on_empty: bool = True) -> list[str]:
    """
    Safe version of glob.glob in the sense that it errors when
    no files are found with the glob pattern.
    :param pattern: glob pattern
    :param error_on_empty: whether to throw an error when no files are found
    :return: files matched by the pattern
    """
    files = glob.glob(pattern.strip())
    if len(files) == 0 and error_on_empty:
        raise RuntimeError(f"Found no files using glob pattern {pattern}")
    return files


def save_checkpoint(
    checkpoint_path: str,
    model_state_dict: dict[str, Any],
    step: int,
    epoch: int,
    epoch_step: int,
    epoch_items: int,
    total_items: int,
    val_loss: float,
    optimizer_state_dict: dict[str, Any] | None = None,
    lr_scheduler_state_dict: dict[str, Any] | None = None,
    loss_fn_state_dict: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """
    Saves a checkpoint to a file.
    :param checkpoint_path: Filepath to save the checkpoint
    :param model: Pytorch module
    :param step: global step
    :param epoch: global epoch
    :param epoch_step: step within epoch
    :param epoch_items: the number of items (batch elements) seen during this epoch,
        if a fixed batch size is used this would be batch_size * epoch_step
    :param total_items: the number of items (batch elements) seen during training,
        if a fixed batch size is used this would be batch_size * epoch_step
    :param val_loss: Validation loss achieved by this checkpoint
    :param optimizer: Pytorch optimizer
    :param lr_scheduler: Pytorch learning rate scheduler,
    :param loss_fn: Pytorch module computing the loss,
    """
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir != "":
        os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        "model_state_dict": model_state_dict,
        "step": step,
        "epoch": epoch,
        "epoch_step": epoch_step,
        "epoch_items": epoch_items,
        "total_items": total_items,
        "val_loss": val_loss,
        "optimizer_state_dict": optimizer_state_dict,
        "lr_scheduler_state_dict": lr_scheduler_state_dict,
        "loss_fn_state_dict": loss_fn_state_dict,
        **kwargs,
    }
    torch.save(state, f=checkpoint_path)


def load_checkpoint(
    checkpoint_path: str, device: torch.device = torch.device("cpu")
) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=device)


def load_text_file(path: str) -> list[str]:
    text = []
    with open(path, "r", encoding="utf8") as inf:
        for line in inf:
            text.append(line.rstrip("\r\n"))
    return text


def load_jsonl_file(path: str) -> list:
    data = []
    with open(path, "r", encoding="utf8") as inf:
        for line in inf:
            data.append(json.loads(line.rstrip("\r\n")))
    return data
