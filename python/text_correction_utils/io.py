import glob
import os.path
from typing import List, Optional, Any, Dict

import torch
from torch import nn, optim
from torch.cuda import amp


def glob_safe(pattern: str, error_on_empty: bool = True) -> List[str]:
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
    model: nn.Module,
    step: int,
    epoch: int,
    epoch_step: int,
    epoch_items: int,
    val_loss: float,
    optimizer: Optional[optim.Optimizer] = None,
    lr_scheduler: Optional[Any] = None,
    loss_fn: Optional[nn.Module] = None,
    grad_scaler: Optional[amp.GradScaler] = None
) -> None:
    """
    Saves a checkpoint to a file.
    :param checkpoint_path: Filepath to save the checkpoint
    :param model: Pytorch module
    :param step: global step
    :param epoch: global epoch
    :param epoch_step: step within epoch
    :param epoch_items: the number of items (batch elements) seen during this epoch, if a fixed batch size is used this would be batch_size * epoch_step
    :param val_loss: Validation loss achieved by this checkpoint
    :param optimizer: Pytorch optimizer
    :param lr_scheduler: Pytorch learning rate scheduler,
    :param loss_fn: Pytorch module computing the loss,
    :param grad_scaler: Pytorch gradient scaler
    """
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir != "":
        os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "epoch": epoch,
        "epoch_step": epoch_step,
        "epoch_items": epoch_items,
        "val_loss": val_loss,
        "optimizer_state_dict": None if optimizer is None
        else optimizer.state_dict(),
        "lr_scheduler_state_dict": None if lr_scheduler is None
        else lr_scheduler.state_dict(),
        "loss_fn_state_dict": None if loss_fn is None
        else loss_fn.state_dict(),
        "grad_scaler_state_dict": None if grad_scaler is None
        else grad_scaler.state_dict()
    }
    torch.save(state, f=checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    return torch.load(checkpoint_path, map_location=device)


def load_text_file(
    path: str
) -> List[str]:
    text = []
    with open(path, "r", encoding="utf8") as inf:
        for line in inf:
            text.append(line.strip())
    return text
