from typing import List

import torch


def square_subsequent_mask(
    length: int,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """

    Return a square boolean mask such that position i is only allowed to
    look at positions <= i.

    :param length: number of position
    :param device: device to put the mask on
    :return: mask tensor

    >>> square_subsequent_mask(4).tolist() # doctest: +NORMALIZE_WHITESPACE
    [[False, True, True, True],
    [False, False, True, True],
    [False, False, False, True],
    [False, False, False, False]]

    """
    return torch.triu(
        torch.ones(length, length, device=device, dtype=torch.bool),
        diagonal=1
    )


def padding_mask(
    lengths: List[int],
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """

    Returns a rectangular boolean mask of size [len(lengths) x max(lengths)]
    such that for each row i only the first lengths[i] entries are allowed to be
    looked at.

    :param lengths: list of lengths
    :param device: device to put the mask on
    :return: mask tensor

    >>> padding_mask([2, 4, 1, 3]).tolist() # doctest: +NORMALIZE_WHITESPACE
    [[False, False, True, True],
    [False, False, False, False],
    [False, True, True, True],
    [False, False, False, True]]

    """
    mask = torch.zeros(len(lengths), max(lengths), dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, length:] = True
    return mask.to(non_blocking=True, device=device)
