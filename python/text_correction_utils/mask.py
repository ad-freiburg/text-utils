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
    [[0.0, -inf, -inf, -inf],
     [0.0,  0.0, -inf, -inf],
     [0.0,  0.0,  0.0, -inf],
     [0.0,  0.0,  0.0,  0.0]]

    """
    return torch.triu(
        torch.full((length, length), fill_value=float("-inf"), device=device, dtype=torch.float),
        diagonal=1
    )
