import torch


def square_subsequent_mask(
    length: int,
    float_mask: bool = False,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """

    Return a square boolean mask such that position i is only allowed to
    look at positions <= i.

    :param length: number of position
    :param device: device to put the mask on
    :return: mask tensor

    >>> square_subsequent_mask(4, float_mask=True).tolist() # doctest: +NORMALIZE_WHITESPACE
    [[0.0, -inf, -inf, -inf],
     [0.0,  0.0, -inf, -inf],
     [0.0,  0.0,  0.0, -inf],
     [0.0,  0.0,  0.0,  0.0]]

    >>> square_subsequent_mask(4).tolist() # doctest: +NORMALIZE_WHITESPACE
    [[False,  True,  True,  True],
     [False, False,  True,  True],
     [False, False, False,  True],
     [False, False, False, False]]
    """
    mask = torch.full(
        (length, length),
        fill_value=float("-inf") if float_mask else 1,
        device=device,
        dtype=torch.float if float_mask else torch.bool
    )
    return torch.triu(
        mask,
        diagonal=1
    )
