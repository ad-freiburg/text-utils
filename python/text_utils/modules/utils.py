from torch import nn


def activation(name: str) -> nn.Module:
    if name == "gelu_approximate":
        act = nn.GELU(approximate="tanh")
    elif name == "gelu":
        act = nn.GELU()
    elif name == "relu":
        act = nn.ReLU()
    else:
        raise ValueError(f"unknown activation {name}")
    return act
