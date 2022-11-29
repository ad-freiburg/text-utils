from typing import List, Any, Dict

import torch
from torch import nn

from text_correction_utils.modules import utils


class Head(nn.Module):
    def forward(self, x: torch.Tensor, lengths: List[int], **kwargs: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError


class ClassificationHead(Head):
    def __init__(
            self,
            dim: int,
            num_classes: int,
            num_layers: int,
            dropout: float,
            activation: str = "gelu_approximate"
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            if i < num_layers - 1:
                layers.append(nn.Linear(dim, dim))
                layers.append(utils.activation(activation))
                layers.append(nn.Dropout(dropout))
            else:
                # final linear layer
                layers.append(nn.Linear(dim, num_classes))

        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, lengths: List[int], **kwargs: Dict[str, Any]) -> torch.Tensor:
        return self.head(x[:, 0, :])


class SequenceClassificationHead(ClassificationHead):
    def forward(self, x: torch.Tensor, lengths: List[int], **kwargs: Dict[str, Any]) -> torch.Tensor:
        return self.head(x)
