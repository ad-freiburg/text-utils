from copy import deepcopy
from typing import Tuple, Dict

import torch
from torch import nn, func


from text_utils.modules import utils
from text_utils.api.utils import to


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden_layers: int = 0,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(utils.activation(activation))
        self.layers.append(nn.Dropout(dropout))

        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(utils.activation(activation))
            self.layers.append(nn.Dropout(dropout))

        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers.append(nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MoeLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        num_hidden_layers: int = 0,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        experts = [
            MLP(input_dim, hidden_dim, output_dim, num_hidden_layers, dropout, activation)
            for _ in range(num_experts)
        ]
        (
            self.expert_params,
            self.expert_buffers
        ) = func.stack_module_state(experts)

        meta_model = deepcopy(experts[0])
        meta_model.to("meta")

        def run_moe(
            params: Dict[str, torch.Tensor],
            buffers: Dict[str, torch.Tensor],
            x: torch.Tensor
        ) -> torch.Tensor:
            return func.functional_call(
                meta_model,
                (params, buffers),
                (x,),
                strict=True
            )

        self.expert_model = run_moe

        self.gate = MLP(input_dim, hidden_dim, num_experts, 1, dropout, activation)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.ndim == 3
        # x: [B, S, H]
        # weights: [E, B, S]
        weights = torch.softmax(self.gate(x), dim=-1).permute(2, 0, 1)
        # move params and buffers to device
        self.expert_params = to(self.expert_params, x.device)
        self.expert_buffers = to(self.expert_buffers, x.device)
        experts_output = func.vmap(
            self.expert_model,
            (0, 0, None),
            randomness="different"
        )(
            self.expert_params,
            self.expert_buffers,
            x
        )
        # combine weights and expert_output: [E, B, S, 1] * [E, B, S, H], then
        # sum over E
        output = (weights[..., None] * experts_output).sum(dim=0)
        return output, weights.permute(1, 2, 0)
