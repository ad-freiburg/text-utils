import functools
from typing import Optional, List, Tuple, Any, Dict

import einops
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.transformer import _get_activation_fn
from torch.nn.utils import rnn

from text_correction_utils.modules import utils
from text_correction_utils.modules.grouping import Grouping


# modified version of pytorch transformer encoder layer
class _TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(
            self,
            d_model,
            nhead,
            with_pos: bool = False,
            dim_feedforward=2048,
            dropout=0.1,
            activation=F.relu,
            layer_norm_eps=1e-5,
            device=None,
            dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
            **factory_kwargs
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        self.with_pos = with_pos

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(_TransformerEncoderLayer, self).__setstate__(state)

    def forward(
            self,
            src: torch.Tensor,
            pos: Optional[torch.Tensor] = None,
            padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            pos: sequence of positional features (optional).
            padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        x = self.norm1(x + self._sa_block(x, pos, padding_mask))
        x = self.norm2(x + self._ff_block(x))
        return x

    def _with_pos(self, x: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        if self.with_pos and pos is not None:
            return x + pos
        else:
            return x

    # self-attention block
    def _sa_block(
            self,
            x: torch.Tensor,
            pos: Optional[torch.Tensor],
            padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x = self.self_attn(
            self._with_pos(x, pos),
            self._with_pos(x, pos),
            x,
            key_padding_mask=padding_mask,
            need_weights=True
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class Encoder(nn.Module):
    def forward(
            self,
            x: torch.Tensor,
            lengths: List[int],
            pos: Optional[torch.Tensor],
            **kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        raise NotImplementedError


class TransformerEncoder(Encoder):
    def __init__(
            self,
            dim: int,
            num_layers: int,
            heads: int,
            ffw_dim: int,
            dropout: float,
            with_pos: bool,
            activation: str = "gelu_approximate",
            share_parameters: bool = False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.share_paramters = share_parameters

        if activation == "gelu_approximate":
            act_fn = functools.partial(F.gelu, approximate="tanh")
        elif activation == "gelu":
            act_fn = F.gelu
        elif activation == "relu":
            act_fn = F.relu
        else:
            raise ValueError(f"unknown activation function {activation}")

        self.transformer = nn.TransformerEncoder(
            encoder_layer=_TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                with_pos=with_pos,
                dim_feedforward=ffw_dim,
                dropout=dropout,
                activation=act_fn
            ),
            num_layers=1 if self.share_parameters else num_layers
        )

    def forward(
            self,
            x: torch.Tensor,
            lengths: List[int],
            pos: Optional[torch.Tensor],
            **kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        padding_mask = torch.zeros(x.shape[:2], device=x.device, dtype=torch.bool)
        for i, length in enumerate(lengths):
            padding_mask[i, length:] = True
        if self.share_paramters:
            enc = x
            for _ in range(self.num_layers):
                enc = self.transformer(enc, src_key_padding_mask=padding_mask)
            return enc
        else:
            return self.transformer(x, src_key_padding_mask=padding_mask)


class RNNEncoder(Encoder):
    def __init__(
            self,
            dim: int,
            num_layers: int,
            rnn_type: str,
            dropout: float,
            bidirectional: bool = True
    ):
        super().__init__()
        self.bidirectional = bidirectional

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                dim,
                dim,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout * (num_layers > 1)
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                dim,
                dim,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout * (num_layers > 1)
            )
        else:
            raise ValueError(f"unknown rnn type {rnn_type}")

    def forward(
            self,
            x: torch.Tensor,
            lengths: List[int],
            pos: Optional[torch.Tensor],
            **kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        packed = rnn.pack_padded_sequence(
            x,
            torch.as_tensor(lengths, dtype=torch.long),
            batch_first=True,
            enforce_sorted=False
        )
        packed, _ = self.rnn(packed)
        unpacked = rnn.unpack_sequence(packed)
        padded = rnn.pad_sequence(unpacked)
        if self.bidirectional:
            padded = padded.view(*x.shape[:2], 2, -1).mean(2)
        return padded


def _cnn_block(
        dim: int,
        k: int,
        stride: int,
        dropout: float,
        activation: str,
        no_activation: bool
) -> nn.Sequential:
    modules = [
        nn.Conv1d(dim, dim, k, stride, padding=k // 2)
    ]
    if not no_activation:
        modules.append(utils.activation(activation))
    modules.append(nn.Dropout(dropout))
    return nn.Sequential(*modules)


class CNNEncoder(Encoder):
    def __init__(
            self,
            dim: int,
            num_layers: int,
            dropout: float,
            kernel_sizes: Optional[Tuple[int, ...]] = None,
            strides: Optional[Tuple[int, ...]] = None,
            activation: str = "gelu_approximate"
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = (1,) * num_layers
        else:
            assert len(kernel_sizes) == num_layers, f"expected {num_layers} kernel sizes, but got {kernel_sizes}"
        if strides is None:
            strides = (1,) * num_layers
        else:
            assert len(strides) == num_layers, f"expected {num_layers} strides, but got {strides}"

        self.cnn = nn.Sequential(*[
            _cnn_block(dim, kernel_sizes[i], strides[i], dropout, activation, no_activation=i + 1 == num_layers)
            for i in range(num_layers)
        ])

    def forward(
            self,
            x: torch.Tensor,
            lengths: List[int],
            pos: Optional[torch.Tensor],
            **kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        x = einops.rearrange(x, "b s c -> b c s")
        x = self.cnn(x)
        return einops.rearrange(x, "b c s -> b s c")


class GroupingEncoder(Encoder):
    def __init__(
            self,
            encoder: Encoder,
            group_first: bool = False,
            group_aggregation: str = "mean"
    ):
        super().__init__()
        self.encoder = encoder
        self.grouping = Grouping(group_aggregation)
        self.group_first = group_first

    def forward(
            self,
            x: torch.Tensor,
            lengths: List[int],
            pos: Optional[torch.Tensor],
            groups: Optional[List[List[int]]] = None,
            **kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        assert groups is not None, "grouping encoder expects groups keyword argument"
        if self.group_first:
            x, lengths = self.grouping(x, groups)
        x = self.encoder(x, lengths, pos, **kwargs)
        if not self.group_first:
            x, _ = self.grouping(x, groups)
        return x


def encoder(config: Dict[str, Any]) -> Encoder:
    encoder_type = config.pop("type")
    assert encoder_type is not None, "required key type not found in encoder config"
    if encoder_type == "transformer":
        return TransformerEncoder(**config)
    elif encoder_type == "rnn":
        return RNNEncoder(**config)
    elif encoder_type == "cnn":
        return CNNEncoder(**config)
    elif encoder_type == "grouping":
        encoder_config = config.pop("encoder")
        assert encoder_config is not None, "required key encoder not found in grouping encoder config"
        return GroupingEncoder(encoder(encoder_config), **config)
    else:
        raise ValueError(f"unknown encoder type {encoder_type}")
