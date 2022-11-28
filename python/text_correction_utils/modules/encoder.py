import functools
from typing import Optional, List, Tuple

import einops
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.transformer import _get_activation_fn
from torch.nn.utils import rnn


# exact copy of pytorch native transformer encoder layer, just with need_weights set to true
class _TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first,
            **factory_kwargs
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(_TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class Encoder(nn.Module):
    def forward(self, x: torch.Tensor, lengths: List[int]) -> torch.Tensor:
        raise NotImplementedError


class TransformerEncoder(Encoder):
    def __init__(
            self,
            dim: int,
            num_layers: int,
            heads: int,
            ffw_dim: int,
            dropout: float,
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
                dim_feedforward=ffw_dim,
                dropout=dropout,
                activation=act_fn,
                batch_first=True
            ),
            num_layers=1 if self.share_parameters else num_layers
        )

    def forward(self, x: torch.Tensor, lengths: List[int]) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, lengths: List[int]) -> torch.Tensor:
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
        if activation == "gelu_approximate":
            act = nn.GELU(approximate="tanh")
        elif activation == "gelu":
            act = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU()
        else:
            raise ValueError(f"unknown activation {activation}")
        modules.append(act)
    modules.append(nn.Dropout1d(dropout))
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

    def forward(self, x: torch.Tensor, lengths: List[int]) -> torch.Tensor:
        x = einops.rearrange(x, "b s c -> b c s")
        x = self.cnn(x)
        return einops.rearrange(x, "b c s -> b s c")
