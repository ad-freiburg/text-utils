import copy
from typing import Optional, Tuple, Any, Dict, Callable

import einops
import torch
from torch import nn
from torch.nn.utils import rnn

from text_utils.modules import utils
from text_utils.mask import square_subsequent_mask
from text_utils.modules.grouping import Grouping
from text_utils.modules.embedding import Alibi
from text_utils.modules.moe import MoeLayer


class Encoder(nn.Module):
    def additional_losses(self) -> Dict[str, torch.Tensor]:
        return {}

    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError


def encoder_from_config(
    cfg: Dict[str, Any],
    additional_encoder_fn: Optional[Callable[[Dict[str, Any]], Encoder]] = None
) -> Encoder:
    enc_cfg = copy.deepcopy(cfg)
    enc_type = enc_cfg.pop("type")
    if enc_type == "transformer":
        return TransformerEncoder(**enc_cfg)
    elif enc_type == "rnn":
        return RNNEncoder(**enc_cfg)
    elif enc_type == "cnn":
        return CNNEncoder(**enc_cfg)
    elif enc_type == "grouping":
        encoder = encoder_from_config(
            enc_cfg.pop("encoder", {}),
            additional_encoder_fn
        )
        return GroupingEncoder(encoder, **enc_cfg)
    else:
        if additional_encoder_fn is not None:
            return additional_encoder_fn(cfg)
        raise ValueError(f"unknown encoder type {enc_type}")


# modified version of pytorch transformer encoder layer
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        add_pos: bool,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        ffw_type: str = "standard",
        ffw_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffw_type = ffw_type
        if ffw_type == "moe":
            assert ffw_kwargs is not None and "num_experts" in ffw_kwargs
            self.ffw = MoeLayer(
                d_model,
                dim_feedforward,
                d_model,
                ffw_kwargs["num_experts"],
                dropout=dropout,
                activation=activation
            )
        elif ffw_type == "standard":
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.dropout2 = nn.Dropout(dropout)
        else:
            raise ValueError(f"unknown ffw type {ffw_type}")

        self.activation = utils.activation(activation)
        self.add_pos = add_pos

    def forward(
        self,
        src: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            pos: sequence of positional features (optional).
            padding_mask: the mask for the src keys per batch (optional).
            attn_mask: mask added to the attention weights (optional).
        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        x = self.norm1(x + self._sa_block(x, pos, padding_mask, attn_mask))
        x = self.norm2(x + self._ff_block(x))
        return x

    def _add_pos(self, x: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        if self.add_pos and pos is not None:
            return x + pos
        else:
            return x

    # self-attention block
    def _sa_block(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x = self.self_attn(
            self._add_pos(x, pos),
            self._add_pos(x, pos),
            x,
            key_padding_mask=padding_mask,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        if self.ffw_type == "moe":
            x, _ = self.ffw(x)
            return x
        else:
            x = self.linear2(self.dropout(self.activation(self.linear1(x))))
            return self.dropout2(x)


class TransformerEncoder(Encoder):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        heads: int,
        ffw_dim: int,
        dropout: float,
        with_pos: Optional[str] = None,
        activation: str = "gelu",
        share_parameters: bool = False,
        padding_mask: str = "padding_mask",
        causal: bool = False,
        ffw_type: str = "standard",
        ffw_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.num_layers = num_layers
        self.share_parameters = share_parameters
        self.padding_mask = padding_mask

        self.transformer = nn.ModuleList(
            TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                add_pos=with_pos == "attention",
                dim_feedforward=ffw_dim,
                dropout=dropout,
                activation=activation,
                ffw_type=ffw_type,
                ffw_kwargs=ffw_kwargs
            ) for _ in range(1 if self.share_parameters else num_layers)
        )

        self.causal = causal
        self.with_pos = with_pos
        if self.with_pos == "alibi":
            self.alibi = Alibi(heads)

    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        padding_mask = kwargs.get(self.padding_mask)
        assert padding_mask is not None, f"expected '{self.padding_mask}' in kwargs"

        if self.causal:
            attn_mask = square_subsequent_mask(
                x.shape[1],
                float_mask=self.with_pos == "alibi",
                device=x.device
            )
        else:
            attn_mask = None

        if self.with_pos == "alibi":
            attn_mask = self.alibi(x)
        elif self.with_pos == "attention":
            assert pos is not None, "expected positional features for with_pos='attention'"
        elif self.with_pos is None:
            pass
        else:
            raise ValueError(f"unknown with_pos={self.with_pos}, must be either None or one of alibi, attention")

        outputs = []
        enc = x
        for i in range(self.num_layers):
            enc = self.transformer[0 if self.share_parameters else i](
                enc, pos, padding_mask=padding_mask, attn_mask=attn_mask
            )
            outputs.append(enc)
        return torch.stack(outputs) if return_intermediate else outputs[-1], kwargs


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
        _: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        lengths = kwargs.get("lengths")
        assert lengths is not None, "lengths must be given for RNNEncoder"
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
        return padded, kwargs


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
        activation: str = "gelu"
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
        _: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x = einops.rearrange(x, "b s c -> b c s")
        x = self.cnn(x)
        return einops.rearrange(x, "b c s -> b s c"), kwargs


class GroupingEncoder(Encoder):
    def __init__(
        self,
        encoder: Encoder,
        group_first: bool = True,
        group_aggregation: str = "mean",
        group_name: str = "groups",
        group_lengths: str = "group_lengths",
        group_padding_mask: str = "group_padding_mask"
    ):
        super().__init__()
        self.encoder = encoder
        self.grouping = Grouping(
            group_aggregation,
            group_name,
            group_lengths,
            group_padding_mask
        )
        self.group_first = group_first

    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.group_first:
            if pos is not None:
                pos, _ = self.grouping(pos, **kwargs)
            x, kwargs = self.grouping(x, **kwargs)
        x, kwargs = self.encoder(x, pos=pos, **kwargs)
        if not self.group_first:
            x, kwargs = self.grouping(x, **kwargs)
        return x, kwargs


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
