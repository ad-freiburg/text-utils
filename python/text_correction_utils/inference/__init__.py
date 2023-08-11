from typing import Callable, Tuple, List, Optional, Union, Dict, Any
import copy

import torch

from torch.nn.utils import rnn


class Beam:
    def __init__(
        self,
        token_ids: List[int],
        log_probs: List[float]
    ) -> None:
        self.token_ids: List[int] = token_ids
        self.log_probs: List[float] = log_probs
        self.info: Dict[str, Any] = {}

    @staticmethod
    def from_beam(
        other: "Beam",
        log_p: float,
        token_id: int
    ) -> "Beam":
        beam = Beam(other.token_ids + [token_id], other.log_probs + [log_p])
        beam.info = copy.deepcopy(other.info)
        return beam

    @property
    def log_prob(self) -> float:
        return sum(self.log_probs)

    def __lt__(self, other: "Beam") -> bool:
        return len(self) < len(other)

    def __len__(self) -> int:
        return len(self.token_ids)

    def __repr__(self) -> str:
        return f"""Beam(
    token_ids={self.token_ids},
    log_probs={self.log_probs}
)"""


# maps from token ids and other kwargs to distribution over next token id and other info
DecodeFn = Callable[..., Tuple[torch.Tensor, Dict[str, Any]]]
# selects indices and scores from given token distributions
IdxSelectFn = Callable[
    [
        # distributions over next token id, shape [batch_size, vocab_size]
        torch.Tensor,
        # indices of input batch elements for which the next token is selected
        List[int]
    ],
    Tuple[
        # indices of selected token
        torch.Tensor,
        # scores (log_probs) of selected token
        torch.Tensor
    ]
]
StopFn = Callable[
    [
        # selected token ids, shape [batch_size]
        torch.Tensor,
        # indices of input batch elements which are checked for stopping
        List[int]
    ],
    torch.Tensor
]
# selects (multiple) indices and scores from a given token distribution
BeamSelectFn = Callable[
    [
        # distribution over next token ids, shape [batch_size, beam_size, vocab_size]
        torch.Tensor,
        # input beams
        List[List[Beam]],
        # indices of input batch elements
        List[int]
    ],
    # new beam candidates, should be sorted descending by score
    List[List[Beam]]
]
BeamStopFn = Callable[
    [
        # beam checked for stopping
        Beam,
        # idx of input batch element which is checked for stopping
        int
    ],
    bool
]
# select specific elements for all the kwargs keys given the mask tensor
MaskSelectFn = Callable[
    [Dict[str, Any], torch.Tensor],
    Dict[str, Any]
]
MaskUpdateFn = Callable[
    [Dict[str, Any], Dict[str, Any], torch.Tensor],
    None
]


def eos_stop_fn(eos_token_id: int) -> StopFn:
    def _stop(token_ids: torch.Tensor, _: List[int]) -> torch.Tensor:
        return token_ids == eos_token_id

    return _stop


def greedy_select_fn() -> IdxSelectFn:
    def _greedy(scores: torch.Tensor, _: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert scores.ndim == 2
        indices = torch.argmax(scores, dim=1)
        scores = torch.gather(scores, -1, indices.unsqueeze(-1)).squeeze(-1)
        return indices, scores

    return _greedy


def sample_select_fn(sample_top_k: int) -> IdxSelectFn:
    def _sample(scores: torch.Tensor, _: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert scores.ndim == 2
        k = min(sample_top_k, scores.shape[-1])
        sampled_indices = torch.randint(
            k,
            (len(scores), 1),
            device=scores.device
        )
        top_k = torch.topk(scores, k, dim=-1)
        indices = torch.gather(top_k.indices, -1, sampled_indices).squeeze(-1)
        scores = torch.gather(top_k.values, -1, sampled_indices).squeeze(-1)
        return indices, scores

    return _sample


def beam_select_fn(beam_width: int) -> BeamSelectFn:
    def _beam(
        scores: torch.Tensor,
        batch_beams: List[List[Beam]],
        _: List[int]
    ) -> List[List[Beam]]:
        num_beams = [len(b) for b in batch_beams]
        assert scores.ndim == 2 and scores.shape[0] == sum(num_beams)
        k = min(beam_width, scores.shape[1])
        top_k = torch.topk(scores, k, dim=1)
        top_k_indices = torch.split(top_k.indices, num_beams)
        top_k_values = torch.split(top_k.values, num_beams)
        batch_candidates = []
        for beams, indices, values in zip(batch_beams, top_k_indices, top_k_values):
            candidates = []
            for idx, (token_ids, log_probs) in enumerate(zip(indices.tolist(), values.tolist())):
                for token_id, log_p in zip(token_ids, log_probs):
                    candidates.append((idx, token_id, log_p))
            candidates = sorted(
                candidates,
                key=lambda item: -(beams[item[0]].log_prob + item[2]),
            )[:2 * beam_width]
            batch_candidates.append([
                Beam.from_beam(beams[idx], log_p, token_id)
                for idx, token_id, log_p in candidates
            ])
        return batch_candidates

    return _beam


def _sub_select(
    inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    mask: Union[int, torch.Tensor]
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(inputs, torch.Tensor):
        return inputs[mask]
    elif isinstance(inputs, dict):
        return {k: v[mask] for k, v in inputs.items()}
    else:
        raise ValueError(
            f"expected inputs to be of type tensor or dict of tensors, but got {type(inputs)}"
        )


@torch.inference_mode()
def search(
    decode_fn: DecodeFn,
    initial_token_ids: List[List[int]],
    pad_token_id: int,
    max_length: int,
    stop_fn: StopFn,
    device: torch.device,
    select_fn: Optional[IdxSelectFn] = None,
    kwargs_select_fn: Optional[MaskSelectFn] = None,
    kwargs_update_fn: Optional[MaskUpdateFn] = None,
    **kwargs: Any,
) -> List[List[int]]:
    batch_size = len(initial_token_ids)
    assert batch_size > 0

    if select_fn is None:
        select_fn = greedy_select_fn()

    lengths = []
    padded_initial_token_ids = []
    for token_ids in initial_token_ids:
        num_tokens = len(token_ids)
        assert num_tokens <= max_length, "initial token ids cannot be longer than max length"
        padded_initial_token_ids.append(
            token_ids + [pad_token_id] * (max_length + 1 - num_tokens))
        lengths.append(num_tokens)

    log_prob = torch.zeros(
        batch_size,
        max_length + 1,
        dtype=torch.float,
        device=device
    )
    token_ids = torch.as_tensor(
        padded_initial_token_ids,
        dtype=torch.long,
        device=device
    )
    lengths = torch.as_tensor(lengths, dtype=torch.long, device=device)

    smaller_max_length_mask = torch.ones(
        batch_size,
        dtype=torch.bool,
        device=device
    )
    smaller_max_length_mask[lengths >= max_length] = False
    non_stop_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    indices = torch.arange(batch_size, dtype=torch.long, device=device)
    mask = non_stop_mask & smaller_max_length_mask

    # all sequences are at max length or stopped by stop_fn
    while torch.sum(mask) > 0:
        decoder_lengths = _sub_select(lengths, mask)
        max_decoder_length = torch.max(decoder_lengths)  # type: ignore

        if kwargs_select_fn is not None:
            decoder_kwargs = kwargs_select_fn(kwargs, mask)
        else:
            decoder_kwargs = kwargs

        decoder_token_ids = _sub_select(
            token_ids, mask
        )[:, :max_decoder_length]  # type: ignore
        # always add a padding mask, indicating which tokens are padding
        # and the lengths of the sequence to the additional arguments
        assert "padding_mask" not in decoder_kwargs and "lengths" not in decoder_kwargs, \
            "padding_mask and lengths are added automatically, do not provide them yourself"
        decoder_kwargs["padding_mask"] = decoder_token_ids == pad_token_id
        decoder_kwargs["lengths"] = decoder_lengths

        decoder_output, decoder_info = decode_fn(
            decoder_token_ids,
            **decoder_kwargs
        )
        if kwargs_update_fn is not None:
            kwargs_update_fn(kwargs, decoder_info, mask)

        lengths_of_decoded_indices = lengths[mask]
        log_softmax_scores = torch.log_softmax(
            decoder_output[
                torch.arange(
                    decoder_output.shape[0],
                    device=device
                ),
                lengths_of_decoded_indices - 1
            ],
            dim=1
        )

        batch_indices = indices[mask].tolist()

        sel_ids, sel_lps = select_fn(log_softmax_scores, batch_indices)
        token_ids[mask, lengths_of_decoded_indices] = sel_ids
        log_prob[mask, lengths_of_decoded_indices] = sel_lps

        lengths[mask] += 1

        max_length_indices = torch.where(lengths >= max_length)[0]
        smaller_max_length_mask[max_length_indices] = False

        stop_mask = stop_fn(sel_ids, batch_indices)
        new_stop_indices = indices[mask][stop_mask]
        non_stop_mask[new_stop_indices] = False

        mask = non_stop_mask & smaller_max_length_mask

    token_ids = token_ids.tolist()

    outputs = []
    for i in range(batch_size):
        length = lengths[i]
        outputs.append(token_ids[i][:length])
    return outputs


def log_likelihood_score(normalize_by_length: bool = True, alpha: float = 1.0) -> Callable[[Beam], float]:
    def _score(beam: Beam) -> float:
        if normalize_by_length:
            return beam.log_prob / (len(beam) ** alpha)
        else:
            return beam.log_prob

    return _score


@torch.inference_mode()
def beam_search(
    decode_fn: DecodeFn,
    initial_token_ids: List[List[int]],
    pad_token_id: int,
    max_length: int,
    stop_fn: BeamStopFn,
    device: torch.device,
    normalize_by_length: bool,
    alpha: float,
    beam_width: int,
    select_fn: Optional[BeamSelectFn] = None,
    kwargs_select_fn: Optional[MaskSelectFn] = None,
    kwargs_update_fn: Optional[MaskUpdateFn] = None,
    **kwargs: Any
) -> List[List[Beam]]:
    batch_size = len(initial_token_ids)

    score_fn = log_likelihood_score(normalize_by_length, alpha)
    if select_fn is None:
        select_fn = beam_select_fn(beam_width)

    beam_queues: List[List[Beam]] = [[] for _ in range(batch_size)]

    search_depths: List[int] = []
    current_beams: List[List[Beam]] = []
    for b in range(batch_size):
        # initialize beams
        token_ids = initial_token_ids[b]
        log_prob = [0.0] * len(token_ids)
        beam = Beam(token_ids, log_prob)
        current_beams.append([beam])
        search_depths.append(len(beam))

    stop_mask = [False for _ in range(batch_size)]

    while True:
        indices_to_decode = []
        for idx, (stop, search_depth, beams) in enumerate(zip(
                stop_mask, search_depths, current_beams
        )):
            if not stop and search_depth < max_length and len(beams) > 0:
                indices_to_decode.append(idx)

        if len(indices_to_decode) == 0:
            break

        num_beams = [len(current_beams[idx]) for idx in indices_to_decode]

        num_beams = []
        decoder_mask = []
        decoder_token_ids = []
        decoder_lengths = []
        for idx in indices_to_decode:
            num_beams.append(len(current_beams[idx]))
            decoder_mask.extend([idx] * num_beams[-1])
            for beam in current_beams[idx]:
                decoder_lengths.append(len(beam))
                decoder_token_ids.append(
                    torch.tensor(beam.token_ids, dtype=torch.long)
                )

        decoder_token_ids = rnn.pad_sequence(
            decoder_token_ids,
            batch_first=True,
            padding_value=pad_token_id
        ).to(non_blocking=True, dtype=torch.long, device=device)
        decoder_mask = torch.tensor(
            decoder_mask, dtype=torch.long, device=device
        )
        decoder_lengths_tensor = torch.tensor(
            decoder_lengths, dtype=torch.long, device=device
        )

        decoder_kwargs = kwargs_select_fn(
            kwargs,
            decoder_mask
        ) if kwargs_select_fn is not None else kwargs
        # always add a padding mask, indicating which tokens are padding
        # and the lengths of the sequence to the additional arguments
        assert "padding_mask" not in decoder_kwargs and "lengths" not in decoder_kwargs, \
            "padding_mask and lengths are added automatically, do not provide them yourself"
        decoder_kwargs["padding_mask"] = decoder_token_ids == pad_token_id
        decoder_kwargs["lengths"] = decoder_lengths

        decoder_outputs, decoder_info = decode_fn(
            decoder_token_ids,
            **decoder_kwargs
        )
        decoder_outputs = decoder_outputs[
            torch.arange(len(decoder_mask), device=device),
            decoder_lengths_tensor - 1,
            ...
        ]
        if kwargs_update_fn is not None:
            kwargs_update_fn(
                kwargs,
                decoder_info,
                decoder_mask
            )

        log_softmax_scores = torch.log_softmax(decoder_outputs, dim=1)
        beam_candidates = select_fn(
            log_softmax_scores,
            [current_beams[idx] for idx in indices_to_decode],
            indices_to_decode
        )

        for idx, candidates in zip(indices_to_decode, beam_candidates):
            new_current_beams = []
            for i, candidate in enumerate(candidates):
                # only consider eos beams if they are in top beam_width beams
                if i < beam_width and stop_fn(candidate, idx):
                    # we record all stop beams, but only stop when the top beam should stop
                    # (because then we are sure there is no better candidate left to decode)
                    beam_queues[idx].append(candidate)
                    if i == 0:
                        stop_mask[idx] = True
                else:
                    new_current_beams.append(candidate)

                if len(new_current_beams) >= beam_width:
                    break

            current_beams[idx] = new_current_beams
            search_depths[idx] += 1

    out_beams = []
    for idx, (beam_queue, active_beams) in enumerate(zip(beam_queues, current_beams)):
        beam_queue = sorted(
            beam_queue, key=lambda b: -score_fn(b)
        )[:beam_width]
        if len(beam_queue) < beam_width:
            active_beams = sorted(active_beams, key=lambda e: -score_fn(e))
            beam_queue.extend(active_beams[:beam_width - len(beam_queue)])
        out_beams.append(beam_queue)
    return out_beams
