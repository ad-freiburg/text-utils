from typing import Callable, Tuple, List, Optional, Union, Dict, Any

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

    @staticmethod
    def from_beam(other: "Beam", log_p: float, token_id: int) -> "Beam":
        beam = Beam(other.token_ids + [token_id], other.log_probs + [log_p])
        return beam

    @property
    def log_prob(self) -> float:
        return sum(self.log_probs)

    def __lt__(self, other: "Beam") -> bool:
        return len(self) < len(other)

    def __len__(self) -> int:
        return len(self.token_ids)

    def __repr__(self) -> str:
        return f"Beam(token_ids={self.token_ids}, log_probs={self.log_probs})"


# maps from token ids and other kwargs to distribution over next token id
DecodeFn = Callable[..., torch.Tensor]
# selects an index and score from a given token distribution
IdxSelectFn = Callable[
    [
        # distribution over next token id, shape [vocab_size]
        torch.Tensor,
        # idx of input batch element for which the next token is selected
        int
    ],
    Tuple[
        # index of selected token
        int,
        # score (log_prob) of selected token
        float
    ]
]
# selects (multiple) indices and scores from a given token distribution
IndicesSelectFn = Callable[
    [
        # distribution over next token id, shape [vocab_size]
        torch.Tensor,
        # idx of input batch element for which the next token is selected
        int
    ],
    Tuple[
        # indices of selected tokens
        torch.Tensor,
        # scores (log_prob) of selected tokens
        torch.Tensor
    ]
]
# select specific elements for all the kwargs keys given the mask tensor
KwargsSelectFn = Callable[
    [Dict[str, Any], torch.Tensor],
    Dict[str, Any]
]
# calculates a penalty or bonus added to the score of a beam
ReScoreFn = Callable[
    [
        # beam
        Beam,
        # idx of input batch element which is rescored
        int
    ],
    float
]
# determines whether to stop decoding a specific batch element
BeamOrRaw = Union[Beam, Tuple[torch.Tensor, torch.Tensor]]
StopFn = Callable[
    [
        # beam or raw tuple of (token_ids, scores)
        BeamOrRaw,
        # idx of input batch element which is checked for stopping
        int
    ],
    bool
]


def eos_stop_fn(eos_token_id: int) -> StopFn:
    def _stop(inputs: BeamOrRaw, _: int) -> bool:
        if isinstance(inputs, Beam):
            return inputs.token_ids[-1] == eos_token_id
        else:
            token_ids, _ = inputs
            return token_ids[-1] == eos_token_id

    return _stop


def greedy_select_fn() -> IdxSelectFn:
    def _greedy(scores: torch.Tensor, _: int) -> Tuple[int, float]:
        idx = torch.argmax(scores, dim=0)
        return int(idx), float(scores[idx])

    return _greedy


def sample_select_fn(sample_top_k: int) -> IdxSelectFn:
    def _sample(scores: torch.Tensor, _: int) -> Tuple[int, float]:
        k = min(sample_top_k, len(scores))
        sampled_idx = torch.randint(k, (1,)).item()
        top_k = torch.topk(scores, k, dim=0)
        return int(top_k.indices[sampled_idx]), int(top_k.values[sampled_idx])

    return _sample


def beam_select_fn(beam_width: int) -> IndicesSelectFn:
    def _beam(scores: torch.Tensor, _: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert scores.ndim == 2
        k = min(beam_width, scores.shape[1])
        top_k = torch.topk(scores, k, dim=1)
        return top_k.indices, top_k.values

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
        select_fn: IdxSelectFn,
        stop_fn: StopFn,
        device: torch.device,
        kwargs_select_fn: Optional[KwargsSelectFn] = None,
        **kwargs: Any,
) -> List[List[int]]:
    batch_size = len(initial_token_ids)
    assert batch_size > 0

    lengths = []
    padded_initial_token_ids = []
    for token_ids in initial_token_ids:
        num_tokens = len(token_ids)
        assert num_tokens <= max_length, "initial token ids cannot be longer than max length"
        padded_initial_token_ids.append(
            token_ids + [pad_token_id] * (max_length + 1 - num_tokens))
        lengths.append(num_tokens)

    log_prob = torch.zeros(batch_size, max_length + 1,
                           dtype=torch.float, device=device)
    token_ids = torch.as_tensor(
        padded_initial_token_ids, dtype=torch.long, device=device)
    lengths = torch.as_tensor(lengths, dtype=torch.long, device=device)

    smaller_max_length_mask = torch.ones(
        batch_size, dtype=torch.bool, device=device)
    smaller_max_length_mask[lengths >= max_length] = False
    non_stop_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    indices = torch.arange(batch_size, dtype=torch.long, device=device)
    mask = non_stop_mask & smaller_max_length_mask

    # all sequences are at max length or stopped by stop_fn
    while torch.sum(mask) > 0:
        decoder_lengths = _sub_select(lengths, mask)
        max_decoder_length = torch.max(decoder_lengths)

        if kwargs_select_fn is not None:
            decoder_kwargs = kwargs_select_fn(kwargs, mask)
        else:
            decoder_kwargs = kwargs

        decoder_token_ids = _sub_select(
            token_ids, mask
        )[:, :max_decoder_length]
        # always add a padding mask, indicating which tokens are padding
        # and the lengths of the sequence to the additional arguments
        assert "padding_mask" not in decoder_kwargs and "lengths" not in decoder_kwargs, \
            "padding_mask and lengths are added automatically, do not provide them yourself"
        decoder_kwargs["padding_mask"] = decoder_token_ids == pad_token_id
        decoder_kwargs["lengths"] = decoder_lengths

        decoder_output = decode_fn(
            decoder_token_ids,
            **decoder_kwargs
        )

        lengths_of_decoded_indices = lengths[mask]
        log_softmax_scores = torch.log_softmax(
            decoder_output[torch.arange(
                decoder_output.shape[0], device=device), lengths_of_decoded_indices - 1],
            dim=1
        )

        batch_indices = indices[mask].tolist()

        for scores, idx in zip(log_softmax_scores, batch_indices):
            token_id, lp = select_fn(scores, idx)
            token_ids[idx, lengths[idx]] = token_id
            log_prob[idx, lengths[idx]] = lp

        lengths[mask] += 1

        max_length_indices = torch.where(lengths >= max_length)[0]
        smaller_max_length_mask[max_length_indices] = False

        new_stop_indices = []
        for idx in batch_indices:
            length = lengths[idx]
            if stop_fn((token_ids[idx, :length], log_prob[idx, :length]), idx):
                new_stop_indices.append(idx)
        non_stop_mask[torch.tensor(new_stop_indices, dtype=torch.long)] = False

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
        vocab_size: int,
        pad_token_id: int,
        max_length: int,
        stop_fn: StopFn,
        device: torch.device,
        normalize_by_length: bool,
        alpha: float,
        beam_width: int,
        kwargs_select_fn: Optional[KwargsSelectFn] = None,
        **kwargs: Any
) -> List[List[Beam]]:
    batch_size = len(initial_token_ids)

    score_fn = log_likelihood_score(normalize_by_length, alpha)
    beam_width = min(beam_width, vocab_size - 1)  # never produce pad
    select_fn = beam_select_fn(2 * beam_width)

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
        decoder_log_probs = []
        decoder_token_ids = []
        decoder_lengths = []
        for idx in indices_to_decode:
            num_beams.append(len(current_beams[idx]))
            decoder_mask.extend([idx] * num_beams[-1])
            for beam in current_beams[idx]:
                decoder_log_probs.append(beam.log_prob)
                decoder_lengths.append(len(beam))
                decoder_token_ids.append(
                    torch.tensor(beam.token_ids, dtype=torch.long)
                )

        decoder_token_ids = rnn.pad_sequence(
            decoder_token_ids,
            batch_first=True,
            padding_value=pad_token_id
        ).to(non_blocking=True, dtype=torch.long, device=device)
        decoder_log_probs_tensor = torch.tensor(
            decoder_log_probs, dtype=torch.float, device=device
        )
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

        decoder_outputs = decode_fn(
            decoder_token_ids,
            **decoder_kwargs
        )[torch.arange(len(decoder_token_ids), device=device), decoder_lengths_tensor - 1, ...]

        log_softmax_scores = torch.log_softmax(decoder_outputs, dim=1)

        for log_scores, current_scores, idx in zip(
                torch.split(log_softmax_scores, num_beams),
                torch.split(decoder_log_probs_tensor, num_beams),
                indices_to_decode
        ):
            beam_token_ids, beam_log_probs = select_fn(log_scores, idx)
            beam_scores = current_scores.unsqueeze(1) + beam_log_probs
            beam_scores = beam_scores.tolist()
            beam_token_ids = beam_token_ids.tolist()
            beam_log_probs = beam_log_probs.tolist()
            beam_candidates = []
            for beam_idx, (beam, scores, token_ids, log_probs) in enumerate(zip(
                    current_beams[idx], beam_scores, beam_token_ids, beam_log_probs
            )):
                for token_id, score, log_prob in zip(token_ids, scores, log_probs):
                    beam_candidate = Beam.from_beam(beam, log_prob, token_id)
                    beam_candidates.append((score, beam_candidate, beam_idx))

            beam_candidates = sorted(
                beam_candidates, key=lambda e: -e[0]
            )[:2 * beam_width]

            new_current_beams = []
            new_current_scores = []
            for i, (score, beam, beam_idx) in enumerate(beam_candidates):
                # only consider eos beams if they are in top beam_width beams
                if i < beam_width and stop_fn(beam, idx):
                    # we record all stop beams, but only stop when the top beam should stop
                    # (because then we are sure there is no better candidate left to decode)
                    beam_queues[idx].append(beam)
                    if i == 0:
                        stop_mask[idx] = True
                else:
                    new_current_beams.append(beam)
                    new_current_scores.append(score)

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
