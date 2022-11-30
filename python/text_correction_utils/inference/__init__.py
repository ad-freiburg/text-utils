import heapq
from typing import Callable, Tuple, List, Optional, Union, Dict

import einops
import torch

from text_correction_utils._internal import inference as windows
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


# maps from token ids and optional encoder outputs and lengths to distribution over next token id
DecodeFn = Callable[[torch.Tensor, Optional[torch.Tensor], Optional[List[int]]], torch.Tensor]
ReScoreFn = Callable[
    [
        # beam or list of token ids
        Beam,
        # optional input string
        Optional[str]
    ],
    float
]
StopFn = Callable[[Union[List[int], torch.Tensor], Optional[str]], bool]
IdxSelectFn = Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


def eos_stop_fn(eos_token_id: int) -> StopFn:
    def _stop(token_ids: Union[List[int], torch.Tensor], _: Optional[str] = None) -> bool:
        return token_ids[-1] == eos_token_id

    return _stop


def greedy_select_fn() -> IdxSelectFn:
    def _greedy(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = torch.argmax(t, dim=1)
        return indices, t[torch.arange(len(t)), indices]

    return _greedy


def sample_select_fn(sample_top_k: int) -> IdxSelectFn:
    def _sample(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k = min(sample_top_k, t.shape[1])
        sampled_indices = torch.randint(k, size=(len(t),))
        top_k = torch.topk(t, k, dim=1)
        batch_indices = torch.arange(len(t))
        return top_k.indices[batch_indices, sampled_indices], top_k.values[batch_indices, sampled_indices]

    return _sample


def beam_select_fn(beam_width: int) -> IdxSelectFn:
    def _beam(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k = min(beam_width, t.shape[1])
        top_k = torch.topk(t, k=k, dim=1)
        return top_k.indices, top_k.values

    return _beam


def log_likelihood_score(normalize_by_length: bool = True, alpha: float = 1.0) -> Callable[[Beam], float]:
    def score(beam: Beam) -> float:
        if normalize_by_length:
            return beam.log_prob / (len(beam) ** alpha)
        else:
            return beam.log_prob

    return score


def _sub_select(
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        mask: Union[int, torch.Tensor]
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(inputs, torch.Tensor):
        return inputs[mask]
    elif isinstance(inputs, dict):
        return {k: v[mask] for k, v in inputs.items()}
    else:
        raise ValueError(f"expected inputs to be of type tensor or dict of tensors, but got {type(inputs)}")


@torch.inference_mode()
def token_inference(
        decode_fn: DecodeFn,
        initial_token_ids: List[List[int]],
        initial_strings: Optional[List[str]],
        encoder_outputs: Optional[torch.Tensor],
        encoder_lengths: Optional[List[int]],
        pad_token_id: int,
        max_length: int,
        select_fn: IdxSelectFn,
        stop_fn: StopFn,
        device: torch.device
) -> List[List[int]]:
    batch_size = len(initial_token_ids)
    assert batch_size > 0

    lengths = []
    padded_initial_token_ids = []
    for token_ids in initial_token_ids:
        num_tokens = len(token_ids)
        assert num_tokens <= max_length, "initial token ids cannot be longer than max length"
        padded_initial_token_ids.append(token_ids + [pad_token_id] * (max_length + 1 - num_tokens))
        lengths.append(num_tokens)

    log_prob = torch.zeros(batch_size, max_length + 1, dtype=torch.float, device=device)
    token_ids = torch.as_tensor(padded_initial_token_ids, dtype=torch.long, device=device)
    lengths = torch.as_tensor(lengths, dtype=torch.long, device=device)

    smaller_max_length_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    smaller_max_length_mask[lengths >= max_length] = False
    non_stop_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    indices_to_decode = non_stop_mask & smaller_max_length_mask

    while True:
        # all sequences are at max length or stopped by stop_fn
        if torch.sum(indices_to_decode) == 0:
            break

        decoder_lengths = _sub_select(lengths, indices_to_decode)
        max_decoder_length = torch.max(decoder_lengths)

        decoder_output = decode_fn(
            _sub_select(token_ids, indices_to_decode)[:, :max_decoder_length],
            _sub_select(encoder_outputs, indices_to_decode)
            if encoder_outputs is not None else None,
            [encoder_lengths[i] for i, needs_decode in enumerate(indices_to_decode) if needs_decode]
            if encoder_lengths is not None else None
        )

        lengths_of_decoded_indices = lengths[indices_to_decode]
        log_softmax_scores = torch.log_softmax(
            torch.stack([decoder_output[i, lengths_of_decoded_indices[i] - 1] for i in range(len(decoder_output))]),
            dim=1
        )
        inferred_token_ids, inferred_log_prob = select_fn(log_softmax_scores)

        token_ids[indices_to_decode, lengths[indices_to_decode]] = inferred_token_ids
        log_prob[indices_to_decode, lengths[indices_to_decode]] = inferred_log_prob

        lengths[indices_to_decode] += 1

        max_length_indices = torch.where(lengths >= max_length)[0]
        smaller_max_length_mask[max_length_indices] = False

        batch_indices = torch.where(indices_to_decode)[0].tolist()
        new_stop_indices = []
        for idx, length in zip(batch_indices, lengths[indices_to_decode]):
            if stop_fn(token_ids[idx, :length], initial_strings[idx] if initial_strings is not None else None):
                new_stop_indices.append(idx)
        non_stop_mask[torch.tensor(new_stop_indices, dtype=torch.long)] = False

        indices_to_decode = non_stop_mask & smaller_max_length_mask

    token_ids = token_ids.tolist()

    outputs = []
    for i in range(batch_size):
        length = lengths[i]
        outputs.append(token_ids[i][:length])
    return outputs


@torch.inference_mode()
def best_first_inference(
        decode_fn: DecodeFn,
        initial_token_ids: List[List[int]],
        initial_strings: Optional[List[str]],
        encoder_outputs: Optional[torch.Tensor],
        encoder_lengths: Optional[List[int]],
        input_strings: Optional[List[str]],
        max_length: int,
        re_score_fn: Optional[ReScoreFn],
        stop_fn: StopFn,
        device: torch.device,
) -> List[List[Beam]]:
    all_beams: List[List[Beam]] = []

    batch_size = len(next(iter(encoder_outputs.values())))

    for b in range(batch_size):
        encoder_output = _sub_select(encoder_outputs, b) if encoder_outputs is not None else None
        encoder_length = encoder_lengths[b] if encoder_lengths is not None else None

        # initialize beams
        beam_queue = []
        finished_beams = []

        token_ids = initial_token_ids[b]
        log_prob = [0.0] * len(token_ids)

        beam = Beam(token_ids, log_prob)
        heapq.heappush(beam_queue, (0, beam))

        while len(beam_queue):
            beam: Beam = heapq.heappop(beam_queue)[1]

            stop = stop_fn(beam.token_ids, initial_strings[b] if initial_strings is not None else None)
            if stop or len(beam) >= max_length:
                finished_beams.append(beam)
                break

            continuations = einops.repeat(
                torch.tensor(beam.token_ids, dtype=torch.long, device=device),
                "l -> b l",
                b=1
            )

            beam_encoder_outputs = einops.repeat(
                encoder_output,
                "l h -> b l h",
                b=len(continuations)
            ) if encoder_output is not None else None
            beam_encoder_lengths = [encoder_length] * len(continuations) if encoder_length is not None else None

            # decoder_output: shape [B, VOC]
            decoder_output = decode_fn(
                continuations,
                beam_encoder_outputs,
                beam_encoder_lengths,
            )[0, -1, ...]

            log_softmax_scores = torch.log_softmax(decoder_output, dim=0)
            scores = log_softmax_scores + beam.log_prob

            for token_id, (score, log_p) in enumerate(zip(scores, log_softmax_scores.tolist())):
                new_beam = Beam.from_beam(beam, log_p, token_id)
                if re_score_fn is not None:
                    penalty = re_score_fn(new_beam, input_strings[b] if input_strings is not None else None)
                    score += penalty
                heapq.heappush(beam_queue, (-score, new_beam))
        all_beams.append(finished_beams)
    return all_beams


@torch.inference_mode()
def beam_inference(
        decode_fn: DecodeFn,
        initial_token_ids: List[List[int]],
        initial_strings: Optional[List[str]],
        encoder_outputs: Optional[torch.Tensor],
        encoder_lengths: Optional[List[int]],
        input_strings: Optional[List[str]],
        vocab_size: int,
        pad_token_id: int,
        max_length: int,
        stop_fn: StopFn,
        device: torch.device,
        normalize_by_length: bool,
        alpha: float,
        beam_width: int
) -> List[List[Beam]]:
    batch_size = len(next(iter(encoder_outputs.values())))

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
        decoder_mask = []
        for stop, beam_queue, search_depth, beams in zip(
                stop_mask, beam_queues, search_depths, current_beams
        ):
            decoder_mask.append(
                not stop
                and search_depth < max_length
                and len(beams)
            )

        if not any(decoder_mask):
            break

        decoder_mask_tensor = torch.tensor(decoder_mask, dtype=torch.bool, device=device)
        indices_to_decode = [i for i, needs_decode in enumerate(decoder_mask) if needs_decode]
        num_beams = [len(beams) for i, beams in enumerate(current_beams) if decoder_mask[i]]

        decoder_log_probs = []
        decoder_inputs = []
        decoder_lengths = []
        for i, beams in enumerate(current_beams):
            if not decoder_mask[i]:
                continue
            for beam in beams:
                decoder_log_probs.append(beam.log_prob)
                decoder_lengths.append(len(beam))
                decoder_inputs.append(torch.tensor(beam.token_ids, dtype=torch.long))

        decoder_inputs = rnn.pad_sequence(
            decoder_inputs,
            batch_first=True,
            padding_value=pad_token_id
        ).to(torch.long, device)
        decoder_log_probs_tensor = torch.tensor(decoder_log_probs, device=device)
        decoder_lengths_tensor = torch.tensor(decoder_lengths, device=device)

        beam_encoder_lengths = [
            encoder_lengths[i] for i in indices_to_decode
        ] if encoder_lengths is not None else None
        beam_encoder_outputs = _sub_select(
            encoder_outputs,
            decoder_mask_tensor
        ) if encoder_outputs is not None else None

        decoder_outputs = decode_fn(
            decoder_inputs,
            beam_encoder_outputs,
            beam_encoder_lengths
        )[torch.arange(len(decoder_inputs), device=device), decoder_lengths_tensor - 1, ...]

        log_softmax_scores = torch.log_softmax(decoder_outputs, dim=1)
        beam_token_ids, beam_log_probs = select_fn(log_softmax_scores)
        beam_scores = (decoder_log_probs_tensor.unsqueeze(1) + beam_log_probs).tolist()
        beam_token_ids, beam_log_probs = beam_token_ids.tolist(), beam_log_probs.tolist()

        for beam_scores_b, token_ids_b, log_probs_b, idx in zip(
                torch.split(beam_scores, num_beams),
                torch.split(beam_token_ids, num_beams),
                torch.split(beam_log_probs, num_beams),
                indices_to_decode
        ):
            beam_candidates = []
            for beam_idx, (beam, scores, token_ids, log_probs) in enumerate(zip(
                    current_beams[idx], beam_scores_b, token_ids_b, log_probs_b
            )):
                for token_id, score, log_prob in zip(token_ids, scores, log_probs):
                    beam_candidate = Beam.from_beam(beam, log_prob, token_id)
                    beam_candidates.append((score, beam_candidate, beam_idx))

            beam_candidates = sorted(beam_candidates, key=lambda e: -e[0])[:2 * beam_width]

            new_current_beams = []
            new_current_scores = []
            for i, (score, beam, beam_idx) in enumerate(beam_candidates):
                # only consider eos beams if they are in top beam_width beams
                if i < beam_width and stop_fn(
                        beam.token_ids,
                        initial_strings[idx] if initial_strings is not None else None
                ):
                    # we record all eos beams, but only stop when the top beam is eos (because then we are sure there
                    # is no better candidate left to decode)
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
    for beam_queue, active_beams in zip(beam_queues, current_beams):
        beam_queue = sorted(beam_queue, key=lambda e: -score_fn(e))[:beam_width]
        if len(beam_queue) < beam_width:
            active_beams = sorted(active_beams, key=lambda e: -score_fn(e))
            beam_queue.extend(active_beams[:beam_width - len(beam_queue)])
        out_beams.append(beam_queue)
    return out_beams
