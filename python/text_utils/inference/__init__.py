from typing import Callable, Iterator, Any

import torch

from torch.nn.utils import rnn

from text_utils.inference.utils import (
    DecodeFn,
    MaskSelectFn,
    MaskUpdateFn,
    LogitFn,
    SampleFn,
    StopFn,
    Beam,
    BeamSampleFn,
    BeamWidthFn,
    BeamCandidateFn,
    BeamStopFn,
    beam_greedy,
    default_beam_candidate_fn,
    greedy
)


def eos_stop_fn(eos_token_id: int) -> StopFn:
    def _stop(token_ids: torch.Tensor, _: int) -> bool:
        return token_ids[-1].item() == eos_token_id

    return _stop


def _sub_select(
    inputs: torch.Tensor | dict[str, torch.Tensor],
    mask: int | torch.Tensor,
) -> torch.Tensor | dict[str, torch.Tensor]:
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
    initial_token_ids: list[list[int]],
    pad_token_id: int,
    max_length: int,
    stop_fn: StopFn,
    device: torch.device,
    sample_fn: SampleFn = greedy(),
    logit_fns: list[LogitFn] | None = None,
    kwargs_select_fn: MaskSelectFn | None = None,
    kwargs_update_fn: MaskUpdateFn | None = None,
    return_full: bool = False,
    yield_intermediate: bool = False,
    **kwargs: Any,
) -> Iterator[list[list[int]]]:
    batch_size = len(initial_token_ids)
    assert batch_size > 0, "empty inputs"

    lengths = []
    padded_initial_token_ids = []
    for token_ids in initial_token_ids:
        num_tokens = len(token_ids)
        assert num_tokens <= max_length, "initial token ids cannot be longer than max length"
        padded_initial_token_ids.append(
            token_ids + [pad_token_id] * (max_length + 1 - num_tokens)
        )
        lengths.append(num_tokens)
    initial_lengths = list(lengths)

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
    lengths = torch.as_tensor(lengths, dtype=torch.long)

    smaller_max_length_mask = torch.ones(
        batch_size,
        dtype=torch.bool,
    )
    smaller_max_length_mask[lengths >= max_length] = False
    non_stop_mask = torch.ones(batch_size, dtype=torch.bool)
    mask = non_stop_mask & smaller_max_length_mask
    indices = torch.arange(batch_size, dtype=torch.long)

    def get_outputs() -> list[list[int]]:
        outputs = []
        for i in range(batch_size):
            length = lengths[i]
            start = 0 if return_full else initial_lengths[i]
            outputs.append(token_ids[i][start:length].tolist())
        return outputs

    # all sequences are at max length or stopped by stop_fn
    while torch.sum(mask) > 0:
        decoder_lengths = _sub_select(lengths, mask)
        assert isinstance(decoder_lengths, torch.Tensor)
        max_decoder_length = torch.max(decoder_lengths)  # type: ignore
        indices_mask = indices[mask]

        decoder_kwargs = kwargs_select_fn(
            kwargs,
            indices_mask
        ) if kwargs_select_fn is not None else {}

        decoder_token_ids = _sub_select(
            token_ids, mask
        )[:, :max_decoder_length]  # type: ignore
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
        b, s, _ = decoder_outputs.shape
        if s == 1:
            decoder_outputs = decoder_outputs[:, 0]
        else:
            decoder_outputs = decoder_outputs[
                torch.arange(b, device=decoder_outputs.device),
                decoder_lengths.to(decoder_outputs.device) - 1
            ]

        # process logits and sample
        batch_indices = indices_mask.tolist()
        for logit_fn in logit_fns or []:
            decoder_outputs = logit_fn(decoder_outputs, batch_indices)
        sel_ids = sample_fn(decoder_outputs, batch_indices)

        # get log prob of selected ids
        sel_lps = torch.gather(
            torch.log_softmax(decoder_outputs, dim=-1),
            -1,
            sel_ids.unsqueeze(-1)
        ).squeeze(-1)

        sel_ids = sel_ids.to(token_ids.device)
        sel_lps = sel_lps.to(log_prob.device)
        token_ids[mask, decoder_lengths] = sel_ids
        log_prob[mask, decoder_lengths] = sel_lps

        lengths[mask] += 1

        max_length_indices = torch.where(lengths >= max_length)[0]
        smaller_max_length_mask[max_length_indices] = False

        for idx in batch_indices:
            if stop_fn(token_ids[idx, :lengths[idx]], idx):
                non_stop_mask[idx] = False

        mask = non_stop_mask & smaller_max_length_mask

        if kwargs_update_fn is not None:
            update_mask = torch.arange(b)[mask[indices_mask]]
            kwargs_update_fn(kwargs, decoder_info, update_mask)

        if yield_intermediate:
            yield get_outputs()

    yield get_outputs()


def log_likelihood_score(
    normalize_by_length: bool = True,
    alpha: float = 1.0
) -> Callable[[Beam], float]:
    def _score(beam: Beam) -> float:
        if normalize_by_length:
            return beam.log_prob / (len(beam) ** alpha)
        else:
            return beam.log_prob

    return _score


@torch.inference_mode()
def beam_search(
    decode_fn: DecodeFn,
    initial: list[list[int]] | list[Beam],
    pad_token_id: int,
    max_length: int,
    stop_fn: BeamStopFn,
    device: torch.device,
    normalize_by_length: bool,
    alpha: float,
    beam_width: int | BeamWidthFn,
    sample_fn: BeamSampleFn = beam_greedy(),
    candidate_fn: BeamCandidateFn = default_beam_candidate_fn(),
    logit_fns: list[LogitFn] | None = None,
    kwargs_select_fn: MaskSelectFn | None = None,
    kwargs_update_fn: MaskUpdateFn | None = None,
    return_full: bool = False,
    yield_intermediate: bool = False,
    **kwargs: Any
) -> Iterator[list[list[Beam]]]:
    batch_size = len(initial)

    score_fn = log_likelihood_score(normalize_by_length, alpha)

    beam_queues: list[list[Beam]] = [[] for _ in range(batch_size)]

    beam_widths: list[int] = []
    current_beams: list[list[Beam]] = []
    initial_lengths = []
    for init in initial:
        if isinstance(init, Beam):
            beam = init
        else:
            beam = Beam(init, [0.0] * len(init))

        initial_lengths.append(len(beam))
        assert initial_lengths[-1] <= max_length, \
            "initial beam cannot be longer than max length"

        if isinstance(beam_width, int):
            beam_widths.append(beam_width)
        else:
            beam_widths.append(beam_width(beam))

        current_beams.append([beam])

    stop_mask = [False for _ in range(batch_size)]

    def get_indices_to_decode() -> list[int]:
        indices_to_decode = []
        for idx, (stop, beams) in enumerate(zip(
            stop_mask,
            current_beams
        )):
            if stop or len(beams) == 0 or len(beams[0]) >= max_length:
                continue
            indices_to_decode.append(idx)
        return indices_to_decode

    indices_to_decode = get_indices_to_decode()

    def get_outputs() -> list[list[Beam]]:
        out_beams = []
        for idx, (beam_queue, active_beams, n) in enumerate(zip(
            beam_queues,
            current_beams,
            beam_widths
        )):
            beam_queue = sorted(
                beam_queue,
                key=lambda b: score_fn(b),
                reverse=True
            )[:n]

            if len(beam_queue) < n:
                active_beams = sorted(
                    active_beams,
                    key=lambda b: score_fn(b),
                    reverse=True
                )
                beam_queue.extend(active_beams[:n - len(beam_queue)])

            pfx = 0 if return_full else initial_lengths[idx]
            out_beams.append([
                beam.truncate_prefix(pfx)
                for beam in beam_queue
            ])

        return out_beams

    while len(indices_to_decode) > 0:
        num_beams = []
        beams = []
        decoder_mask = []
        decoder_token_ids = []
        decoder_lengths = []
        for idx in indices_to_decode:
            num_beams.append(len(current_beams[idx]))
            decoder_mask.extend([idx] * num_beams[-1])
            for beam in current_beams[idx]:
                beams.append(beam)
                decoder_lengths.append(len(beam))
                decoder_token_ids.append(
                    torch.tensor(beam.token_ids, dtype=torch.long)
                )

        decoder_token_ids = rnn.pad_sequence(
            decoder_token_ids,
            batch_first=True,
            padding_value=pad_token_id
        ).to(non_blocking=True, dtype=torch.long, device=device)
        decoder_mask = torch.tensor(decoder_mask, dtype=torch.long)
        decoder_lengths_tensor = torch.tensor(
            decoder_lengths,
            dtype=torch.long
        )

        decoder_kwargs = kwargs_select_fn(
            kwargs,
            decoder_mask
        ) if kwargs_select_fn is not None else {}
        # always add a padding mask, indicating which tokens are padding
        # and the lengths of the sequence to the additional arguments
        assert "padding_mask" not in decoder_kwargs and "lengths" not in decoder_kwargs, \
            "padding_mask and lengths are added automatically, do not provide them yourself"
        decoder_kwargs["padding_mask"] = decoder_token_ids == pad_token_id
        decoder_kwargs["lengths"] = decoder_lengths_tensor

        decoder_outputs, decoder_info = decode_fn(
            decoder_token_ids,
            **decoder_kwargs
        )
        b, s, _ = decoder_outputs.shape
        if s == 1:
            decoder_outputs = decoder_outputs[:, 0]
        else:
            decoder_outputs = decoder_outputs[
                torch.arange(b),
                decoder_lengths_tensor - 1
            ]

        # apply logit functions
        for logit_fn in logit_fns or []:
            decoder_outputs = logit_fn(decoder_outputs, beams)

        log_probs = torch.log_softmax(decoder_outputs, dim=-1)

        update_info = {}
        for i, (idx, log_probs) in enumerate(zip(
            indices_to_decode,
            torch.split(log_probs, num_beams)
        )):
            n = beam_widths[idx]
            candidates: list[tuple[Beam, int, float]] = []
            for beam_idx, beam in enumerate(current_beams[idx]):
                for token_id in sample_fn(log_probs[beam_idx], n).tolist():
                    candidates.append((
                        beam,
                        token_id,
                        log_probs[beam_idx, token_id].item()
                    ))

            new_beams = []
            for num, (beam, token_id, log_prob) in enumerate(sorted(
                candidates,
                key=lambda item: item[0].log_prob + item[2],
                reverse=True
            )):
                # update candidates
                candidate = candidate_fn(beam, token_id, log_prob)
                if candidate is None:
                    # skip invalid candidates
                    continue
                # only consider eos beams if they are in top beam_width beams
                stop = stop_fn(candidate)
                if num < n and stop:
                    # we record all stop beams, but only stop when the top beam should stop
                    # (because then we are sure there is no better candidate left to decode)
                    beam_queues[idx].append(candidate)
                    stop_mask[idx] |= num == 0
                elif not stop:
                    new_beams.append(candidate)

                if len(new_beams) >= n:
                    break

            if len(new_beams) == 0:
                stop_mask[idx] = True
            else:
                current_beams[idx] = new_beams
                update_info[idx] = (i, len(new_beams))

        indices_to_decode = get_indices_to_decode()

        if kwargs_update_fn is not None:
            update_mask = []
            for idx in indices_to_decode:
                if idx not in update_info:
                    continue
                i, num = update_info[idx]
                update_mask.extend([i] * num)
            kwargs_update_fn(
                kwargs,
                decoder_info,
                torch.tensor(update_mask, dtype=torch.long)
            )

        if yield_intermediate:
            yield get_outputs()

    yield get_outputs()
