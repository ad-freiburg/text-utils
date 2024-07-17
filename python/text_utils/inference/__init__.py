from typing import Any, Iterator

import torch
from torch.nn.utils import rnn

from text_utils.inference.utils import (
    DecodeFn,
    SampleFn,
    LogitFn,
    MaskSelectFn,
    MaskUpdateFn,
    Beam,
    BeamWidthFn,
    CandidateFn,
    StopFn,
    ScoreFn,
    default_beam_candidate_fn,
    log_likelihood_score,
    greedy
)


@torch.inference_mode()
def beam_search(
    decode_fn: DecodeFn,
    initial: list[list[int]] | list[Beam],
    pad_token_id: int,
    max_length: int,
    stop_fn: StopFn,
    device: torch.device,
    normalize_by_length: bool,
    alpha: float,
    beam_width: int | BeamWidthFn,
    sample_fn: SampleFn = greedy(),
    candidate_fn: CandidateFn = default_beam_candidate_fn(),
    score_fn: ScoreFn = log_likelihood_score(True, 1.0),
    logit_fns: list[LogitFn] | None = None,
    kwargs_select_fn: MaskSelectFn | None = None,
    kwargs_update_fn: MaskUpdateFn | None = None,
    return_full: bool = False,
    return_incomplete: bool = False,
    yield_intermediate: bool = False,
    **kwargs: Any
) -> Iterator[list[list[Beam]]]:
    batch_size = len(initial)

    score_fn = log_likelihood_score(normalize_by_length, alpha)

    beam_widths: list[int] = []
    current_beams: list[list[Beam]] = []
    beam_queues: list[list[Beam]] = []
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
        beam_queues.append([])

    def get_indices_to_decode() -> list[int]:
        indices_to_decode = []
        for idx in range(batch_size):
            beams = current_beams[idx]
            beam_queue = beam_queues[idx]
            if (
                len(beam_queue) >= beam_widths[idx]
                or len(beams) == 0
                or len(beams[0]) >= max_length
            ):
                continue
            indices_to_decode.append(idx)
        return indices_to_decode

    indices_to_decode = get_indices_to_decode()

    def get_outputs(intermediate: bool) -> list[list[Beam]]:
        out_beams = []
        for idx in range(batch_size):
            beam_queue = beam_queues[idx]
            current = current_beams[idx]
            if intermediate:
                # for intermediate outputs we
                # return the active beams, so swap here
                beam_queue, current = current, beam_queue

            beam_queue = sorted(
                beam_queue,
                key=lambda b: score_fn(b),
                reverse=True
            )
            if len(beam_queue) == 0 and (return_incomplete or intermediate):
                beam_queue = sorted(
                    current,
                    key=lambda b: score_fn(b),
                    reverse=True
                )

            pfx = 0 if return_full else initial_lengths[idx]
            n = beam_widths[idx]
            out_beams.append([
                beam.truncate_prefix(pfx)
                for beam in beam_queue[:n]
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
            decoder_outputs = logit_fn(decoder_token_ids, decoder_outputs, beams)

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

            # reset current beams and fill with best candidates
            current_beams[idx] = []
            for num, (beam, token_id, log_prob) in enumerate(sorted(
                candidates,
                key=lambda item: item[0].log_prob + item[2],
                reverse=True
            )[:n]):
                # update candidates
                candidate = candidate_fn(beam, token_id, log_prob)
                if candidate is None:
                    # skip invalid candidates
                    continue
                elif stop_fn(candidate):
                    # add stop candidates to beam queue
                    beam_queues[idx].append(candidate)
                else:
                    current_beams[idx].append(candidate)

            update_info[idx] = (i, len(current_beams[idx]))

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
            yield get_outputs(intermediate=True)

    yield get_outputs(intermediate=False)
