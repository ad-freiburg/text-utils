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
    UpdateFn,
    StopFn,
    ScoreFn,
    identity_update_fn,
    log_likelihood_score,
    greedy
)


@torch.inference_mode()
def beam_search(
    decode_fn: DecodeFn,
    initial: list[list[int]] | list[Beam] | list[list[Beam]],
    pad_token_id: int,
    max_length: int,
    stop_fn: StopFn,
    device: torch.device,
    beam_width: int,
    sample_fn: SampleFn = greedy(),
    update_fn: UpdateFn = identity_update_fn(),
    score_fn: ScoreFn = log_likelihood_score(),
    logit_fns: list[LogitFn] | None = None,
    kwargs_select_fn: MaskSelectFn | None = None,
    kwargs_update_fn: MaskUpdateFn | None = None,
    max_outputs: int | list[int] | None = None,
    max_new_tokens: int | None = None,
    return_incomplete: bool = False,
    yield_intermediate: bool = False,
    **kwargs: Any
) -> Iterator[list[list[Beam]]]:
    assert max_new_tokens is None or max_new_tokens > 0, \
        "max_new_tokens must be None or positive"
    batch_size = len(initial)

    decoder_info: Any | None = None
    update_info: list[int] = []
    current_beams: list[list[Beam]] = []
    beam_queues: list[list[Beam]] = []
    if max_outputs is None:
        max_outputs = [beam_width] * batch_size
    elif isinstance(max_outputs, int):
        max_outputs = [max_outputs] * batch_size
    else:
        assert len(max_outputs) == batch_size, \
            "max_outputs must be None, int or list of length batch_size"

    for init in initial:
        if isinstance(init, Beam):
            beams = [init]
        elif len(init) == 0:
            beams = []
        elif isinstance(init[0], int):
            beams = [Beam(init)]  # type: ignore
        elif isinstance(init[0], Beam):
            beams = init
        else:
            raise ValueError("invalid initial beam type")

        current_beams.append(beams)  # type: ignore
        update_info.append(len(beams))
        beam_queues.append([])

    def should_stop(beam: Beam) -> bool:
        return (
            stop_fn(beam)
            or len(beam) >= max_length
            or beam.decoded_length >= (max_new_tokens or beam.decoded_length + 1)
        )

    def filter_beams() -> bool:
        finished = True
        for idx in range(batch_size):
            new_beams = []
            for beam in current_beams[idx]:
                if should_stop(beam):
                    beam_queues[idx].append(beam)
                else:
                    new_beams.append(beam)

            current_beams[idx] = new_beams
            finished = finished and (
                len(current_beams[idx]) == 0
                or len(beam_queues[idx]) >= max_outputs[idx]
            )
        return finished

    def get_outputs(intermediate: bool) -> list[list[Beam]]:
        outputs = []
        for idx in range(batch_size):
            beam_queue = beam_queues[idx]
            current = current_beams[idx]
            if intermediate:
                # for intermediate outputs we
                # return the active beams, so swap here
                beam_queue, current = current, beam_queue
                n = beam_width
            else:
                n = max_outputs[idx]

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

            outputs.append(beam_queue[:n])

        return outputs

    while not filter_beams():
        num_beams = []
        beams = []
        decoder_mask = []
        decoder_token_ids = []
        decoder_lengths = []
        for idx in range(batch_size):
            num = len(current_beams[idx])
            num_beams.append(num)
            decoder_mask.extend([idx] * num)
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

        if kwargs_update_fn is not None and decoder_info is not None:
            update_mask = []
            for idx in range(batch_size):
                update_mask.extend([idx] * update_info[idx])
            kwargs_update_fn(
                kwargs,
                decoder_info,
                torch.tensor(update_mask, dtype=torch.long)
            )

        if kwargs_select_fn is not None:
            decoder_kwargs = kwargs_select_fn(
                kwargs,
                decoder_mask
            )
        else:
            decoder_kwargs = {}
        # lengths are added automatically, do not provide them yourself"
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

        org_log_probs = torch.log_softmax(decoder_outputs, dim=-1)

        # apply logit functions
        for logit_fn in logit_fns or []:
            decoder_outputs = logit_fn(
                decoder_token_ids,
                decoder_outputs,
                beams
            )

        log_probs = torch.log_softmax(decoder_outputs, dim=-1)

        for idx, (log_probs, org_log_probs) in enumerate(zip(
            torch.split(log_probs, num_beams),
            torch.split(org_log_probs, num_beams)
        )):
            candidates: list[Beam] = []
            for beam_idx, beam in enumerate(current_beams[idx]):
                for token_id in sample_fn(log_probs[beam_idx], beam_width).tolist():
                    candidate = beam.clone()
                    candidate.add(
                        token_id,
                        org_log_probs[beam_idx, token_id].item()
                    )
                    candidates.append(candidate)

            # reset current beams and fill with best candidates
            current_beams[idx] = []
            for candidate in sorted(
                candidates,
                key=lambda b: score_fn(b),
                reverse=True
            )[:beam_width]:
                # update candidates
                candidate = update_fn(candidate)
                if candidate is None:
                    # skip invalid candidates
                    continue
                else:
                    current_beams[idx].append(candidate)

            update_info[idx] = len(current_beams[idx])

        if yield_intermediate:
            yield get_outputs(intermediate=True)

    yield get_outputs(intermediate=False)
