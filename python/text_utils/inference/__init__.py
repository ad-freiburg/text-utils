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
    greedy,
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
    stop_condition: str = "estimated_score",
    max_new_tokens: int | None = None,
    return_incomplete: bool = False,
    yield_intermediate: bool = False,
    **kwargs: Any,
) -> Iterator[list[list[Beam]]]:
    assert (
        max_new_tokens is None or max_new_tokens > 0
    ), "max_new_tokens must be None or positive"
    assert stop_condition in {
        "max_score",
        "estimated_score",
        "max_outputs",
    }, "stop condition must be 'max_score', 'estimated_score' or 'max_outputs'"
    batch_size = len(initial)

    decoder_info: Any | None = None
    update_info: list[int] = []
    current_beams: list[list[Beam]] = []
    finished_beams: list[list[Beam]] = []
    too_long_beams: list[list[Beam]] = []

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
        finished_beams.append([])
        too_long_beams.append([])

    def filter_beams() -> bool:
        finished = True
        for idx in range(batch_size):
            new_beams = []
            for beam in current_beams[idx]:
                if stop_fn(beam):
                    finished_beams[idx].append(beam)
                elif len(beam) >= max_length or beam.decoded_length >= (
                    max_new_tokens or (beam.decoded_length + 1)
                ):
                    too_long_beams[idx].append(beam)
                else:
                    new_beams.append(beam)

            current_beams[idx] = new_beams
            if not current_beams[idx]:
                # we are done with this batch element
                continue

            elif len(finished_beams[idx]) < beam_width:
                finished = False
                continue

            elif stop_condition == "max_outputs":
                # we are done with this batch element
                # because we have enough finished beams
                current_beams[idx] = []
                continue

            worst_finished = min(
                (score_fn(b) for b in finished_beams[idx]), default=float("-inf")
            )
            if stop_condition == "estimated_score":
                # best current calculated from current length
                # idea: is a current active beam better than the worst finished beam?
                best_current = max(score_fn(b) for b in current_beams[idx])
            else:
                # best current calculated from maximum length
                # idea: assume all remaining tokens are perfectly predicted
                # with probability 1.0, can a current active beam be better
                # than the worst finished beam?
                current = current_beams[idx][0]
                max_decoded_length = max_length - current.initial_length
                length = min(max_decoded_length, max_new_tokens or max_decoded_length)
                best_current = max(score_fn(b, length) for b in current_beams[idx])

            if worst_finished >= best_current:
                # set current beams to empty list to stop processing
                current_beams[idx] = []
            else:
                finished = False

        return finished

    def get_outputs(intermediate: bool) -> list[list[Beam]]:
        outputs = []
        for idx in range(batch_size):
            current = current_beams[idx]

            if return_incomplete:
                finished = finished_beams[idx] + too_long_beams[idx]
            else:
                finished = finished_beams[idx]

            if intermediate:
                # for intermediate outputs we
                # return the active beams first if available
                beams = current if current else finished
            else:
                beams = finished

            beams = sorted(beams, key=score_fn, reverse=True)
            outputs.append(beams[:beam_width])

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
                decoder_token_ids.append(torch.tensor(beam.token_ids, dtype=torch.long))

        decoder_token_ids = rnn.pad_sequence(
            decoder_token_ids, batch_first=True, padding_value=pad_token_id
        ).to(non_blocking=True, dtype=torch.long, device=device)
        decoder_mask = torch.tensor(decoder_mask, dtype=torch.long)
        decoder_lengths_tensor = torch.tensor(decoder_lengths, dtype=torch.long)

        if kwargs_update_fn is not None and decoder_info is not None:
            update_mask = []
            for idx in range(batch_size):
                update_mask.extend([idx] * update_info[idx])
            kwargs_update_fn(
                kwargs, decoder_info, torch.tensor(update_mask, dtype=torch.long)
            )

        if kwargs_select_fn is not None:
            decoder_kwargs = kwargs_select_fn(kwargs, decoder_mask)
        else:
            decoder_kwargs = {}
        # lengths are added automatically, do not provide them yourself"
        decoder_kwargs["lengths"] = decoder_lengths_tensor

        decoder_outputs, decoder_info = decode_fn(decoder_token_ids, **decoder_kwargs)
        b, s, _ = decoder_outputs.shape
        if s == 1:
            decoder_outputs = decoder_outputs[:, 0]
        else:
            decoder_outputs = decoder_outputs[
                torch.arange(b), decoder_lengths_tensor - 1
            ]

        raw_log_probs = torch.log_softmax(decoder_outputs, dim=-1)

        # apply logit functions
        for logit_fn in logit_fns or []:
            decoder_outputs = logit_fn(decoder_token_ids, decoder_outputs, beams)

        log_probs = torch.log_softmax(decoder_outputs, dim=-1)

        raw_log_probs = torch.split(raw_log_probs, num_beams)
        log_probs = torch.split(log_probs, num_beams)
        for idx, (raw_log_prob, log_prob) in enumerate(zip(raw_log_probs, log_probs)):
            candidates: list[Beam] = []
            for beam_idx, beam in enumerate(current_beams[idx]):
                for token_id in sample_fn(log_prob[beam_idx], beam_width).tolist():
                    candidate = beam.clone()
                    candidate.add(token_id, raw_log_prob[beam_idx, token_id].item())
                    candidates.append(candidate)

            # reset current beams and fill with best candidates
            current_beams[idx] = []
            for candidate in sorted(candidates, key=score_fn, reverse=True):
                # update candidates
                candidate = update_fn(candidate)
                if candidate is None:
                    # skip invalid candidates
                    continue
                elif len(current_beams[idx]) < beam_width:
                    current_beams[idx].append(candidate)
                else:
                    break

            update_info[idx] = len(current_beams[idx])

        if yield_intermediate:
            yield get_outputs(intermediate=True)

    yield get_outputs(intermediate=False)
