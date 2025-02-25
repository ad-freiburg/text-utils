from typing import Generator

import torch
from torch.nn.utils.rnn import pad_sequence

from text_utils.inference.utils import (
    Beam,
    CacheFn,
    DecodeFn,
    LogitFn,
    SampleFn,
    ScoreFn,
    StopFn,
    UpdateFn,
    greedy,
    identity_update,
    log_likelihood_score,
)


@torch.inference_mode()
def beam_search(
    decode_fn: DecodeFn,
    initial: list[list[int]] | list[Beam],
    pad_token_id: int,
    max_length: int,
    stop_fn: StopFn,
    device: torch.device,
    beam_width: int,
    sample_fn: SampleFn = greedy(),
    update_fn: UpdateFn = identity_update(),
    score_fn: ScoreFn = log_likelihood_score(),
    logit_fns: list[LogitFn] | None = None,
    cache_fn: CacheFn | None = None,
    stop_condition: str = "estimated_score",
    max_new_tokens: int | None = None,
    yield_intermediate: bool = False,
) -> Generator[list[list[Beam]], None, list[list[Beam]]]:
    assert (
        max_new_tokens is None or max_new_tokens > 0
    ), "max_new_tokens must be None or positive"
    assert stop_condition in {
        "max_score",
        "estimated_score",
        "max_outputs",
    }, "stop condition must be 'max_score', 'estimated_score' or 'max_outputs'"
    assert beam_width >= 1, "beam width must be greater than or equal to 1"
    batch_size = len(initial)

    current_beams: list[list[Beam]] = []
    finished_beams: list[list[Beam]] = []
    too_long_beams: list[list[Beam]] = []

    for init in initial:
        if isinstance(init, Beam):
            beams = [init.clone()]
        else:
            # init beam from token ids
            beams = [Beam(init)]

        current_beams.append(beams)  # type: ignore
        finished_beams.append([])
        too_long_beams.append([])

    def too_long(beam: Beam) -> bool:
        if len(beam) >= max_length:
            return True
        elif max_new_tokens is None:
            return False
        else:
            return beam.decoded_length >= max_new_tokens

    def filter_beams() -> tuple[list[Beam], list[int]]:
        for idx in range(batch_size):
            new_beams = []
            for beam in current_beams[idx]:
                if stop_fn(beam):
                    beam.stop_reason = "done"
                    finished_beams[idx].append(beam)
                elif too_long(beam):
                    beam.stop_reason = "length"
                    too_long_beams[idx].append(beam)
                else:
                    new_beams.append(beam)

            current_beams[idx] = new_beams
            if not new_beams:
                # we are done with this batch element
                continue

            elif len(finished_beams[idx]) < beam_width:
                continue

            elif stop_condition == "max_outputs":
                # we are done with this batch element
                # because we have enough finished beams
                current_beams[idx] = []
                continue

            worst_finished = min(
                (score_fn(b) for b in finished_beams[idx]),
                default=float("-inf"),
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
                current = next(b for b in current_beams[idx])
                max_decoded_length = max_length - current.initial_length
                length = min(max_decoded_length, max_new_tokens or max_decoded_length)
                best_current = max(score_fn(b, length) for b in current_beams[idx] if b)

            if worst_finished >= best_current:
                # set current beams to None list to stop processing
                current_beams[idx] = []

        beams = []
        indices = []
        for idx in range(batch_size):
            beams.extend(current_beams[idx])
            indices.extend([idx] * len(current_beams[idx]))
        return beams, indices

    def get_outputs() -> list[list[Beam]]:
        outputs = []
        for batch_idx in range(batch_size):
            output_beams = finished_beams[batch_idx] + too_long_beams[batch_idx]
            output_beams = sorted(output_beams, key=score_fn, reverse=True)
            outputs.append(output_beams)

        return outputs

    single = beam_width == 1
    beams, indices = filter_beams()
    cache = None
    cache_mask = None

    while beams:
        if cache is not None and all(beam.cache is not None for beam in beams):
            assert cache_fn is not None, "cache_fn must be provided if cache is used"
            new_cache_mask = [beam.cache for beam in beams]
            if new_cache_mask != cache_mask:
                # update cache with mask, only if it has changed
                # (saves lots of unnecessary cache updates)
                cache = cache_fn(cache, new_cache_mask)  # type: ignore
                cache_mask = new_cache_mask

            # if we have a cache, only the last input token is needed
            input_ids = torch.tensor(
                [beam.last_token_id for beam in beams],
                dtype=torch.long,
                device=device,
            ).unsqueeze(1)
        else:
            # is there is no cache or any beam has its cache reset,
            # provide the full input sequence to the model
            input_ids = pad_sequence(
                [torch.tensor(beam.token_ids) for beam in beams],
                batch_first=True,
                padding_value=pad_token_id,
                padding_side="left",
            ).to(device)
            if cache is not None:
                # reset cache
                cache = None
                cache_mask = None
                torch.cuda.empty_cache()

        logits, cache = decode_fn(input_ids, cache)
        log_probs = torch.log_softmax(logits, dim=-1)

        # apply logit functions
        for logit_fn in logit_fns or []:
            logits = logit_fn(logits, beams)

        selected_ids, selected_logits = sample_fn(logits, beam_width)
        # filter out invalid ids by checking logits for -inf
        # (prob = 0 after softmax)
        valid_ids = torch.logical_not(torch.isneginf(selected_logits))

        batch_candidates: list[list[Beam]] = [[] for _ in range(batch_size)]

        for i, batch_idx in enumerate(indices):
            beam = beams[i]
            # set cache index for beam
            beam.cache = i

            for token_id in selected_ids[i, valid_ids[i]]:
                if single:
                    candidate = beam
                else:
                    # must clone here when beam_width > 1
                    candidate = beam.clone()

                token_id = int(token_id.item())
                candidate.add(token_id, log_probs[i, token_id].item())
                batch_candidates[batch_idx].append(candidate)

        for batch_idx, candidates in enumerate(batch_candidates):
            # reset current beams and fill with best candidates
            current_beams[batch_idx] = []

            # score and sort candidates
            candidates = sorted(candidates, key=score_fn, reverse=True)

            for candidate in candidates:
                # update candidates
                candidate = update_fn(candidate)
                if candidate is None:
                    # skip invalid candidates
                    continue

                current_beams[batch_idx].append(candidate)
                if len(current_beams[batch_idx]) >= beam_width:
                    break

        beams, indices = filter_beams()

        if yield_intermediate:
            yield get_outputs()

    return get_outputs()
