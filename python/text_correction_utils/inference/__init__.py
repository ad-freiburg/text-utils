import heapq
from typing import Callable, Tuple, List, Optional, Union, Dict, Any

import einops
import torch

DecodeFn = Callable[[torch.Tensor, ...], torch.Tensor]


class Beam:
    def __init__(
            self,
            token_ids: List[int],
            log_prob: List[float],
            num_initial_tokens: Optional[int] = None
    ) -> None:
        self._token_ids: List[int] = token_ids
        self._log_prob: List[float] = log_prob
        if num_initial_tokens is not None:
            assert num_initial_tokens <= len(token_ids), \
                f"number of initial tokens {num_initial_tokens} cannot be large than {len(token_ids)}"
        else:
            num_initial_tokens = len(token_ids)
        self._num_initial_tokens: int = num_initial_tokens

    @property
    def token_ids(self) -> List[int]:
        return self._token_ids

    @property
    def decoded_log_p(self) -> float:
        return float(sum(self._log_prob[self._num_initial_tokens:]))

    @property
    def decoded_token_ids(self) -> List[int]:
        return self._token_ids[self._num_initial_tokens:]

    @property
    def decoded_token_length(self) -> int:
        return len(self) - self._num_initial_tokens  # do not count initial tokens to decoded length

    @staticmethod
    def from_beam(other: "Beam", log_p: float, token_id: int) -> "Beam":
        beam = Beam(other._token_ids + [token_id], other._log_prob + [log_p], other._num_initial_tokens)
        return beam

    def __lt__(self, other: "Beam") -> bool:
        assert self._num_initial_tokens == other._num_initial_tokens, \
            "only compare beams with the same number of initial tokens"
        return self.decoded_token_length < other.decoded_token_length

    def __len__(self) -> int:
        return len(self._token_ids)

    def __repr__(self) -> str:
        return f"Beam(token_ids={self._token_ids}, log_prob={self._log_prob}, " \
               f"num_initial_tokens={self._num_initial_tokens})"


TokFn = Callable[[str], List[int]]
DeTokFn = Callable[[List[int]], str]
ReScoreFn = Callable[
    [
        # beam or list of token ids
        Beam,
        # optional input string
        Optional[str]
    ],
    float
]
StopFn = Callable[[List[int], Optional[str]], bool]
IdxSelectFn = Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


def eos_stop_fn(eos_token_id: int) -> StopFn:
    def _stop(token_ids: List[int], _: Optional[str] = None) -> bool:
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
            assert beam.decoded_token_length > 0, \
                "expected to only score beams with normalization if at least one token was decoded"
            return beam.decoded_log_p / (beam.decoded_token_length ** alpha)
        else:
            return beam.decoded_log_p

    return score


def _sub_select(inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], mask: Union[int, torch.Tensor]) -> \
        Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(inputs, torch.Tensor):
        return inputs[mask]
    elif isinstance(inputs, dict):
        return {k: v[mask] for k, v in inputs.items()}
    else:
        raise ValueError(f"expected inputs to be of type tensor or dict of tensors, but got {type(inputs)}")


@torch.inference_mode()
def token_inference(
        model: DecodeFn,
        device: torch.device,
        encoder_outputs: Dict[str, torch.Tensor],
        encoder_lengths: Dict[str, torch.Tensor],
        bos_token_id: int,
        pad_token_id: int,
        max_length: int,
        select_fn: IdxSelectFn,
        stop_fn: StopFn,
        decoder_positions: Optional[torch.Tensor] = None,
        tok_fn: Optional[TokFn] = None,
        output_strings: Optional[List[str]] = None
) -> List[List[int]]:
    batch_size = len(next(iter(encoder_outputs.values())))

    log_prob = torch.full((batch_size, max_length + 1), fill_value=-1.0, device=device)
    token_ids = torch.full((batch_size, max_length + 1), fill_value=pad_token_id, dtype=torch.long, device=device)
    if output_strings is None:
        log_prob[:, 0] = 0.0
        token_ids[:, 0] = bos_token_id
        lengths = torch.ones(batch_size, dtype=torch.long, device=device)
    else:
        assert tok_fn is not None, "tokenization function must be given when output strings are specified"
        assert len(output_strings) == batch_size
        lengths = []
        for i, output_string in enumerate(output_strings):
            output_token_ids = tok_fn(output_string)
            token_ids[i, :len(output_token_ids)] = torch.tensor(
                output_token_ids, dtype=torch.long, device=token_ids.device)
            log_prob[i, :len(output_token_ids)] = 0.0
            lengths.append(len(output_token_ids))
        lengths = torch.tensor(lengths, dtype=torch.long, device=device)

    if decoder_positions is not None:
        positions = torch.stack([
            torch.arange(pos, pos + max_length + 1, device=device)
            for pos in decoder_positions
        ])
    else:
        positions = einops.repeat(torch.arange(max_length + 1, device=device), "l -> b l", b=batch_size)

    smaller_max_length_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    smaller_max_length_mask[lengths + positions[:, 0] >= max_length] = False
    non_stop_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    indices_to_decode = non_stop_mask & smaller_max_length_mask

    while True:
        # all sequences are at max length or stopped by stop_fn
        if torch.sum(indices_to_decode) == 0:
            break

        decoder_lengths = _sub_select(lengths, indices_to_decode)
        max_decoder_length = max(decoder_lengths)
        decoder_positions = _sub_select(positions, indices_to_decode)[:, :max_decoder_length]

        decoder_output = model.decode(
            decoder_inputs=_sub_select(token_ids, indices_to_decode)[:, :max_decoder_length],
            decoder_lengths=decoder_lengths,
            encoder_outputs=_sub_select(encoder_outputs, indices_to_decode),
            encoder_lengths=_sub_select(encoder_lengths, indices_to_decode),
            decoder_positions=decoder_positions
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

        max_length_indices = torch.where(lengths + positions[:, 0] >= max_length)[0]
        smaller_max_length_mask[max_length_indices] = False

        batch_indices = torch.where(indices_to_decode)[0].tolist()
        new_stop_indices = []
        for idx, length in zip(batch_indices, lengths[indices_to_decode]):
            if stop_fn(token_ids[idx][:length].tolist(), output_strings[idx] if output_strings is not None else None):
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
        model: DecoderMixin,
        encoder_outputs: Dict[str, torch.Tensor],
        encoder_lengths: Dict[str, torch.Tensor],
        bos_token_id: int,
        max_length: int,
        stop_fn: StopFn,
        re_score_fn: Optional[ReScoreFn] = None,
        input_strings: Optional[List[str]] = None,
        decoder_positions: Optional[torch.Tensor] = None,
        tok_fn: Optional[TokFn] = None,
        output_strings: Optional[List[str]] = None
) -> List[List[Beam]]:
    model.eval()
    device = utils.device_from_model(model)

    all_beams: List[List[Beam]] = []

    batch_size = len(next(iter(encoder_outputs.values())))
    positions = decoder_positions if decoder_positions is not None else torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        # encoder_outputs_b: shape [L, H]
        encoder_outputs_b = _sub_select(encoder_outputs, b)
        encoder_lengths_b = _sub_select(encoder_lengths, b)

        # initialize beams
        beam_queue = []
        finished_beams = []

        if output_strings is not None:
            token_ids = tok_fn(output_strings[b])
            log_prob = [0.0] * len(token_ids)
        else:
            token_ids = [bos_token_id]
            log_prob = [0.0]

        beam = Beam(token_ids, log_prob)
        heapq.heappush(beam_queue, (0, beam))

        while len(beam_queue):
            beam: Beam = heapq.heappop(beam_queue)[1]

            if (
                    stop_fn(beam.token_ids, output_strings[b] if output_strings is not None else None)
                    or positions[b] + len(beam) >= max_length
            ):
                finished_beams.append(beam)
                break

            continuations = einops.repeat(
                torch.tensor(beam.token_ids, dtype=torch.long, device=device),
                "l -> b l", b=1
            )
            decoder_lengths = torch.tensor(
                [len(beam)],
                dtype=torch.long,
                device=device
            )
            decoder_positions = torch.arange(
                positions[b], positions[b] + len(beam),
                device=device,
                dtype=torch.long
            ).unsqueeze(0)

            beam_encoder_feats_b = {k: einops.repeat(v, "l h -> b l h", b=len(continuations))
                                    for k, v in encoder_outputs_b.items()}
            beam_encoder_lengths_b = {k: einops.repeat(v, "-> repeat", repeat=len(continuations))
                                      for k, v in encoder_lengths_b.items()}

            # decoder_output: shape [B, VOC]
            decoder_output = model.decode(
                decoder_inputs=continuations,
                decoder_lengths=decoder_lengths,
                encoder_outputs=beam_encoder_feats_b,
                encoder_lengths=beam_encoder_lengths_b,
                decoder_positions=decoder_positions
            )[0, -1, ...]

            log_softmax_scores = torch.log_softmax(decoder_output, dim=0)
            scores = log_softmax_scores + beam.decoded_log_p

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
        model: DecoderMixin,
        encoder_outputs: Dict[str, torch.Tensor],
        encoder_lengths: Dict[str, torch.Tensor],
        bos_token_id: int,
        pad_token_id: int,
        vocab_size: int,
        max_length: int,
        stop_fn: StopFn,
        normalize_by_length: bool,
        alpha: float,
        beam_width: int,
        decoder_positions: Optional[torch.Tensor] = None,
        tok_fn: Optional[TokFn] = None,
        output_strings: Optional[List[str]] = None
) -> List[List[Beam]]:
    model.eval()
    device = utils.device_from_model(model)

    batch_size = len(next(iter(encoder_outputs.values())))

    score_fn = log_likelihood_score(normalize_by_length, alpha)
    beam_width = min(beam_width, vocab_size - 1)  # never produce pad
    select_fn = beam_select_fn(2 * beam_width)

    positions = decoder_positions.cpu() if decoder_positions is not None else torch.zeros(batch_size, dtype=torch.long)
    beam_queues: List[List[Beam]] = [[] for _ in range(batch_size)]

    search_depths: List[int] = []
    current_beams: List[List[Beam]] = []
    for b in range(batch_size):
        # initialize beams
        if output_strings is not None:
            token_ids = tok_fn(output_strings[b])
            log_prob = [0.0] * len(token_ids)
        else:
            token_ids = [bos_token_id]
            log_prob = [0.0]
        beam = Beam(token_ids, log_prob)
        current_beams.append([beam])
        search_depths.append(len(beam))

    stop_mask = [False for _ in range(batch_size)]

    while True:
        decoder_mask = []
        for stop, beam_queue, position, search_depth, beams in zip(
                stop_mask, beam_queues, positions, search_depths, current_beams
        ):
            decoder_mask.append(
                not stop
                and position + search_depth < max_length
                and len(beams)
            )

        if not any(decoder_mask):
            break

        decoder_mask_tensor = torch.tensor(decoder_mask, dtype=torch.bool, device=device)
        indices_to_decode = torch.where(decoder_mask_tensor)[0]
        num_beams = [len(beams) for i, beams in enumerate(current_beams) if decoder_mask[i]]
        num_beams_tensor = torch.tensor(num_beams, dtype=torch.long)

        decoder_log_probs = []
        decoder_inputs = []
        decoder_lengths = []
        for i, beams in enumerate(current_beams):
            if not decoder_mask[i]:
                continue
            for beam in beams:
                decoder_log_probs.append(beam.decoded_log_p)
                decoder_lengths.append(len(beam))
                decoder_inputs.append(torch.tensor(beam.token_ids, dtype=torch.long))

        decoder_inputs = to(utils.pad(decoder_inputs, pad_token_id), device)
        decoder_log_probs_tensor = torch.tensor(decoder_log_probs, device=device)
        decoder_lengths_tensor = torch.tensor(decoder_lengths, device=device)

        decoder_positions = to(
            torch.repeat_interleave(
                torch.stack([
                    torch.arange(pos, pos + max(decoder_lengths), dtype=torch.long)
                    for pos in positions[decoder_mask_tensor]]
                ),
                num_beams_tensor,
                dim=0
            ),
            device
        )

        beam_encoder_lengths = {
            k: torch.repeat_interleave(v, num_beams_tensor, dim=0)
            for k, v in _sub_select(encoder_lengths, decoder_mask_tensor).items()
        }
        num_beams_tensor = to(num_beams_tensor, device)
        beam_encoder_outputs = {
            k: torch.repeat_interleave(v, num_beams_tensor, dim=0)
            for k, v in _sub_select(encoder_outputs, decoder_mask_tensor).items()
        }

        decoder_outputs = model.decode(
            decoder_inputs=decoder_inputs,
            decoder_lengths=decoder_lengths_tensor,
            encoder_outputs=beam_encoder_outputs,
            encoder_lengths=beam_encoder_lengths,
            decoder_positions=decoder_positions
        )[torch.arange(len(decoder_inputs), device=device), decoder_lengths_tensor - 1, ...]

        log_softmax_scores = torch.log_softmax(decoder_outputs, dim=1)
        beam_token_ids, beam_log_probs = select_fn(log_softmax_scores)
        beam_scores = (decoder_log_probs_tensor.unsqueeze(1) + beam_log_probs).tolist()
        beam_token_ids, beam_log_probs = beam_token_ids.tolist(), beam_log_probs.tolist()

        for beam_scores_b, token_ids_b, log_probs_b, idx in zip(
                utils.split(beam_scores, num_beams),
                utils.split(beam_token_ids, num_beams),
                utils.split(beam_log_probs, num_beams),
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
                        output_strings[idx] if output_strings is not None else None
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


def get_tok_fn(output_tokenizer: tokenization.Tokenizer, bos_token_id: int) -> Callable[[str], List[int]]:
    def tokenize(string: str) -> List[int]:
        return [bos_token_id] + output_tokenizer.tokenize(string)

    return tokenize


def get_de_tok_fn(output_tokenizer: tokenization.Tokenizer, bos_token_id: int, eos_token_id: int) \
        -> Callable[[List[int]], str]:
    def de_tokenize(token_ids: List[int]) -> str:
        # strip bos and eos tokens before de-tokenization
        if len(token_ids) and token_ids[0] == bos_token_id:
            token_ids = token_ids[1:]
        if len(token_ids) and token_ids[-1] == eos_token_id:
            token_ids = token_ids[:-1]
        return output_tokenizer.de_tokenize(token_ids)

    return de_tokenize


def run_inference(
        model: DecoderMixin,
        output_tokenizer: tokenization.Tokenizer,
        encoder_outputs: Dict[str, torch.Tensor],
        encoder_lengths: Dict[str, torch.Tensor],
        max_length: int,
        search_mode: str = "greedy",
        stop_fn: Optional[StopFn] = None,
        normalize_by_length: bool = True,
        alpha: float = 1.0,
        re_score_fn: Optional[ReScoreFn] = None,
        input_strings: Optional[List[str]] = None,
        output_strings: Optional[List[str]] = None,
        decoder_positions: Optional[torch.Tensor] = None,
        **kwargs: Any
) -> List[List[str]]:
    bos_token_id = output_tokenizer.token_to_id(tokenization.BOS)
    eos_token_id = output_tokenizer.token_to_id(tokenization.EOS)
    pad_token_id = output_tokenizer.token_to_id(tokenization.PAD)
    tok_fn = get_tok_fn(output_tokenizer, bos_token_id)
    de_tok_fn = get_de_tok_fn(output_tokenizer, bos_token_id, eos_token_id)
    stop_fn = stop_fn or eos_stop_fn(eos_token_id)

    if search_mode == "greedy":
        outputs = token_inference(
            model=model,
            encoder_outputs=encoder_outputs,
            encoder_lengths=encoder_lengths,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            max_length=max_length,
            select_fn=greedy_select_fn(),
            stop_fn=stop_fn,
            tok_fn=tok_fn,
            output_strings=output_strings,
            decoder_positions=decoder_positions
        )
    elif search_mode == "sample":
        sample_top_k = kwargs.pop("sample_top_k", 5)
        outputs = token_inference(
            model=model,
            encoder_outputs=encoder_outputs,
            encoder_lengths=encoder_lengths,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            max_length=max_length,
            select_fn=sample_select_fn(sample_top_k),
            stop_fn=stop_fn,
            tok_fn=tok_fn,
            output_strings=output_strings,
            decoder_positions=decoder_positions
        )
    elif search_mode == "beam":
        beam_width = kwargs.pop("beam_width", 5)
        outputs = beam_inference(
            model=model,
            encoder_outputs=encoder_outputs,
            encoder_lengths=encoder_lengths,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            vocab_size=output_tokenizer.vocab_size,
            max_length=max_length,
            stop_fn=stop_fn,
            normalize_by_length=normalize_by_length,
            alpha=alpha,
            beam_width=beam_width,
            tok_fn=tok_fn,
            output_strings=output_strings,
            decoder_positions=decoder_positions
        )
    elif search_mode == "best_first":
        outputs = best_first_inference(
            model=model,
            encoder_outputs=encoder_outputs,
            encoder_lengths=encoder_lengths,
            bos_token_id=bos_token_id,
            max_length=max_length,
            stop_fn=stop_fn,
            re_score_fn=re_score_fn,
            input_strings=input_strings,
            tok_fn=tok_fn,
            output_strings=output_strings,
            decoder_positions=decoder_positions
        )
    else:
        raise ValueError(f"unknown search mode {search_mode}")

    outputs = [inference_result_to_sequence(ir, de_tok_fn) for ir in outputs]
    return outputs


def inference_result_to_sequence(
        inference_result: Union[List[int], List[Beam]],
        de_tok_fn: DeTokFn
) -> List[str]:
    if isinstance(inference_result, list) and all(isinstance(e, int) for e in inference_result):
        # greedy or sample inference
        return [de_tok_fn(inference_result)]
    elif isinstance(inference_result, list) and all(isinstance(e, Beam) for e in inference_result):
        # beam or best first inference
        return [de_tok_fn(beam.token_ids) for beam in inference_result]
    else:
        raise ValueError(f"expected inference result to be either a list of token ids"
                         f"or a list of beams, but got {type(inference_result)}")


def inference_output_to_str(output: Union[int, List[int], List[str], str]) -> str:
    if isinstance(output, int):
        return str(output)
    elif isinstance(output, list) and all(isinstance(o, int) for o in output):
        return " ".join(str(o) for o in output)
    elif isinstance(output, list) and all(isinstance(o, str) for o in output):
        # for inference outputs consisting of multiple strings (e.g. beam search) only take the top one
        return output[0]
    elif isinstance(output, str):
        return output
    else:
        raise ValueError(f"output has to be either an int, a list of ints or strings, or a string, "
                         f"but got {type(output)} ({output})")
