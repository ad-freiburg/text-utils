from typing import Callable, Any

import torch

from grammar_utils.constrain import Constraint

# maps from token ids, length, and other kwargs to distribution over next token id and other info
DecodeFn = Callable[..., tuple[torch.Tensor, dict[str, Any]]]

# select specific elements for all the kwargs keys given the mask tensor
MaskSelectFn = Callable[[dict[str, Any], torch.Tensor], dict[str, Any]]

# update specific elements for all the kwargs keys given the mask tensor
MaskUpdateFn = Callable[[dict[str, Any], dict[str, Any], torch.Tensor], None]


class Beam:
    def __init__(
        self,
        token_ids: list[int],
        log_probs: list[float] | None = None,
        info: dict[str, Any] | None = None,
        initial_length: int | None = None,
    ) -> None:
        if log_probs is None:
            log_probs = [0.0] * len(token_ids)
        assert len(token_ids) == len(
            log_probs
        ), "expected token_ids and log_probs to have the same length"
        self.token_ids = token_ids
        self.log_probs = log_probs
        if initial_length is None:
            initial_length = len(token_ids)
        self.initial_length = initial_length
        self.info: dict[str, Any] = info or {}

    def add(self, token_id: int, log_p: float) -> None:
        self.token_ids.append(token_id)
        self.log_probs.append(log_p)

    def clone(self) -> "Beam":
        return Beam(
            self.token_ids.copy(),
            self.log_probs.copy(),
            self.info.copy(),
            self.initial_length,
        )

    @property
    def initial_token_ids(self) -> list[int]:
        return self.token_ids[: self.initial_length]

    @property
    def initial_log_probs(self) -> list[float]:
        return self.log_probs[: self.initial_length]

    @property
    def decoded_token_ids(self) -> list[int]:
        return self.token_ids[self.initial_length :]

    @property
    def decoded_log_probs(self) -> list[float]:
        return self.log_probs[self.initial_length :]

    @property
    def log_prob(self) -> float:
        return sum(self.log_probs)

    @property
    def decoded_log_prob(self) -> float:
        return sum(self.decoded_log_probs)

    @property
    def decoded_length(self) -> int:
        return len(self) - self.initial_length

    def __lt__(self, other: "Beam") -> bool:
        return len(self) < len(other)

    def __len__(self) -> int:
        return len(self.token_ids)

    def __repr__(self) -> str:
        return f"Beam(token_ids={self.token_ids}, log_prob={self.log_prob:.4f})"


# processes logits and returns new logits
LogitFn = Callable[
    [
        # token ids, shape [batch_size, seq_len]
        torch.Tensor,
        # logits, shape [batch_size, vocab_size]
        torch.Tensor,
        # beams being processed
        list[Beam],
    ],
    # new logits, shape [batch_size, vocab_size]
    torch.Tensor,
]

# takes in log probs and beam width and returns
# beam width samples
SampleFn = Callable[
    [
        # distribution over next tokens, shape [vocab_size]
        torch.Tensor,
        # beam width
        int,
    ],
    # indices of selected tokens, shape [<= beam_width]
    torch.Tensor,
]

# checks if beam should be stopped
StopFn = Callable[
    [
        # beam checked for stopping
        Beam,
    ],
    # bool indicating if beam should be stopped
    bool,
]

# takes in a beam candidate and returns an updated beam
UpdateFn = Callable[[Beam], Beam | None]


BeamWidthFn = Callable[
    [
        # beam to calculate beam width from
        Beam
    ],
    # beam width
    int,
]


# takes in a beam and returns a scalar score
ScoreFn = Callable[[Beam], float]


def log_likelihood_score(
    normalize_by_length: bool = True, alpha: float = 1.0, full: bool = False
) -> ScoreFn:
    def _score(beam: Beam) -> float:
        if full:
            log_prob = beam.log_prob
            length = len(beam)
        else:
            log_prob = beam.decoded_log_prob
            length = beam.decoded_length

        if normalize_by_length and length > 0:
            return log_prob / (length**alpha)
        else:
            return log_prob

    return _score


def constraint_logit_fn(
    retrieve_constraint_fn: Callable[[Beam], Constraint | None], eos_token_id: int
) -> LogitFn:
    def _constrain_logits(
        _: torch.Tensor, logits: torch.Tensor, beams: list[Beam]
    ) -> torch.Tensor:
        zeros = torch.full_like(logits, float("-inf"))

        for i, beam_or_idx in enumerate(beams):
            constraint = retrieve_constraint_fn(beam_or_idx)

            if constraint is None or constraint.is_invalid():
                zeros[i] = logits[i]
                continue

            indices = torch.from_numpy(constraint.get()).to(torch.int32)
            zeros[i, indices] = logits[i, indices]

            if constraint.is_match():
                zeros[i, eos_token_id] = logits[i, eos_token_id]

        return zeros

    return _constrain_logits


def allow_tokens_logit_fn(allowed_tokens: list[int]) -> LogitFn:
    allowed = torch.tensor(allowed_tokens, dtype=torch.long)

    def _allow_tokens(
        _input_ids: torch.Tensor, logits: torch.Tensor, _beams: list[Beam]
    ) -> torch.Tensor:
        zeros = torch.full_like(logits, float("-inf"))
        zeros[:, allowed] = logits[:, allowed]
        return zeros

    return _allow_tokens


def identity_update_fn() -> UpdateFn:
    def _update_fn(beam: Beam) -> Beam:
        return beam

    return _update_fn


def sample() -> SampleFn:
    def _sample(logits: torch.Tensor, k: int) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        k = min(k, probs.shape[-1], int(torch.sum(probs > 0).item()))
        return torch.multinomial(probs, k)

    return _sample


def greedy() -> SampleFn:
    def _greedy(logits: torch.Tensor, k: int) -> torch.Tensor:
        k = min(k, logits.shape[-1] - int(torch.sum(torch.isinf(logits)).item()))
        return torch.topk(logits, k, dim=-1).indices

    return _greedy


def repetition_penalty(penalty: float) -> LogitFn:
    def _repetition_penalty(
        input_ids: torch.Tensor, logits: torch.Tensor, _beams: list[Beam]
    ) -> torch.Tensor:
        logit = torch.gather(logits, -1, input_ids)
        logit = torch.where(logit > 0, logit / penalty, logit * penalty)
        return logits.scatter_(-1, input_ids, logit)

    return _repetition_penalty


def temperature_scaling(temp: float) -> LogitFn:
    def _temperature_scaling(
        _input_ids: torch.Tensor, logits: torch.Tensor, _beams: list[Beam]
    ) -> torch.Tensor:
        return logits / temp

    return _temperature_scaling


def top_k_masking(k: int) -> LogitFn:
    def _top_k(
        _input_ids: torch.Tensor, logits: torch.Tensor, _beams: list[Beam]
    ) -> torch.Tensor:
        topk = torch.full_like(logits, float("-inf"))
        values, indices = torch.topk(logits, min(k, logits.shape[-1]), dim=-1)
        topk.scatter_(-1, indices, values)
        return topk

    return _top_k


def nucleus_masking(p: float) -> LogitFn:
    def _nuc(
        _input_ids: torch.Tensor, logits: torch.Tensor, _: list[Beam]
    ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
        cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
        nucleus = cum_sum_probs < p
        nucleus = torch.cat(
            [nucleus.new_ones((len(nucleus), 1)), nucleus[:, :-1]], dim=-1
        )
        sorted_logits = torch.gather(logits, -1, indices)
        sorted_logits[torch.logical_not(nucleus)] = float("-inf")
        return sorted_logits.gather(-1, indices.argsort(-1))

    return _nuc
