from typing import Any, Callable, Protocol

import torch
from grammar_utils.constrain import Constraint

# maps from token ids and optional cache
# to distribution over next tokens and optional cache
DecodeFn = Callable[
    [torch.Tensor, Any | None],
    tuple[torch.Tensor, Any | None],
]

# update cache according to given mask
CacheFn = Callable[[Any, list[int]], Any]


class Beam:
    def __init__(
        self,
        token_ids: list[int],
        log_probs: list[float] | None = None,
        initial_length: int | None = None,
        info: dict[str, Any] | None = None,
        cache: int | None = None,
        stop_reason: str | None = None,
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
        self.cache = cache
        self.stop_reason = stop_reason

    def add(self, token_id: int, log_p: float) -> None:
        self.token_ids.append(token_id)
        self.log_probs.append(log_p)

    def clone(self) -> "Beam":
        return Beam(
            self.token_ids.copy(),
            self.log_probs.copy(),
            self.initial_length,
            self.info.copy(),
            self.cache,
            self.stop_reason,
        )

    @property
    def last_token_id(self) -> int:
        return self.token_ids[-1]

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
    def length(self) -> int:
        return len(self)

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
        # logits over next tokens, shape [batch_size, vocab_size]
        torch.Tensor,
        # num samples / beam width
        int,
    ],
    # indices of selected tokens, shape [batch_size, beam_width]
    # and logits of selected tokens, shape [batch_size, beam_width]
    # (can be used to filter for invalid selection with -inf logit)
    tuple[torch.Tensor, torch.Tensor],
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

# takes in a beam candidate and returns updated beam or None
UpdateFn = Callable[[Beam], Beam | None]


BeamWidthFn = Callable[
    [
        # beam to calculate beam width from
        Beam
    ],
    # beam width
    int,
]


# takes in a beam (and optional length) and returns a scalar score
class ScoreFn(Protocol):
    def __call__(self, beam: Beam, length: int | None = None) -> float: ...


def log_likelihood_score(normalize: bool = True, alpha: float = 1.0) -> ScoreFn:
    assert alpha >= 0.0, "alpha must be positive"

    def _score(beam: Beam, length: int | None = None) -> float:
        log_prob = beam.decoded_log_prob
        if length is None:
            length = beam.decoded_length

        if normalize and length > 0:
            return log_prob / (length**alpha)
        else:
            return log_prob

    return _score


def constrain(
    get_constraint: Callable[[Beam], Constraint | None],
    eos_token_id: int,
    always_allow_eos: bool = False,
) -> LogitFn:
    def _constrain(
        logits: torch.Tensor,
        beams: list[Beam],
    ) -> torch.Tensor:
        zeros = torch.full_like(logits, float("-inf"))

        for i, beam in enumerate(beams):
            constraint = get_constraint(beam)

            if constraint is None or constraint.is_invalid():
                # fallback to unconstrained logits
                zeros[i] = logits[i]
                continue

            indices = torch.from_numpy(constraint.get()).to(torch.int32)
            zeros[i, indices] = logits[i, indices]

            if constraint.is_match() or always_allow_eos:
                zeros[i, eos_token_id] = logits[i, eos_token_id]
            else:
                # eos token might still be valid on non-match beams
                # if its text representation, e.g. <|endoftext|>, is allowed
                # by the constraint, e.g. ".+"; avoid this by setting the
                # logit to negative infinity explicitly on non-match beams
                zeros[i, eos_token_id] = float("-inf")

        return zeros

    return _constrain


def allow_tokens(allowed_tokens: list[int]) -> LogitFn:
    assert len(allowed_tokens) > 0, "allowed_tokens must be non-empty"
    allowed = torch.tensor(allowed_tokens, dtype=torch.long)

    def _allow_tokens(
        logits: torch.Tensor,
        _beams: list[Beam],
    ) -> torch.Tensor:
        zeros = torch.full_like(logits, float("-inf"))
        zeros[:, allowed] = logits[:, allowed]
        return zeros

    return _allow_tokens


def identity_update() -> UpdateFn:
    def _update_fn(beam: Beam) -> Beam:
        return beam

    return _update_fn


def sample() -> SampleFn:
    def _sample(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert logits.ndim == 2, "expected logits to be 2D"
        k = min(k, logits.shape[-1])
        probs = torch.softmax(logits, dim=-1)
        indices = torch.multinomial(probs, k)
        return indices, logits.gather(-1, indices)

    return _sample


def greedy() -> SampleFn:
    def _greedy(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert logits.ndim == 2, "expected logits to be 2D"
        k = min(k, logits.shape[-1])
        top_k = torch.topk(logits, k)
        return top_k.indices, top_k.values

    return _greedy


def repeat_penalty(penalty: float) -> LogitFn:
    assert penalty > 0.0, "penalty must be positive"

    def _repeat_penalty(
        logits: torch.Tensor,
        beams: list[Beam],
    ) -> torch.Tensor:
        logits = logits.clone()
        for logit, beam in zip(logits, beams):
            mask = torch.tensor(beam.token_ids, device=logit.device)
            seen = torch.gather(logit, -1, mask)
            seen = torch.where(seen > 0, seen / penalty, seen * penalty)
            logit.scatter_(-1, mask, seen)
        return logits

    return _repeat_penalty


def temperature_scaling(temp: float) -> LogitFn:
    assert temp > 0.0, "temperature must be positive"

    def _temperature_scaling(
        logits: torch.Tensor,
        _beams: list[Beam],
    ) -> torch.Tensor:
        return logits / temp

    return _temperature_scaling


def top_k_masking(k: int) -> LogitFn:
    assert k > 0, "k must be positive"

    def _top_k(
        logits: torch.Tensor,
        _beams: list[Beam],
    ) -> torch.Tensor:
        topk = torch.full_like(logits, float("-inf"))
        values, indices = torch.topk(logits, min(k, logits.shape[-1]), dim=-1)
        topk.scatter_(-1, indices, values)
        return topk

    return _top_k


def nucleus_masking(p: float, keep_min: int = 1) -> LogitFn:
    assert 0.0 <= p <= 1.0, "p must be in [0, 1]"
    assert keep_min > 0, "keep_min must be positive"

    def _nuc(
        logits: torch.Tensor,
        _: list[Beam],
    ) -> torch.Tensor:
        keep = min(keep_min, logits.shape[-1])
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
        nucleus = cum_sum_probs < p
        nucleus = torch.cat(
            [nucleus.new_ones((len(nucleus), keep)), nucleus[:, :-keep]], dim=-1
        )
        sorted_logits = torch.gather(logits, -1, sorted_indices)
        sorted_logits[torch.logical_not(nucleus)] = float("-inf")
        return sorted_logits.gather(-1, sorted_indices.argsort(-1))

    return _nuc


def min_p_masking(min_p: float, keep_min: int = 1) -> LogitFn:
    assert 0.0 <= min_p <= 1.0, "min_p must be in [0, 1]"
    assert keep_min > 0, "keep_min must be positive"

    def _min_p(
        logits: torch.Tensor,
        _: list[Beam],
    ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1, keepdim=True)[0]

        mask = probs < max_probs * min_p

        sorted_indices = torch.argsort(probs, descending=True, dim=-1)
        sorted_indices_mask = torch.gather(mask, dim=-1, index=sorted_indices)
        sorted_indices_mask[..., :keep_min] = False

        indices_mask = sorted_indices_mask.scatter(
            -1, sorted_indices, sorted_indices_mask
        )
        return logits.masked_fill(indices_mask, float("-inf"))

    return _min_p
