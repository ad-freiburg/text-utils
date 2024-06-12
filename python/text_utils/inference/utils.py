from typing import Callable, Any

import torch

from text_utils.constraints import Constraint

# maps from token ids, length, and other kwargs to distribution over next token id and other info
DecodeFn = Callable[..., tuple[torch.Tensor, dict[str, Any]]]

# select specific elements for all the kwargs keys given the mask tensor
MaskSelectFn = Callable[
    [dict[str, Any], torch.Tensor],
    dict[str, Any]
]

# update specific elements for all the kwargs keys given the mask tensor
MaskUpdateFn = Callable[
    [dict[str, Any], dict[str, Any], torch.Tensor],
    None
]


class Beam:
    def __init__(
        self,
        token_ids: list[int],
        log_probs: list[float],
        info: dict[str, Any] | None = None
    ) -> None:
        self.token_ids = token_ids
        self.log_probs = log_probs
        self.info: dict[str, Any] = info or {}

    @staticmethod
    def from_beam(
        other: "Beam",
        token_id: int,
        log_p: float
    ) -> "Beam":
        return Beam(
            other.token_ids + [token_id],
            other.log_probs + [log_p],
            {k: v for k, v in other.info.items()}
        )

    def truncate_prefix(
        self,
        length: int
    ) -> "Beam":
        return Beam(
            self.token_ids[length:],
            self.log_probs[length:],
            {k: v for k, v in self.info.items()}
        )

    @property
    def log_prob(self) -> float:
        return sum(self.log_probs)

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
        # indices of the input elements currently being processed
        list[int] | list[Beam]
    ],
    # new logits, shape [batch_size, vocab_size]
    torch.Tensor,
]

# selects indices and scores from given token distributions
SampleFn = Callable[
    [
        # distributions over next token id, shape [batch_size, vocab_size]
        torch.Tensor,
        # indices of input batch elements which are sampled
        list[int]
    ],
    # indices of selected tokens, shape [batch_size]
    torch.Tensor,
]

# checks if decoding should be stopped
StopFn = Callable[
    [
        # token ids decoded so far
        torch.Tensor,
        # index of input batch element checked for stopping
        int
    ],
    # bool indicating if decoding should be stopped
    bool
]

# takes in log probs and beam width and returns
# beam width samples
BeamSampleFn = Callable[
    [
        # distribution over next tokens, shape [vocab_size]
        torch.Tensor,
        # beam width
        int
    ],
    # indices of selected tokens, shape [<= beam_width]
    torch.Tensor
]

# checks if beam should be stopped
BeamStopFn = Callable[
    [
        # beam checked for stopping
        Beam,
    ],
    # bool indicating if beam should be stopped
    bool
]

# takes in the beam, token id, and log prob,
# returns a new updated beam
# (having this as a separate function allows for state transfer
# between beams)
BeamCandidateFn = Callable[
    [
        Beam,
        int,
        float
    ],
    Beam
]


def constraint_logit_fn(
    retrieve_constraint_fn: Callable[[int | Beam], Constraint | None],
    eos_token_id: int
) -> LogitFn:
    import time

    times = []

    def _constrain_logits(
        logits: torch.Tensor,
        beams_or_indices:  list[int] | list[Beam]
    ) -> torch.Tensor:
        nonlocal times
        zeros = torch.full_like(logits, float("-inf"))

        for i, beam_or_idx in enumerate(beams_or_indices):
            constraint = retrieve_constraint_fn(beam_or_idx)

            if constraint is None:
                zeros[i] = logits[i]
                continue

            constrain_to, is_match = constraint.get()
            constrain_to = torch.from_numpy(constrain_to).to(torch.long)
            zeros[i, constrain_to] = logits[i, constrain_to]

            if len(constrain_to) == 0 or is_match:
                zeros[i, eos_token_id] = logits[i, eos_token_id]

        return zeros

    return _constrain_logits


def constraint_sample_fn(
    retrieve_constraint_fn: Callable[[int], Constraint | None],
    sample_fn: SampleFn,
    eos_token_id: int
) -> SampleFn:
    def _constrain_sample(
        logits: torch.Tensor,
        indices: list[int]
    ) -> torch.Tensor:
        token_ids = sample_fn(logits, indices)
        for idx, token_id in zip(indices, token_ids.tolist()):
            if token_id == eos_token_id:
                continue

            constraint = retrieve_constraint_fn(idx)
            if constraint is not None:
                constraint.next(token_id)

        return token_ids

    return _constrain_sample


def default_beam_candidate_fn() -> BeamCandidateFn:
    def _default_beam_candidate_fn(
        beam: Beam,
        token_id: int,
        log_prob: float
    ) -> Beam:
        return Beam.from_beam(beam, token_id, log_prob)

    return _default_beam_candidate_fn


def sample() -> SampleFn:
    def _sample(logits: torch.Tensor, _: list[int]) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    return _sample


def beam_sample() -> BeamSampleFn:
    def _beam_sample(logits: torch.Tensor, beam_width: int) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        beam_width = min(
            beam_width,
            probs.shape[-1],
            int(torch.sum(probs > 0).item())
        )
        return torch.multinomial(probs, beam_width)

    return _beam_sample


def greedy() -> SampleFn:
    def _greedy(logits: torch.Tensor, _: list[int]) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)

    return _greedy


def beam_greedy() -> BeamSampleFn:
    def _beam_greedy(logits: torch.Tensor, beam_width: int) -> torch.Tensor:
        beam_width = min(
            beam_width,
            logits.shape[-1] - int(torch.sum(torch.isinf(logits)).item())
        )
        return torch.topk(logits, beam_width, dim=-1).indices

    return _beam_greedy


def temperature_scaling(temp: float) -> LogitFn:
    def _temperature_scaling(logits: torch.Tensor, _: list[int] | list[Beam]) -> torch.Tensor:
        return logits / temp

    return _temperature_scaling


def top_k_masking(k: int) -> LogitFn:
    def _top_k(logits: torch.Tensor, _: list[int] | list[Beam]) -> torch.Tensor:
        topk = torch.full_like(logits, float("-inf"))
        values, indices = torch.topk(
            logits,
            min(k, logits.shape[-1]),
            dim=-1
        )
        topk.scatter_(-1, indices, values)
        return topk

    return _top_k


def nucleus_masking(p: float) -> LogitFn:
    def _nuc(logits: torch.Tensor, _: list[int] | list[Beam]) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
        cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
        nucleus = cum_sum_probs < p
        nucleus = torch.cat([
            nucleus.new_ones((len(nucleus), 1)),
            nucleus[:, :-1]
        ], dim=-1)
        sorted_logits = torch.gather(logits, -1, indices)
        sorted_logits[torch.logical_not(nucleus)] = float("-inf")
        return sorted_logits.gather(-1, indices.argsort(-1))

    return _nuc
