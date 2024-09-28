from functools import reduce
import numpy as np

from text_utils._internal import grammar
from text_utils._internal import continuations


# re-export grammar constraints
RegexConstraint = grammar.RegexConstraint
LR1Constraint = grammar.LR1Constraint


class Constraint:
    """
    Base class for constraints.
    """

    def get(self) -> np.ndarray:
        """
        Returns the current constraint indices.
        """
        raise NotImplementedError

    def reset(self, input: bytes | None = None) -> None:
        """
        Resets the constraint to the initial state.
        """
        raise NotImplementedError

    def next(self, index: int) -> None:
        """
        Updates the constraint based on the chosen index / token id.
        """
        raise NotImplementedError

    def is_match(self) -> bool:
        """
        Returns whether the current state matches the constraint.
        """
        raise NotImplementedError

    def is_invalid(self) -> bool:
        """
        Returns whether the current state is invalid.
        This must be true iff get() returns an empty list of indices in
        a non-match state.
        We have a separate function for that because depending on the constraint
        this can be implemented more efficiently.
        """
        return not self.is_match() and len(self.get()) == 0

    def clone(self) -> 'Constraint':
        """
        Returns a copy of the constraint.
        """
        raise NotImplementedError


class ContinuationConstraint(Constraint):
    """
    Constraint for only allowing certain continuations for
    a given prefix.
    """

    def __init__(
        self,
        cont_index: continuations.MmapContinuationIndex,
    ):
        self.prefix = bytes()
        self.indices, self.value = cont_index.get(self.prefix)
        self.cont_index = cont_index

    def get(self) -> np.ndarray:
        return self.indices

    def reset(self, input: bytes | None = None) -> None:
        self.prefix = input or bytes()
        self.indices, self.value = self.cont_index.get(self.prefix)

    def next(self, index: int) -> None:
        self.prefix += bytes(self.cont_index.continuation_at(index))
        self.indices, self.value = self.cont_index.get(self.prefix)

    def is_match(self) -> bool:
        return self.value is not None

    def clone(self) -> 'ContinuationConstraint':
        const = ContinuationConstraint(self.cont_index)
        const.reset(self.prefix)
        return const

    def get_value(self) -> str | None:
        return self.value


def array_intersection(*arrays: np.ndarray) -> np.ndarray:
    """
    Returns the intersection of multiple arrays.
    """
    assert len(arrays) > 0, "at least one array required"
    return reduce(np.intersect1d, arrays)


class AndConstraint(Constraint):
    def __init__(self, constraints: list[Constraint]):
        assert len(constraints) > 0, "at least one constraint required"
        self.constraints = constraints

    def get(self) -> np.ndarray:
        return array_intersection(*(c.get() for c in self.constraints))

    def reset(self, input: bytes | None = None) -> None:
        for c in self.constraints:
            c.reset(input)

    def next(self, index: int) -> None:
        for c in self.constraints:
            c.next(index)

    def is_match(self) -> bool:
        return all(c.is_match() for c in self.constraints)

    def clone(self) -> 'AndConstraint':
        return AndConstraint([c.clone() for c in self.constraints])


class AvoidConstraint(Constraint):
    def __init__(
        self,
        avoid: list[bytes],
        continuations: list[bytes],
        eos_token_id: int
    ):
        self.avoid = avoid
        self.value = bytes()
        self.continuations = continuations
        self.eos_token_id = eos_token_id
        self.all = np.arange(len(continuations))
        self.all_but_eos = np.delete(self.all, self.eos_token_id)

    def get(self) -> np.ndarray:
        return self.all if self.is_match() else self.all_but_eos

    def reset(self, input: bytes | None = None) -> None:
        self.value = input or bytes()

    def next(self, index: int) -> None:
        self.value += self.continuations[index]

    def is_match(self) -> bool:
        return all(avoid != self.value for avoid in self.avoid)

    def clone(self) -> 'AvoidConstraint':
        const = AvoidConstraint(
            self.avoid,
            self.continuations,
            self.eos_token_id
        )
        const.reset(self.value)
        return const
