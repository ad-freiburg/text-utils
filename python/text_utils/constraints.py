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
        prefix: bytes | None = None
    ):
        self.prefix = prefix or bytes()
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
        return ContinuationConstraint(
            self.cont_index,
            self.prefix
        )

    def get_value(self) -> str | None:
        return self.value
