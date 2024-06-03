from text_utils._internal import grammar
from text_utils._internal import continuations


# re-export grammar constraints
RegexConstraint = grammar.RegexConstraint
LR1Constraint = grammar.LR1Constraint


class Constraint:
    """
    Base class for constraints.
    """

    def get(self) -> tuple[list[int], bool]:
        """
        Returns the current constraint indices and whether we
        are in a state that matches the constraint.
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
        cont_index: continuations.ContinuationIndex,
        prefix: bytes | None = None
    ):
        self.prefix = prefix or bytes()
        self.value = cont_index.get_value(self.prefix)
        self.cont_index = cont_index

    def get(self) -> tuple[list[int], bool]:
        indices, value = self.cont_index.get(self.prefix)
        self.value = value
        return indices, self.is_match()

    def reset(self, input: bytes | None = None) -> None:
        self.prefix = input or bytes()

    def next(self, index: int) -> None:
        self.prefix += self.cont_index.get_continuation(index)

    def is_match(self) -> bool:
        return self.value is not None

    def clone(self) -> 'ContinuationConstraint':
        return ContinuationConstraint(
            self.cont_index,
            self.prefix
        )

    def get_value(self) -> str:
        return self.value
