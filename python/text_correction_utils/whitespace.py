import random
import re
import itertools
from typing import List, Tuple


def remove(sequence: str) -> str:
    return "".join(sequence.split())


def operations(from_sequence: str, to_sequence: str) -> List[int]:
    """

    Get the whitespace operations that turns from_sequence into to_sequence (after applying the repair function)

    :param from_sequence: sequence that the returned whitespace operations should be applied to to get the to_sequence
    :param to_sequence: sequence that should result from applying the whitespace operations to the from_sequence
    :return: list of whitespace operations
    """
    assert remove(from_sequence) == remove(to_sequence), \
        f"make sure from_sequence and to_sequence only differ in whitespaces:\n{from_sequence}\n{to_sequence}"

    from_sequence_ptr = 0
    to_sequence_ptr = 0

    repair_tokens = []

    while from_sequence_ptr < len(from_sequence):  # and to_sequence_ptr < len(to_sequence):
        from_char = from_sequence[from_sequence_ptr]
        to_char = to_sequence[to_sequence_ptr] if to_sequence_ptr < len(to_sequence) else ""

        if from_char == to_char:
            repair_tokens.append(0)
            from_sequence_ptr += 1
            to_sequence_ptr += 1

        elif to_char == " ":
            repair_tokens.append(1)
            from_sequence_ptr += 1
            to_sequence_ptr += 2

        elif from_char == " ":
            repair_tokens.append(2)
            from_sequence_ptr += 1

        else:
            raise ValueError("should not happen")

    return repair_tokens


def repair(sequence: str, operations: List[int]) -> str:
    """

    Repair the white spacing in the given sequence using the given whitespace operations.

    :param sequence: string which has to be repaired
    :param operations: list with 0's, 1's and 2's indicating to keep the char, insert a whitespace
        or delete a whitespace.
    :return: repaired string
    """
    if len(sequence) > len(operations):
        operations.extend([0] * (len(sequence) - len(operations)))
    else:
        operations = operations[:len(sequence)]

    allowed_ops = {0, 1, 2}
    assert all(op in allowed_ops for op in operations), \
        f"only 0's, 1's and 2's are allowed as whitespace operations, but got {operations} for sequence \"{sequence}\""

    sequence_ptr = 0
    op_ptr = 0

    repaired_sequence = ""
    while sequence_ptr < len(sequence):
        char = sequence[sequence_ptr]
        prev_char = sequence[sequence_ptr - 1] if sequence_ptr > 0 else ""
        token = operations[op_ptr]

        if token == 1 and char != " " and prev_char != " ":
            # if we should insert a whitespace and the current and previous character are not whitespaces,
            # add a whitespace in front of the character
            repaired_sequence += " " + char

        elif token == 2 and char == " ":
            # if we should delete a whitespace and we are at a whitespace, just skip
            pass

        else:
            # keep current character in all other cases
            repaired_sequence += char

        sequence_ptr += 1
        op_ptr += 1

    return repaired_sequence


def find_substring_ignoring_spaces(s: str, substring: str) -> Tuple[int, int]:
    substring_pattern = re.compile(r"\s?".join(
        re.escape(char) for char in substring if char != " "
    ))
    match = substring_pattern.search(s)
    assert match is not None, f"could not find substring \"{substring}\" in \"{s}\""
    return match.start(), match.end()
