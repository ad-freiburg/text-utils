import itertools
import random
import re
from typing import Union, List, Tuple


def clean(sequence: str) -> str:
    # about 5 times faster than re.sub("\s+", " ", sequence)
    return " ".join(sequence.strip().split())


def word_boundaries(s: str) -> List[Tuple[int, int]]:
    # this function assumes that s is cleaned with clean_sequence above
    # assert s == clean(s), "clean the input sequence before applying this function"

    boundaries = []
    start_idx = 0
    for word in s.split():
        end_idx = start_idx + len(word)
        boundaries.append((start_idx, end_idx))
        start_idx = end_idx + 1
    return boundaries


def possible_character_subsequences(s: str, max_chars: int) -> List[Tuple[int, int]]:
    return [(i, max(len(s), i + max_chars)) for i in range(max(1, len(s) - max_chars + 1))]


def random_character_susequence(
        s: str,
        max_chars: int,
        rand: random.Random
) -> Tuple[int, int]:
    possible_substrings = possible_character_subsequences(s, max_chars)
    return possible_substrings[rand.randint(0, len(possible_substrings) - 1)]


def _find_subsequences_with_sum_close_to_but_max_k(
        values: List[int],
        k: int
) -> List[Tuple[int, int]]:
    if len(values) == 0:
        return [(0, 0)]
    # this is linear
    cum_values = list(itertools.accumulate(values))
    if cum_values[-1] <= k:
        return [(0, len(cum_values))]
    start = 0
    # move start pointer to first valid start position (element smaller or equal to k)
    while start < len(values) and values[start] > k:
        start += 1
    if start >= len(values):
        return []
    end = start
    subsequences = []
    while start < len(cum_values) and end < len(cum_values):
        next_end_v = values[end + 1] if end + 1 < len(cum_values) else 0
        if next_end_v > k:
            subsequences.append((start, end + 1))
            start = end + 2
            end = start
        else:
            cum_next_end_v = cum_values[end] + next_end_v
            cum_up_to_start = cum_values[start] - values[start]
            if cum_next_end_v - cum_up_to_start > k:
                if len(subsequences) == 0 or subsequences[-1][1] < end + 1:
                    subsequences.append((start, end + 1))
                start += 1
            else:
                end += 1
    if start != end:
        subsequences.append((start, end))
    return subsequences


def possible_byte_subsequences(s: str, max_bytes: int) -> List[Tuple[int, int]]:
    num_bytes = list(len(c.encode("utf8")) for c in s)
    return _find_subsequences_with_sum_close_to_but_max_k(num_bytes, max_bytes)


def random_byte_subsequence(
        s: str,
        max_bytes: int,
        rand: random.Random
) -> Tuple[int, int]:
    possible_subsequences = possible_byte_subsequences(s, max_bytes)
    return possible_subsequences[rand.randint(0, len(possible_subsequences) - 1)]


def natural_sort(unsorted: List[str], reverse: bool = False) -> List[str]:
    def _convert(s: str) -> Union[str, int]:
        return int(s) if s.isdigit() else s.lower()

    def _alphanum_key(key: str) -> List[Union[int, str]]:
        return [_convert(c) for c in re.split(r"([0-9]+)", key)]

    return sorted(unsorted, key=_alphanum_key, reverse=reverse)
