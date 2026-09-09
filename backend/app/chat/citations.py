"""Citation markers: the [n] an answer cites a numbered context block by, and how an answer
is read back against the blocks it was given."""

import re
from collections.abc import Sequence

MARKER = re.compile(r"\[(\d+)\]")
"""A citation marker as the system prompt asks for it, like [1] or the [2][3] of a pair."""


def strip_markers(text: str) -> str:
    """The text without its citation markers: an earlier answer carried into a later turn
    numbered blocks that turn will not have."""
    return MARKER.sub("", text)


def find_cited_markers(answer: str) -> tuple[int, ...]:
    """The distinct [n] markers the answer leans on, in the order it first cites them.
    Distinct, so citing one block repeatedly does not weight it by how often it is named."""
    seen = dict.fromkeys(int(match) for match in MARKER.findall(answer))
    return tuple(seen)


def find_cited_sources[T](answer: str, sources: Sequence[T]) -> tuple[tuple[int, T], ...]:
    """Each marker the answer cites paired with the block it addresses, in cited order; a
    marker addressing no block is left out."""
    return tuple(
        (marker, sources[marker - 1])
        for marker in find_cited_markers(answer)
        if 1 <= marker <= len(sources)
    )
