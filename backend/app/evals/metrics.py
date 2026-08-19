"""Eval scoring: what counts as a retrieved reference and a grounded citation."""

import re
from collections.abc import Sequence

from app.retrieval.models import ReferenceTarget, RetrievedChunk

MARKER = re.compile(r"\[(\d+)\]")
"""A citation marker as the system prompt asks for it, like [1] or the [2][3] of a pair."""


def _division(item: ReferenceTarget | RetrievedChunk) -> tuple[str, str | None, str | None]:
    """The act and division a reference names, keyed as `follow` matches it: article
    case-folded, annex verbatim, and "" an unnumbered annex rather than none."""
    article = item.article.lower() if item.article is not None else None
    return item.celex, article, item.annex


def score_reference_recall(
    targets: Sequence[ReferenceTarget], chunks: Sequence[RetrievedChunk]
) -> float:
    """Share of a case's authored references some retrieved chunk covers."""
    if not targets:
        return 0.0
    retrieved = {_division(chunk) for chunk in chunks}
    return sum(_division(target) in retrieved for target in targets) / len(targets)


def find_cited_markers(answer: str) -> tuple[int, ...]:
    """The distinct [n] markers the answer leans on, in the order it first cites them.
    Distinct, so citing one block repeatedly does not weight it by how often it is named."""
    seen = dict.fromkeys(int(match) for match in MARKER.findall(answer))
    return tuple(seen)


def score_citation_validity(answer: str, sources: Sequence[RetrievedChunk]) -> float | None:
    """Share of the answer's markers addressing a block it was given; None when it cited
    nothing, which is unmeasured rather than zero."""
    markers = find_cited_markers(answer)
    if not markers:
        return None
    return sum(1 <= marker <= len(sources) for marker in markers) / len(markers)


def score_reference_citation_rate(
    answer: str,
    sources: Sequence[RetrievedChunk],
    targets: Sequence[ReferenceTarget],
) -> float | None:
    """Share of a case's authored references the answer cited, scored over the references
    so an extra citation is not an error; None when the case names none."""
    if not targets:
        return None
    cited = {
        _division(sources[marker - 1])
        for marker in find_cited_markers(answer)
        if 1 <= marker <= len(sources)
    }
    return sum(_division(target) in cited for target in targets) / len(targets)
