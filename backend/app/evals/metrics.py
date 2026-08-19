"""Eval scoring: what counts as a retrieved reference, a correct citation, a refusal."""

import re
from collections.abc import Sequence

from app.chat.prompts import REFUSAL_ANSWER
from app.retrieval.models import ReferenceTarget, RetrievedChunk

MARKER = re.compile(r"\[(\d+)\]")
"""A citation marker as the system prompt asks for it, like [1] or the [2][3] of a pair."""


def _division(item: ReferenceTarget | RetrievedChunk) -> tuple[str, str | None, str | None]:
    """Which division of which act, at the grain a gold reference names: article or annex,
    never the paragraph inside one. Cased as `follow` cases it, so 4a reaches 4A."""
    article = item.article.lower() if item.article else None
    annex = item.annex.lower() if item.annex else None
    return item.celex, article, annex


def score_reference_recall(
    targets: Sequence[ReferenceTarget], chunks: Sequence[RetrievedChunk]
) -> float:
    """Share of a case's gold references some retrieved chunk covers.

    Applied twice per case — to the raw search hits, then to the expanded sources — so the
    layer that found the reference is the layer that gets the credit.
    """
    if not targets:
        return 0.0
    retrieved = {_division(chunk) for chunk in chunks}
    return sum(_division(target) in retrieved for target in targets) / len(targets)


def find_cited_markers(answer: str) -> tuple[int, ...]:
    """The distinct [n] markers the answer leans on, in the order it first cites them.
    Distinct, so citing one block repeatedly does not weight it by how often it is named."""
    seen = dict.fromkeys(int(match) for match in MARKER.findall(answer))
    return tuple(seen)


def score_citation_precision(
    answer: str,
    sources: Sequence[RetrievedChunk],
    targets: Sequence[ReferenceTarget],
) -> float | None:
    """Share of the blocks the answer cited that a gold reference names; None when it cited
    nothing, which is unmeasured rather than zero. A marker past the end of the context is
    counted against the answer: it cites a block the model was never given."""
    markers = find_cited_markers(answer)
    if not markers:
        return None
    gold = {_division(target) for target in targets}
    correct = sum(
        1 <= marker <= len(sources) and _division(sources[marker - 1]) in gold for marker in markers
    )
    return correct / len(markers)


def is_gate_refusal(answer: str) -> bool:
    """Whether the score gate refused before any model call, which the fixed wording marks.

    A model that declines in its own words has not gate-refused; judging such an answer
    needs a model of its own, which is the judge's job rather than this runner's.
    """
    return answer.strip() == REFUSAL_ANSWER
