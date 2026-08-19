"""Eval scoring: what counts as a retrieved reference, a grounded citation, a refusal."""

import re
from collections.abc import Sequence

from app.chat.prompts import REFUSAL_ANSWER
from app.retrieval.models import ReferenceTarget, RetrievedChunk

MARKER = re.compile(r"\[(\d+)\]")
"""A citation marker as the system prompt asks for it, like [1] or the [2][3] of a pair."""


def _division(item: ReferenceTarget | RetrievedChunk) -> tuple[str, str | None, str | None]:
    """Which division of which act, at the grain an authored reference names: article or
    annex, never the paragraph inside one.

    Matched as `follow` matches it — article case-folded so 4a reaches 4A, annex compared
    exactly — so `check` and `run` never disagree about the same reference. An annex of ""
    is the act's one unnumbered annex, which no truthiness test may fold into no annex.
    """
    article = item.article.lower() if item.article is not None else None
    return item.celex, article, item.annex


def score_reference_recall(
    targets: Sequence[ReferenceTarget], chunks: Sequence[RetrievedChunk]
) -> float:
    """Share of a case's authored references some retrieved chunk covers.

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


def score_citation_validity(answer: str, sources: Sequence[RetrievedChunk]) -> float | None:
    """Share of the markers the answer cites that address a block it was actually given;
    None when it cited nothing, which is unmeasured rather than zero.

    A marker past the end of the context is the one citation fault this can judge without a
    model: the answer points at a block that was never in its prompt.
    """
    markers = find_cited_markers(answer)
    if not markers:
        return None
    return sum(1 <= marker <= len(sources) for marker in markers) / len(markers)


def score_reference_citation_rate(
    answer: str,
    sources: Sequence[RetrievedChunk],
    targets: Sequence[ReferenceTarget],
) -> float | None:
    """Share of a case's authored references the answer actually cited; None when the case
    names none, which is unmeasured rather than zero.

    Scored over the references rather than over the markers, because the authored set names
    the references an answer must lean on, not every chunk that may legitimately support it:
    citing a further relevant article is not an error, and must not read as one.
    """
    if not targets:
        return None
    cited = {
        _division(sources[marker - 1])
        for marker in find_cited_markers(answer)
        if 1 <= marker <= len(sources)
    }
    return sum(_division(target) in cited for target in targets) / len(targets)


def is_gate_refusal(answer: str) -> bool:
    """Whether the score gate refused before any model call, which the fixed wording marks.

    A model that declines in its own words has not gate-refused; judging such an answer
    needs a model of its own, which is the judge's job rather than this runner's.
    """
    return answer.strip() == REFUSAL_ANSWER
