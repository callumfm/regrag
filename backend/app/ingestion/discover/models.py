"""What discovery found: one candidate act, and the document it selects.

An *act* is a regulation, directive or decision — the EU's collective word for a piece of
legislation. Discovery reads acts out of CELLAR; the rest of the pipeline handles documents.
"""

from app.core.models import FrozenModel


class ActsQueryRow(FrozenModel):
    """One line of CELLAR's answer: an act, and one consolidated text including it.

    An act with no consolidations still gets a line, with the consolidation empty; anything that
    is not law carries no in-force flag at all.
    """

    celex: str
    in_force: bool | None = None
    consolidation: str | None = None


class CandidateAct(FrozenModel):
    """One act the topic query returned, before select.py decides whether it is worth fetching."""

    celex: str
    in_force: bool | None = None
    consolidations: frozenset[str] = frozenset()


class DiscoveredDocument(FrozenModel):
    """One document discovery found: what to fetch, and which version to try first."""

    topic: str
    source: str
    celex: str
    candidate_celex: str | None
