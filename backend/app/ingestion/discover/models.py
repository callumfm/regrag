"""What discovery found: one candidate act, and the document it selects.

An *act* is a regulation, directive or decision — the EU's collective word for a piece of
legislation. Discovery reads acts out of CELLAR; the rest of the pipeline handles documents.
"""

from app.core.models import FrozenModel


class CandidateAct(FrozenModel):
    """One act the topic query returned, before select.py decides whether it is worth fetching."""

    celex: str
    in_force: str | None = None
    consolidations: frozenset[str] = frozenset()
    """Every consolidated text including this act: as its base act, or as an amendment folded in."""


class DiscoveredDocument(FrozenModel):
    """One document discovery found: what to fetch, and which version to try first."""

    topic: str
    source: str
    celex: str
    candidate_celex: str | None
