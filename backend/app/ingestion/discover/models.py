"""What discovery found: one candidate act, and the document it selects."""

from app.core.models import FrozenModel


class CandidateAct(FrozenModel):
    """One act the topic query returned, with every binding for it folded together."""

    celex: str
    in_force: str | None = None
    consolidations: frozenset[str] = frozenset()


class DiscoveredDocument(FrozenModel):
    """One document discovery found: what to fetch, and which version to try first."""

    topic: str
    source: str
    celex: str
    candidate_celex: str | None
