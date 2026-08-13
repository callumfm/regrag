"""The policy over what the topic query returned: which acts to fetch, and at what version."""

from collections.abc import Iterable

from app.ingestion import celex
from app.ingestion.discover.models import CandidateAct, DiscoveredDocument

IN_FORCE_FLAG = "1"


def _extract_consolidations_of_this_act(act: CandidateAct) -> set[str]:
    """The consolidated texts this act is the base of, rather than an amendment folded into."""
    stem = celex.consolidated_stem(act.celex)
    return {version for version in act.consolidations if version.startswith(stem)}


def _is_in_force(act: CandidateAct) -> bool:
    """CELLAR flags a live act with '1'; repealed and unstated acts are not fetched."""
    return act.in_force == IN_FORCE_FLAG


def _is_folded_into_another_act(act: CandidateAct) -> bool:
    """Every consolidation of this act belongs to another act, which now supersedes it."""
    return bool(act.consolidations) and not _extract_consolidations_of_this_act(act)


def _latest_consolidation(act: CandidateAct) -> str | None:
    """The newest consolidated text of this act, or None if it has never been consolidated."""
    own = _extract_consolidations_of_this_act(act)
    return max(own) if own else None


def select_topic_documents(topic: str, acts: Iterable[CandidateAct]) -> list[DiscoveredDocument]:
    """The acts worth fetching, each pointed at the version to try first."""
    return [
        DiscoveredDocument(
            topic=topic,
            source="eurlex",
            celex=act.celex,
            candidate_celex=_latest_consolidation(act),
        )
        for act in acts
        if _is_in_force(act) and not _is_folded_into_another_act(act)
    ]
