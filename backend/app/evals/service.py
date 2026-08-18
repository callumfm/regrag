"""Eval checks against the corpus."""

from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.evals.models import EvalCase
from app.retrieval.follow import follow_reference
from app.retrieval.models import ReferenceTarget


async def unresolved_references(
    session: AsyncSession, cases: Sequence[EvalCase]
) -> tuple[tuple[str, ReferenceTarget], ...]:
    """Every case reference no stored chunk answers to, with its case id; empty when the dataset
    still matches the corpus. A reference goes stale when an act is re-ingested renumbered."""
    unresolved = []
    for case in cases:
        for target in case.references:
            if not await follow_reference(session, target):
                unresolved.append((case.id, target))
    return tuple(unresolved)
