"""Eval checks against the corpus."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.evals.models import EvalDataset, UnresolvedReference
from app.retrieval.follow import reference_exists


async def find_unresolved_references(
    session: AsyncSession, dataset: EvalDataset
) -> tuple[UnresolvedReference, ...]:
    """Every case reference with no stored chunk for its celex + article/annex, with its case id.
    Stale after a renumbered re-ingest or a typo; a run would score it as a retrieval miss."""
    unresolved = []
    for case in dataset.cases:
        for target in case.references:
            if not await reference_exists(session, target):
                unresolved.append(UnresolvedReference(case_id=case.id, target=target))
    return tuple(unresolved)
