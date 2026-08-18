"""Eval checks against the corpus."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.evals.models import EvalDataset, UnresolvedReference
from app.retrieval.follow import reference_exists


async def find_unresolved_references(
    session: AsyncSession, dataset: EvalDataset
) -> tuple[UnresolvedReference, ...]:
    """Look up every case's references in the corpus and return those that find no stored chunk.

    A reference resolves when a chunk is stored for its celex + article/annex. It stops resolving
    when the act is re-ingested renumbered or the case was authored with a typo; a run would then
    score that case as a retrieval miss for the wrong reason, so this check runs first.
    """
    unresolved = []
    for case in dataset.cases:
        for target in case.references:
            if not await reference_exists(session, target):
                unresolved.append(UnresolvedReference(case_id=case.id, target=target))
    return tuple(unresolved)
