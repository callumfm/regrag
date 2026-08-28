"""Dataset drift: how far it has come adrift of the corpus it was authored against."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.session import get_session
from app.evals.dataset.enums import DriftKind
from app.evals.dataset.models import CaseReference, DriftedReference, EvalDataset
from app.evals.dataset.stamp import current_stamp
from app.ingestion.service import get_latest_corpus_version


async def find_drift(session: AsyncSession, dataset: EvalDataset) -> tuple[DriftedReference, ...]:
    """Every case reference the corpus has moved out from under, by how it moved."""
    drifted = []
    for case in dataset.cases:
        for target in case.references:
            current = await current_stamp(session, target)
            kind = _classify(target, current)
            if kind is not None:
                drifted.append(DriftedReference(case_id=case.id, target=target, kind=kind))
    return tuple(drifted)


def _classify(target: CaseReference, current: tuple[str, ...]) -> DriftKind | None:
    if not current:
        return DriftKind.UNRESOLVED
    if not target.content_hashes:
        return DriftKind.UNSTAMPED
    return DriftKind.STALE if current != target.content_hashes else None


async def check_against_corpus(
    dataset: EvalDataset,
) -> tuple[tuple[DriftedReference, ...], str | None]:
    """Both reads in one session: every drifted reference, and the corpus version the
    store currently stands at."""
    async with get_session(auto_commit=False) as session:
        return await find_drift(session, dataset), await get_latest_corpus_version(session)


def find_moved_corpus(dataset: EvalDataset, current: str | None) -> str | None:
    """The current corpus version, when the dataset was stamped against a different one."""
    if dataset.corpus is None:
        return None
    return current if current != dataset.corpus.corpus_version else None


def stale_case_ids(drifted: tuple[DriftedReference, ...]) -> tuple[str, ...]:
    """The stale cases named once each, however many of their references moved."""
    stale = (item.case_id for item in drifted if item.kind is DriftKind.STALE)
    return tuple(dict.fromkeys(stale))


HEADINGS = {
    DriftKind.UNRESOLVED: "unresolved (no stored chunk answers to it):",
    DriftKind.STALE: "stale (cited text changed since authoring):",
    DriftKind.UNSTAMPED: "unstamped (nothing recorded to compare against):",
}


def format_drift(drifted: tuple[DriftedReference, ...], moved_to: str | None = None) -> list[str]:
    """Each kind under its own heading, worst first, the case column sized to its block."""
    lines = []
    for kind, heading in HEADINGS.items():
        block = [item for item in drifted if item.kind is kind]
        if not block:
            continue
        width = max(len(item.case_id) for item in block)
        lines.append(heading)
        lines += [
            f"  {item.case_id:<{width}}  {item.target.celex} {item.target.citation}"
            for item in block
        ]
    if moved_to:
        lines.append(f"corpus moved since stamping (now {moved_to})")
    return lines
