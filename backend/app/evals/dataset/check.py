"""`evals check`: how far the dataset has come adrift of the corpus it was authored against."""

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.session import get_session
from app.core.models import FrozenModel
from app.evals.dataset.enums import DriftKind
from app.evals.dataset.models import STAMP_LENGTH, CaseReference, EvalDataset
from app.ingestion.service import get_latest_corpus_version
from app.retrieval.follow import division_content_hashes


class DriftedReference(FrozenModel):
    """A case reference the corpus no longer answers to as it did when the case was authored."""

    case_id: str
    target: CaseReference
    kind: DriftKind


async def current_stamp(session: AsyncSession, target: CaseReference) -> tuple[str, ...]:
    """What the cited division hashes to now, cut to the length a stamp records."""
    hashes = await division_content_hashes(session, target)
    return tuple(digest[:STAMP_LENGTH] for digest in hashes)


async def find_drift(session: AsyncSession, dataset: EvalDataset) -> tuple[DriftedReference, ...]:
    """Every case reference the corpus has moved out from under, by how it moved.

    One read per reference answers all three: a division nothing covers is unresolved, one
    the case never recorded is unstamped, and one hashing to something other than what was
    recorded is stale — the amendment that still retrieves cleanly and so scores green.
    """
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


async def find_moved_corpus(session: AsyncSession, dataset: EvalDataset) -> str | None:
    """The corpus version now, when the dataset was stamped against a different one.

    Coarser than a drifted reference and reported as its own line: it says the ground moved,
    not that any case is wrong. It also separates the two ways a stamp can go out of date —
    a version that moved with it means the law changed, one that did not means we rechunked.
    """
    if dataset.corpus is None:
        return None
    current = await get_latest_corpus_version(session)
    return current if current != dataset.corpus.corpus_version else None


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


async def _inspect() -> tuple[tuple[DriftedReference, ...], str | None]:
    async with get_session(auto_commit=False) as session:
        dataset = EvalDataset.load()
        return await find_drift(session, dataset), await find_moved_corpus(session, dataset)


def check_dataset() -> int:
    """Report every way the dataset has drifted. Only an unresolved reference fails: a stale
    case needs a human re-reading the new text, which no build can do on its behalf."""
    drifted, moved_to = asyncio.run(_inspect())
    lines = format_drift(drifted, moved_to)
    print("\n".join(lines) if lines else "every case reference resolves and is stamped")
    return 1 if any(item.kind is DriftKind.UNRESOLVED for item in drifted) else 0
