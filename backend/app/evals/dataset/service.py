"""Checking the dataset against the corpus, and stamping it with what the corpus now says."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import utc_today
from app.evals.dataset.models import (
    STAMP_LENGTH,
    CaseReference,
    ChangedDocument,
    CorpusStamp,
    DatasetDrift,
    EvalCase,
    EvalDataset,
    StaleReference,
    UnresolvedReference,
)
from app.ingestion.fetch.models import RawDocsQuery
from app.ingestion.fetch.service import get_raw_documents
from app.ingestion.service import get_latest_corpus_version
from app.retrieval.follow import division_content_hashes, reference_exists


async def find_unresolved_references(
    session: AsyncSession, dataset: EvalDataset
) -> tuple[UnresolvedReference, ...]:
    """Every case reference with no stored chunk for its celex + article/annex, with its case id.
    Hard drift: a renumbered re-ingest or a typo, which a run would score as a retrieval miss."""
    unresolved = []
    for case in dataset.cases:
        for target in case.references:
            if not await reference_exists(session, target):
                unresolved.append(UnresolvedReference(case_id=case.id, target=target))
    return tuple(unresolved)


async def _current_hashes(session: AsyncSession, target: CaseReference) -> tuple[str, ...]:
    """What the cited division hashes to now, stamped the length a case stores."""
    hashes = await division_content_hashes(session, target)
    return tuple(digest[:STAMP_LENGTH] for digest in hashes)


async def find_stale_references(
    session: AsyncSession, dataset: EvalDataset
) -> tuple[StaleReference, ...]:
    """Every stamped reference whose cited text has changed since the case was authored.

    Soft drift: the division still resolves and still retrieves, so nothing else catches it,
    but the case's answer was written against text that has since moved. An unstamped
    reference is passed over — nothing was recorded to compare against.
    """
    stale = []
    for case in dataset.cases:
        for target in case.references:
            if not target.content_hashes:
                continue
            current = await _current_hashes(session, target)
            if current != target.content_hashes:
                stale.append(
                    StaleReference(
                        case_id=case.id,
                        target=target,
                        stamped=target.content_hashes,
                        current=current,
                    )
                )
    return tuple(stale)


async def find_changed_documents(
    session: AsyncSession, dataset: EvalDataset
) -> tuple[ChangedDocument, ...]:
    """Every stamped act whose bytes differ from the standing corpus. Coarser than a stale
    reference and reported as its own line: it says the ground moved, not that a case is wrong."""
    if dataset.corpus is None:
        return ()
    documents = await get_raw_documents(session, RawDocsQuery())
    current = {celex: document.sha256 for celex, document in documents.items()}
    return tuple(
        ChangedDocument(celex=celex, stamped=stamped, current=current.get(celex))
        for celex, stamped in sorted(dataset.corpus.documents.items())
        if current.get(celex) != stamped
    )


async def _stamp_case(session: AsyncSession, case: EvalCase) -> EvalCase:
    """The case with every reference carrying what its division hashes to now."""
    references = [
        reference.model_copy(update={"content_hashes": await _current_hashes(session, reference)})
        for reference in case.references
    ]
    return case.model_copy(update={"references": tuple(references)})


async def stamp_dataset(session: AsyncSession, dataset: EvalDataset) -> EvalDataset:
    """The dataset restamped against the corpus as it stands.

    Only the selected cases are stamped, so repairing one case cannot silently clear the
    staleness of the others. The corpus block covers the whole dataset, so it is rewritten
    only by an unfiltered stamp, which is the one that can honestly claim to.
    """
    selected = {case.id for case in dataset.selected_cases}
    cases = tuple(
        [
            await _stamp_case(session, case) if case.id in selected else case
            for case in dataset.cases
        ]
    )
    corpus = dataset.corpus if dataset.case_filter else await _stamp_corpus(session, cases)
    return dataset.model_copy(update={"cases": cases, "corpus": corpus})


async def _stamp_corpus(session: AsyncSession, cases: tuple[EvalCase, ...]) -> CorpusStamp:
    """The corpus block: the version the acts the cases cite were last ingested at."""
    cited = {reference.celex for case in cases for reference in case.references}
    documents = await get_raw_documents(session, RawDocsQuery())
    return CorpusStamp(
        corpus_version=await get_latest_corpus_version(session),
        stamped_at=utc_today(),
        documents={
            celex: document.sha256
            for celex, document in sorted(documents.items())
            if celex in cited
        },
    )


async def inspect_dataset(session: AsyncSession, dataset: EvalDataset) -> DatasetDrift:
    """Every drift signal the dataset carries, gathered in one pass for `check` and `run`."""
    return DatasetDrift(
        unresolved=await find_unresolved_references(session, dataset),
        stale=await find_stale_references(session, dataset),
        changed_documents=await find_changed_documents(session, dataset),
        unstamped=dataset.unstamped_cases,
    )
