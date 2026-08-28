"""`evals stamp`: record what the cited text says now, asserting it has been read."""

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import utc_today
from app.core.config import config
from app.evals.dataset.models import CaseReference, CorpusStamp, EvalCase, EvalDataset
from app.ingestion.service import get_latest_corpus_version
from app.retrieval.follow import division_content_hashes

STAMP_LENGTH = 12
"""How much of a chunk's content hash a stamp carries. Short enough to read in a diff, long
enough that two chunks of one corpus cannot collide."""


async def current_stamp(session: AsyncSession, target: CaseReference) -> tuple[str, ...]:
    """What the cited division hashes to now, cut to the length a stamp records."""
    hashes = await division_content_hashes(session, target)
    return tuple(digest[:STAMP_LENGTH] for digest in hashes)


async def _stamp_case(session: AsyncSession, case: EvalCase) -> EvalCase:
    """The case with every reference restamped to what its division hashes to now."""
    references = [
        reference.model_copy(update={"content_hashes": await current_stamp(session, reference)})
        for reference in case.references
    ]
    return case.model_copy(update={"references": tuple(references)})


async def stamp_dataset(session: AsyncSession, dataset: EvalDataset) -> EvalDataset:
    """Restamp only the selected cases; the corpus stamp covers the whole dataset, so it is
    rewritten only by an unfiltered stamp."""
    selected = {case.id for case in dataset.selected_cases}
    cases = [
        await _stamp_case(session, case) if case.id in selected else case for case in dataset.cases
    ]
    if dataset.case_filter:
        corpus = dataset.corpus
    else:
        version = await get_latest_corpus_version(session)
        corpus = CorpusStamp(corpus_version=version, stamped_at=utc_today())

    return dataset.model_copy(update={"cases": tuple(cases), "corpus": corpus})


def save_dataset(dataset: EvalDataset, path: Path = config.EVAL_DATASET_PATH) -> None:
    """Write the dataset back out, without the per-run filter or anything a case left unset."""
    rendered = dataset.model_dump_json(indent=2, exclude={"case_filter"}, exclude_defaults=True)
    path.write_text(rendered + "\n")
