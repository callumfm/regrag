"""`evals stamp`: record what the cited text says now, asserting it has been read."""

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import utc_today
from app.core.config import config
from app.evals.dataset.exceptions import UnresolvedReferenceError
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


Stamps = dict[CaseReference, tuple[str, ...]]


async def _selected_stamps(session: AsyncSession, dataset: EvalDataset) -> Stamps:
    """What every selected reference hashes to now, refusing the whole stamp rather than
    recording an empty one for a division the corpus cannot resolve."""
    stamps: Stamps = {}
    unresolved = []
    for case in dataset.selected_cases:
        for reference in case.references:
            stamps[reference] = await current_stamp(session, reference)
            if not stamps[reference]:
                unresolved.append(f"  {case.id}  {reference.celex} {reference.citation}")

    if unresolved:
        raise UnresolvedReferenceError(
            "no stored chunk answers to these references, so nothing was stamped:\n"
            + "\n".join(unresolved)
        )
    return stamps


def _stamp_case(case: EvalCase, stamps: Stamps) -> EvalCase:
    """The case with every reference restamped to what its division hashes to now."""
    references = [
        reference.model_copy(update={"content_hashes": stamps[reference]})
        for reference in case.references
    ]
    return case.model_copy(update={"references": tuple(references)})


async def stamp_dataset(session: AsyncSession, dataset: EvalDataset) -> EvalDataset:
    """Restamp only the selected cases; the corpus stamp covers the whole dataset, so it is
    rewritten only by an unfiltered stamp."""
    stamps = await _selected_stamps(session, dataset)
    selected = {case.id for case in dataset.selected_cases}
    cases = [_stamp_case(case, stamps) if case.id in selected else case for case in dataset.cases]
    if dataset.selection.selects_a_subset:
        corpus = dataset.corpus
    else:
        version = await get_latest_corpus_version(session)
        corpus = CorpusStamp(corpus_version=version, stamped_at=utc_today())

    return dataset.model_copy(update={"cases": tuple(cases), "corpus": corpus})


def save_dataset(dataset: EvalDataset, path: Path = config.EVAL_DATASET_PATH) -> None:
    """Write the dataset back out, without the per-run selection or anything a case left unset."""
    rendered = dataset.model_dump_json(indent=2, exclude={"selection"}, exclude_defaults=True)
    path.write_text(rendered + "\n")
