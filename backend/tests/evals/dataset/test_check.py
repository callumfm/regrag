"""`evals check`: how a drifted reference and a moved corpus are classified."""

from collections.abc import Callable

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.evals.dataset.check import find_drift, find_moved_corpus, stale_case_ids
from app.evals.dataset.enums import DriftKind
from app.evals.dataset.models import CaseReference, CorpusStamp, DriftedReference
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.schemas import IngestRun
from tests.evals.conftest import eval_case, eval_dataset, out_of_corpus_case

pytestmark = pytest.mark.anyio

ARTICLE_4 = ("b" * 12,)
"""The first 12 characters of the make_chunk_row default content hash."""

MOVED = CaseReference(celex="32023R1805", article="4", content_hashes=("0" * 12,))
STAMPED = CaseReference(celex="32023R1805", article="4", content_hashes=ARTICLE_4)
MISSING = CaseReference(celex="32023R1805", article="999")


@pytest.fixture
async def article_4(
    db_session: AsyncSession, ingest_run: IngestRun, make_chunk_row: Callable[..., DocumentChunk]
) -> DocumentChunk:
    """One stored chunk covering the division every factory case cites."""
    chunk = make_chunk_row(ingest_run)
    db_session.add(chunk)
    await db_session.flush()
    return chunk


async def test_a_case_stamped_against_the_text_that_is_stored_has_not_drifted(
    db_session: AsyncSession, article_4: DocumentChunk
) -> None:
    assert await find_drift(db_session, eval_dataset(eval_case(references=(STAMPED,)))) == ()


async def test_a_case_whose_cited_text_changed_is_stale(
    db_session: AsyncSession, article_4: DocumentChunk
) -> None:
    """The amendment that keeps retrieving: the division still resolves, so nothing else
    catches it, but the answer was written against text that has since been rewritten."""
    [drifted] = await find_drift(
        db_session, eval_dataset(eval_case(id="amended", references=(MOVED,)))
    )

    assert drifted == DriftedReference(case_id="amended", target=MOVED, kind=DriftKind.STALE)


async def test_a_reference_no_chunk_answers_to_is_unresolved(
    db_session: AsyncSession, article_4: DocumentChunk
) -> None:
    [drifted] = await find_drift(
        db_session,
        eval_dataset(
            eval_case(id="ok", references=(STAMPED,)),
            eval_case(id="gone", references=(MISSING,)),
        ),
    )

    assert (drifted.case_id, drifted.kind) == ("gone", DriftKind.UNRESOLVED)


async def test_a_stamped_reference_that_stopped_resolving_reads_as_unresolved_not_stale(
    db_session: AsyncSession, article_4: DocumentChunk
) -> None:
    """Hard drift is the louder fact and the one that fails the command, so it wins."""
    stamped_but_gone = CaseReference(celex="32023R1805", article="999", content_hashes=ARTICLE_4)

    [drifted] = await find_drift(
        db_session, eval_dataset(eval_case(references=(stamped_but_gone,)))
    )

    assert drifted.kind is DriftKind.UNRESOLVED


async def test_a_reference_that_was_never_stamped_is_unstamped_rather_than_stale(
    db_session: AsyncSession, article_4: DocumentChunk
) -> None:
    """Nothing was recorded to compare against, so calling it stale would be an invention."""
    [drifted] = await find_drift(db_session, eval_dataset(eval_case()))

    assert drifted.kind is DriftKind.UNSTAMPED


async def test_out_of_corpus_cases_have_nothing_to_drift(db_session: AsyncSession) -> None:
    assert await find_drift(db_session, eval_dataset(out_of_corpus_case())) == ()


def test_stale_case_ids_names_a_case_once_and_leaves_the_other_kinds_out() -> None:
    article_5 = CaseReference(celex="32023R1805", article="5", content_hashes=("1" * 12,))
    drifted = (
        DriftedReference(case_id="amended", target=MOVED, kind=DriftKind.STALE),
        DriftedReference(case_id="amended", target=article_5, kind=DriftKind.STALE),
        DriftedReference(case_id="gone", target=MISSING, kind=DriftKind.UNRESOLVED),
    )

    assert stale_case_ids(drifted) == ("amended",)


# Whether the corpus itself has moved


def _stamped_at(version: str | None):
    return eval_dataset(eval_case()).model_copy(
        update={"corpus": CorpusStamp(corpus_version=version, stamped_at="2026-08-28")}
    )


CURRENT = "2026-09-02-4e81a90"


def test_a_dataset_stamped_at_the_current_version_has_not_moved() -> None:
    assert find_moved_corpus(_stamped_at(CURRENT), CURRENT) is None


def test_a_dataset_stamped_at_an_older_version_names_the_current_one() -> None:
    assert find_moved_corpus(_stamped_at("2026-08-15-2cc038d"), CURRENT) == CURRENT


def test_an_unstamped_dataset_has_no_corpus_move_to_report() -> None:
    assert find_moved_corpus(eval_dataset(eval_case()), CURRENT) is None
