"""Checking the dataset against the corpus, and stamping it with what the corpus now says."""

from collections.abc import Callable

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.evals.dataset.models import CaseReference, CorpusStamp, UnresolvedReference
from app.evals.dataset.service import (
    find_changed_documents,
    find_stale_references,
    find_unresolved_references,
    inspect_dataset,
    stamp_dataset,
)
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.schemas import IngestRun
from tests.evals.conftest import REFERENCE, eval_case, eval_dataset, out_of_corpus_case

pytestmark = pytest.mark.anyio

STORED_ANNEX = CaseReference(celex="32023R1805", annex="IV")
MISSING = CaseReference(celex="32023R1805", article="999")
ARTICLE_4_HASH = ("b" * 12,)
"""The first 12 characters of the make_chunk_row default content hash."""


async def test_a_reference_a_stored_chunk_answers_to_is_resolved(
    db_session: AsyncSession, ingest_run: IngestRun, make_chunk_row: Callable[..., DocumentChunk]
) -> None:
    db_session.add(make_chunk_row(ingest_run))
    db_session.add(
        make_chunk_row(
            ingest_run, article=None, annex="IV", citation="Annex IV", content_hash="c" * 64
        )
    )
    await db_session.flush()

    unresolved = await find_unresolved_references(
        db_session, eval_dataset(eval_case(references=(REFERENCE, STORED_ANNEX)))
    )

    assert unresolved == ()


async def test_a_reference_no_chunk_answers_to_is_reported_with_its_case(
    db_session: AsyncSession, ingest_run: IngestRun, make_chunk_row: Callable[..., DocumentChunk]
) -> None:
    db_session.add(make_chunk_row(ingest_run))
    await db_session.flush()

    unresolved = await find_unresolved_references(
        db_session,
        eval_dataset(eval_case(id="ok"), eval_case(id="stale", references=(MISSING,))),
    )

    assert unresolved == (UnresolvedReference(case_id="stale", target=MISSING),)


async def test_out_of_corpus_cases_have_nothing_to_resolve(db_session: AsyncSession) -> None:
    unresolved = await find_unresolved_references(db_session, eval_dataset(out_of_corpus_case()))

    assert unresolved == ()


# Soft drift: the cited text moved under a case that still resolves


@pytest.fixture
async def article_4(
    db_session: AsyncSession, ingest_run: IngestRun, make_chunk_row: Callable[..., DocumentChunk]
) -> DocumentChunk:
    """One stored chunk covering the division every factory case cites."""
    chunk = make_chunk_row(ingest_run)
    db_session.add(chunk)
    await db_session.flush()
    return chunk


async def test_a_case_stamped_against_the_text_that_is_stored_is_not_stale(
    db_session: AsyncSession, article_4: DocumentChunk
) -> None:
    stamped = CaseReference(celex="32023R1805", article="4", content_hashes=ARTICLE_4_HASH)

    assert (
        await find_stale_references(db_session, eval_dataset(eval_case(references=(stamped,))))
        == ()
    )


async def test_a_case_whose_cited_text_changed_is_reported_with_both_stamps(
    db_session: AsyncSession, article_4: DocumentChunk
) -> None:
    """The amendment that keeps retrieving: the division still resolves, so nothing else
    catches it, but the answer was written against text that has since been rewritten."""
    stamped = CaseReference(celex="32023R1805", article="4", content_hashes=("0" * 12,))

    [stale] = await find_stale_references(
        db_session, eval_dataset(eval_case(id="amended", references=(stamped,)))
    )

    assert stale.case_id == "amended"
    assert stale.stamped == ("0" * 12,)
    assert stale.current == ARTICLE_4_HASH


async def test_an_unstamped_reference_is_passed_over_rather_than_called_stale(
    db_session: AsyncSession, article_4: DocumentChunk
) -> None:
    """Nothing was recorded to compare against, so calling it stale would be an invention."""
    assert await find_stale_references(db_session, eval_dataset(eval_case())) == ()


async def test_a_reference_that_stopped_resolving_reads_as_stale_too(
    db_session: AsyncSession, article_4: DocumentChunk
) -> None:
    """Hard drift is reported by find_unresolved_references as well; a stamped reference
    hashing to nothing is still text that moved, so it must not read as unchanged."""
    stamped = CaseReference(celex="32023R1805", article="999", content_hashes=("0" * 12,))

    [stale] = await find_stale_references(
        db_session, eval_dataset(eval_case(references=(stamped,)))
    )

    assert stale.current == ()


# Coarse drift: a cited act's bytes moved


def _stamped_against(**documents: str):
    return eval_dataset(eval_case()).model_copy(
        update={"corpus": CorpusStamp(stamped_at="2026-08-28", documents=dict(documents))}
    )


async def test_an_act_whose_bytes_are_unchanged_is_not_reported(
    db_session: AsyncSession, ingest_run: IngestRun, make_document: Callable[..., RawDocument]
) -> None:
    db_session.add(make_document(ingest_run, sha256="9f2c"))
    await db_session.flush()

    changed = await find_changed_documents(db_session, _stamped_against(**{"32023R1805": "9f2c"}))

    assert changed == ()


async def test_an_amended_act_is_named_with_what_it_hashed_to_and_hashes_to_now(
    db_session: AsyncSession, ingest_run: IngestRun, make_document: Callable[..., RawDocument]
) -> None:
    db_session.add(make_document(ingest_run, sha256="4e81"))
    await db_session.flush()

    [changed] = await find_changed_documents(db_session, _stamped_against(**{"32023R1805": "9f2c"}))

    assert (changed.celex, changed.stamped, changed.current) == ("32023R1805", "9f2c", "4e81")


async def test_an_act_that_left_the_corpus_has_no_current_hash(db_session: AsyncSession) -> None:
    [changed] = await find_changed_documents(db_session, _stamped_against(**{"32023R1805": "9f2c"}))

    assert changed.current is None


async def test_an_unstamped_dataset_has_no_document_drift_to_report(
    db_session: AsyncSession,
) -> None:
    assert await find_changed_documents(db_session, eval_dataset(eval_case())) == ()


# Stamping


async def test_stamping_records_what_each_cited_division_hashes_to_now(
    db_session: AsyncSession,
    ingest_run: IngestRun,
    article_4: DocumentChunk,
    make_document: Callable[..., RawDocument],
) -> None:
    db_session.add(make_document(ingest_run, sha256="9f2c"))
    await db_session.flush()

    stamped = await stamp_dataset(db_session, eval_dataset(eval_case()))

    assert stamped.cases[0].references[0].content_hashes == ARTICLE_4_HASH
    assert stamped.corpus is not None
    assert stamped.corpus.documents == {"32023R1805": "9f2c"}


async def test_stamping_clears_the_staleness_it_recorded(
    db_session: AsyncSession, article_4: DocumentChunk
) -> None:
    was_stale = CaseReference(celex="32023R1805", article="4", content_hashes=("0" * 12,))
    dataset = eval_dataset(eval_case(references=(was_stale,)))

    stamped = await stamp_dataset(db_session, dataset)

    assert await find_stale_references(db_session, stamped) == ()


async def test_a_filtered_stamp_leaves_the_cases_it_did_not_select_alone(
    db_session: AsyncSession, article_4: DocumentChunk
) -> None:
    """Repairing one case must not silently clear the staleness of every other one."""
    was_stale = CaseReference(celex="32023R1805", article="4", content_hashes=("0" * 12,))
    dataset = eval_dataset(
        eval_case(id="fueleu-repaired", references=(was_stale,)),
        eval_case(id="mrv-untouched", references=(was_stale,)),
        case_filter="fueleu",
    )

    stamped = await stamp_dataset(db_session, dataset)

    assert stamped.cases[0].references[0].content_hashes == ARTICLE_4_HASH
    assert stamped.cases[1].references[0].content_hashes == ("0" * 12,)


async def test_a_filtered_stamp_leaves_the_corpus_block_alone(
    db_session: AsyncSession,
    ingest_run: IngestRun,
    article_4: DocumentChunk,
    make_document: Callable[..., RawDocument],
) -> None:
    """The block covers the whole dataset, so only an unfiltered stamp can honestly claim it."""
    db_session.add(make_document(ingest_run, sha256="4e81"))
    await db_session.flush()
    dataset = _stamped_against(**{"32023R1805": "9f2c"}).model_copy(update={"case_filter": "case"})

    stamped = await stamp_dataset(db_session, dataset)

    assert stamped.corpus is not None
    assert stamped.corpus.documents == {"32023R1805": "9f2c"}


async def test_inspect_gathers_every_drift_signal_in_one_pass(
    db_session: AsyncSession, article_4: DocumentChunk
) -> None:
    moved = CaseReference(celex="32023R1805", article="4", content_hashes=("0" * 12,))
    dataset = eval_dataset(
        eval_case(id="moved", references=(moved,)),
        eval_case(id="gone", references=(MISSING,)),
        eval_case(id="never-stamped"),
    )

    drift = await inspect_dataset(db_session, dataset)

    assert drift.stale_case_ids == ("moved",)
    assert [item.case_id for item in drift.unresolved] == ["gone"]
    assert drift.unstamped == ("gone", "never-stamped")
