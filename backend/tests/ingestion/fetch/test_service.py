"""The standing-corpus query the fetch diff and the fingerprint read."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.models import RawDocsQuery
from app.ingestion.fetch.service import get_raw_documents
from app.ingestion.schemas import IngestRun

pytestmark = pytest.mark.anyio


async def test_baseline_empty_when_no_prior_runs(db_session: AsyncSession):
    assert await get_raw_documents(db_session, RawDocsQuery(include_topics=["mrv"])) == {}


async def test_an_empty_topic_filter_matches_nothing(db_session: AsyncSession, make_document):
    """None leaves the filter off; an explicit empty list must not widen to the whole corpus."""
    run = IngestRun(status=IngestRunStatus.SUCCESS)
    db_session.add(make_document(run, "32015R0757", topic="mrv"))
    await db_session.flush()

    assert await get_raw_documents(db_session, RawDocsQuery(include_topics=[])) == {}
    assert set(await get_raw_documents(db_session, RawDocsQuery())) == {"32015R0757"}


async def test_baseline_is_latest_run_with_rows_filtered_to_topics(
    db_session: AsyncSession, make_document
):
    old = IngestRun(status=IngestRunStatus.SUCCESS)
    latest = IngestRun(status=IngestRunStatus.SUCCESS)
    db_session.add_all(
        [
            make_document(old, "32014R0666", topic="mrv"),
            make_document(latest, "32015R0757", topic="mrv", resolved_celex="02015R0757-20250101"),
            make_document(latest, "32023R1805", topic="fueleu"),
        ]
    )
    await db_session.flush()

    previous = await get_raw_documents(db_session, RawDocsQuery(include_topics=["mrv"]))
    assert set(previous) == {"32015R0757"}
    assert previous["32015R0757"].resolved_celex == "02015R0757-20250101"


async def test_baseline_survives_a_newer_run_of_another_topic(
    db_session: AsyncSession, make_document
):
    mrv_run = IngestRun(status=IngestRunStatus.SUCCESS)
    db_session.add(make_document(mrv_run, "32015R0757", topic="mrv"))
    await db_session.flush()

    fueleu_run = IngestRun(status=IngestRunStatus.SUCCESS)
    db_session.add(make_document(fueleu_run, "32023R1805", topic="fueleu"))
    await db_session.flush()

    assert fueleu_run.id > mrv_run.id
    previous = await get_raw_documents(db_session, RawDocsQuery(include_topics=["mrv"]))
    assert set(previous) == {"32015R0757"}


async def test_a_topic_with_no_successful_run_still_holds_what_it_downloaded(
    db_session: AsyncSession, make_document
):
    """A topic yet to complete a run has no id to measure against, so every row it has stands.

    Its rows must not be retired on another topic's success: the next run reuses these bytes.
    """
    first_attempt = IngestRun(status=IngestRunStatus.FAILED)
    other_topic = IngestRun(status=IngestRunStatus.SUCCESS)
    db_session.add_all(
        [
            make_document(first_attempt, "32015R0757", topic="mrv"),
            make_document(other_topic, "32023R1805", topic="fueleu"),
        ]
    )
    await db_session.flush()

    assert set(await get_raw_documents(db_session, RawDocsQuery(include_topics=["mrv"]))) == {
        "32015R0757"
    }
    assert set(await get_raw_documents(db_session, RawDocsQuery())) == {
        "32015R0757",
        "32023R1805",
    }


async def test_baseline_skips_newer_run_without_rows(db_session: AsyncSession, make_document):
    with_rows = IngestRun(status=IngestRunStatus.SUCCESS)
    db_session.add(make_document(with_rows, "32015R0757", topic="mrv"))
    db_session.add(IngestRun(status=IngestRunStatus.FAILED))
    await db_session.flush()

    previous = await get_raw_documents(db_session, RawDocsQuery(include_topics=["mrv"]))
    assert set(previous) == {"32015R0757"}


async def test_a_failed_run_does_not_retire_what_it_could_not_download(
    db_session: AsyncSession, make_document
):
    """A run that downloaded 1 of 3 holds a prefix too, so it must not stand for the topic."""
    complete = IngestRun(status=IngestRunStatus.SUCCESS)
    failed = IngestRun(status=IngestRunStatus.FAILED)
    db_session.add_all(
        [
            make_document(complete, "32015R0757", topic="mrv"),
            make_document(complete, "32023R2449", topic="mrv"),
            make_document(complete, "32014R0666", topic="mrv"),
            make_document(failed, "32015R0757", topic="mrv"),
        ]
    )
    await db_session.flush()

    held = {"32015R0757", "32023R2449", "32014R0666"}
    assert set(await get_raw_documents(db_session, RawDocsQuery(include_topics=["mrv"]))) == held
    assert set(await get_raw_documents(db_session, RawDocsQuery(exclude_topics=["fueleu"]))) == held


async def test_an_aborted_runs_rows_are_still_held(db_session: AsyncSession, make_document):
    """A run that died mid-loop downloaded what it committed, so the next run may reuse it."""
    complete = IngestRun(status=IngestRunStatus.SUCCESS)
    aborted = IngestRun(status=IngestRunStatus.ABORTED)
    db_session.add_all(
        [
            make_document(complete, "32015R0757", topic="mrv"),
            make_document(complete, "32023R2449", topic="mrv"),
            make_document(aborted, "32015R0757", topic="mrv", resolved_celex="02015R0757-20250101"),
        ]
    )
    await db_session.flush()

    previous = await get_raw_documents(db_session, RawDocsQuery(include_topics=["mrv"]))
    assert set(previous) == {"32015R0757", "32023R2449"}
    assert previous["32015R0757"].resolved_celex == "02015R0757-20250101"


async def test_an_aborted_run_does_not_stand_for_the_corpus(
    db_session: AsyncSession, make_document
):
    """The fingerprint describes a whole corpus, so a prefix must not be the topic's latest."""
    complete = IngestRun(status=IngestRunStatus.SUCCESS)
    aborted = IngestRun(status=IngestRunStatus.ABORTED)
    db_session.add_all(
        [
            make_document(complete, "32015R0757", topic="mrv"),
            make_document(complete, "32023R2449", topic="mrv"),
            make_document(aborted, "32015R0757", topic="mrv"),
        ]
    )
    await db_session.flush()

    assert set(await get_raw_documents(db_session, RawDocsQuery())) == {
        "32015R0757",
        "32023R2449",
    }


async def test_a_failed_run_does_not_stand_for_the_corpus(db_session: AsyncSession, make_document):
    """A FAILED run holds a corpus with holes, so the fingerprint must look past it."""
    complete = IngestRun(status=IngestRunStatus.SUCCESS)
    failed = IngestRun(status=IngestRunStatus.FAILED)
    db_session.add_all(
        [
            make_document(complete, "32015R0757", topic="mrv"),
            make_document(complete, "32023R2449", topic="mrv"),
            make_document(failed, "32015R0757", topic="mrv"),
        ]
    )
    await db_session.flush()

    assert set(await get_raw_documents(db_session, RawDocsQuery())) == {
        "32015R0757",
        "32023R2449",
    }


async def test_corpus_takes_the_newest_row_per_celex(db_session: AsyncSession, make_document):
    complete = IngestRun(status=IngestRunStatus.SUCCESS)
    failed = IngestRun(status=IngestRunStatus.FAILED)
    db_session.add_all(
        [
            make_document(complete, "32015R0757", topic="mrv"),
            make_document(complete, "32023R2449", topic="mrv"),
            make_document(failed, "32015R0757", topic="mrv", resolved_celex="02015R0757-20250101"),
        ]
    )
    await db_session.flush()

    docs = await get_raw_documents(db_session, RawDocsQuery())
    assert set(docs) == {"32015R0757", "32023R2449"}
    assert docs["32015R0757"].resolved_celex == "02015R0757-20250101"
