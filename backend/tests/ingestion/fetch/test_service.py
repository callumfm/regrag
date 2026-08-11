"""The previous-run query the fetch diff reads."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.service import get_corpus_docs, get_previous_docs
from app.ingestion.schemas import IngestRun

pytestmark = pytest.mark.anyio


async def test_baseline_empty_when_no_prior_runs(db_session: AsyncSession):
    assert await get_previous_docs(db_session, ["mrv"]) == {}


async def test_baseline_is_latest_run_with_rows_filtered_to_topics(
    db_session: AsyncSession, make_document
):
    old = IngestRun(status=IngestRunStatus.COMPLETED)
    latest = IngestRun(status=IngestRunStatus.FAILED)
    db_session.add_all(
        [
            make_document(old, "32014R0666", topic="mrv"),
            make_document(latest, "32015R0757", topic="mrv", resolved_celex="02015R0757-20250101"),
            make_document(latest, "32023R1805", topic="fueleu"),
        ]
    )
    await db_session.flush()

    previous = await get_previous_docs(db_session, ["mrv"])
    assert set(previous) == {"32015R0757"}
    assert previous["32015R0757"].resolved_celex == "02015R0757-20250101"


async def test_baseline_survives_a_newer_run_of_another_topic(
    db_session: AsyncSession, make_document
):
    mrv_run = IngestRun(status=IngestRunStatus.COMPLETED)
    db_session.add(make_document(mrv_run, "32015R0757", topic="mrv"))
    await db_session.flush()

    fueleu_run = IngestRun(status=IngestRunStatus.COMPLETED)
    db_session.add(make_document(fueleu_run, "32023R1805", topic="fueleu"))
    await db_session.flush()

    assert fueleu_run.id > mrv_run.id
    assert set(await get_previous_docs(db_session, ["mrv"])) == {"32015R0757"}


async def test_baseline_skips_newer_run_without_rows(db_session: AsyncSession, make_document):
    with_rows = IngestRun(status=IngestRunStatus.COMPLETED)
    db_session.add(make_document(with_rows, "32015R0757", topic="mrv"))
    db_session.add(IngestRun(status=IngestRunStatus.FAILED))
    await db_session.flush()

    assert set(await get_previous_docs(db_session, ["mrv"])) == {"32015R0757"}


async def test_an_aborted_runs_rows_are_still_held(db_session: AsyncSession, make_document):
    """A run that died mid-loop downloaded what it committed, so the next run may reuse it."""
    complete = IngestRun(status=IngestRunStatus.COMPLETED)
    aborted = IngestRun(status=IngestRunStatus.ABORTED)
    db_session.add_all(
        [
            make_document(complete, "32015R0757", topic="mrv"),
            make_document(complete, "32023R2449", topic="mrv"),
            make_document(aborted, "32015R0757", topic="mrv", resolved_celex="02015R0757-20250101"),
        ]
    )
    await db_session.flush()

    previous = await get_previous_docs(db_session, ["mrv"])
    assert set(previous) == {"32015R0757", "32023R2449"}
    assert previous["32015R0757"].resolved_celex == "02015R0757-20250101"


async def test_an_aborted_run_does_not_stand_for_the_corpus(
    db_session: AsyncSession, make_document
):
    """The fingerprint describes a whole corpus, so a prefix must not be the topic's latest."""
    complete = IngestRun(status=IngestRunStatus.COMPLETED)
    aborted = IngestRun(status=IngestRunStatus.ABORTED)
    db_session.add_all(
        [
            make_document(complete, "32015R0757", topic="mrv"),
            make_document(complete, "32023R2449", topic="mrv"),
            make_document(aborted, "32015R0757", topic="mrv"),
        ]
    )
    await db_session.flush()

    assert {doc.celex for doc in await get_corpus_docs(db_session)} == {
        "32015R0757",
        "32023R2449",
    }
