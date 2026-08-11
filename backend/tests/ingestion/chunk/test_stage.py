"""The chunk stage on one parsed document, and the corpus-wide prune that follows the loop."""

from collections.abc import Callable

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk import stage
from app.ingestion.chunk.chunker import chunk_document
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.chunk.stage import chunk_and_store_document, prune_chunks
from app.ingestion.enums import SectionKind
from app.ingestion.exceptions import ParseError
from app.ingestion.parse.models import ParsedDocument, Section
from app.ingestion.schemas import IngestRun
from tests.conftest import chunk_versions

pytestmark = pytest.mark.anyio


def parsed(celex: str) -> ParsedDocument:
    """A one-paragraph document, distinct per celex so its chunks are distinct too."""
    return ParsedDocument(
        celex=celex,
        topic="fueleu",
        sections=(Section(kind=SectionKind.PARAGRAPH, number="1", text=f"Text of {celex}."),),
    )


async def test_reconciles_the_document_and_reports_what_changed(
    db_session: AsyncSession, ingest_run: IngestRun
) -> None:
    result = await chunk_and_store_document(
        db_session, parsed("32023R1805"), ingest_run_id=ingest_run.id
    )
    assert (result.added, result.removed, result.unchanged) == (1, 0, 0)


async def test_a_second_identical_run_changes_nothing(
    db_session: AsyncSession, ingest_run: IngestRun
) -> None:
    document = parsed("32023R1805")
    await chunk_and_store_document(db_session, document, ingest_run_id=ingest_run.id)
    result = await chunk_and_store_document(db_session, document, ingest_run_id=ingest_run.id)
    assert (result.added, result.removed, result.unchanged) == (0, 0, 1)


async def test_a_document_that_will_not_chunk_is_recorded_not_raised(
    db_session: AsyncSession, ingest_run: IngestRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    def chunk_one(document: ParsedDocument):
        if document.celex == "broken":
            raise ParseError("no sections")
        return chunk_document(document)

    monkeypatch.setattr("app.ingestion.chunk.stage.chunk_document", chunk_one)
    result = await chunk_and_store_document(
        db_session, parsed("broken"), ingest_run_id=ingest_run.id
    )

    assert "broken" in result.failed
    assert not result.ok


async def test_a_database_failure_is_contained_by_the_documents_own_savepoint(
    db_session: AsyncSession, ingest_run: IngestRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed flush poisons the transaction, so each document reconciles in its own savepoint."""

    async def fail(session, *, celex, chunks, ingest_run_id):
        raise IntegrityError("INSERT", {}, Exception("duplicate key"))

    monkeypatch.setattr(stage, "upsert_document_chunks", fail)
    result = await chunk_and_store_document(
        db_session, parsed("broken"), ingest_run_id=ingest_run.id
    )
    assert "IntegrityError" in result.failed["broken"]

    monkeypatch.undo()
    survives = await chunk_and_store_document(
        db_session, parsed("32023R1805"), ingest_run_id=ingest_run.id
    )
    assert survives.added == 1


async def test_prune_chunks_removes_every_celex_outside_the_corpus(
    db_session: AsyncSession,
    ingest_run: IngestRun,
    make_chunk_row: Callable[..., DocumentChunk],
) -> None:
    db_session.add(make_chunk_row(ingest_run, celex="repealed", topic="fueleu"))
    await db_session.flush()

    result = await prune_chunks(db_session, corpus_celexes=["32023R1805"])

    assert result.removed == 1
    assert await chunk_versions(db_session, "repealed") == set()


async def test_chunking_no_longer_prunes(
    db_session: AsyncSession,
    ingest_run: IngestRun,
    make_chunk_row: Callable[..., DocumentChunk],
) -> None:
    """Pruning needs the whole corpus, so a per-document chunk call must leave outsiders alone."""
    db_session.add(make_chunk_row(ingest_run, celex="repealed", topic="fueleu"))
    await db_session.flush()

    result = await chunk_and_store_document(
        db_session, parsed("32023R1805"), ingest_run_id=ingest_run.id
    )

    assert result.removed == 0
    assert await chunk_versions(db_session, "repealed") != set()
