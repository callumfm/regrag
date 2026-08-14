"""The chunk stage: one parsed document reconciled against the rows already stored for it."""

from collections.abc import Callable

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk import stage
from app.ingestion.chunk.models import ChunkCounts
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.chunk.stage import chunk_and_store_document
from app.ingestion.chunk.tree import chunk_document
from app.ingestion.enums import SectionKind, Stage
from app.ingestion.exceptions import DocumentFailed, ParseError
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
    counts = await chunk_and_store_document(
        db_session, parsed("32023R1805"), ingest_run_id=ingest_run.id
    )
    assert counts == ChunkCounts(added=1)


async def test_a_second_identical_run_changes_nothing(
    db_session: AsyncSession, ingest_run: IngestRun
) -> None:
    document = parsed("32023R1805")
    await chunk_and_store_document(db_session, document, ingest_run_id=ingest_run.id)
    counts = await chunk_and_store_document(db_session, document, ingest_run_id=ingest_run.id)
    assert counts == ChunkCounts(kept=1)


async def test_a_document_that_will_not_chunk_fails_at_the_chunk_stage(
    db_session: AsyncSession, ingest_run: IngestRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    def chunk_one(document: ParsedDocument):
        if document.celex == "broken":
            raise ParseError("no sections")
        return chunk_document(document)

    monkeypatch.setattr("app.ingestion.chunk.stage.chunk_document", chunk_one)

    with pytest.raises(DocumentFailed) as excinfo:
        await chunk_and_store_document(db_session, parsed("broken"), ingest_run_id=ingest_run.id)

    assert (excinfo.value.stage, excinfo.value.celex) == (Stage.CHUNK, "broken")


async def test_a_document_that_chunks_to_nothing_fails_instead_of_deleting_itself(
    db_session: AsyncSession, ingest_run: IngestRun
) -> None:
    """The run must not close SUCCESS having quietly emptied a document out of the corpus."""
    await chunk_and_store_document(db_session, parsed("32023R1805"), ingest_run_id=ingest_run.id)

    empty = ParsedDocument(
        celex="32023R1805",
        topic="fueleu",
        sections=(Section(kind=SectionKind.ARTICLE, number="1"),),
    )
    with pytest.raises(DocumentFailed, match="chunked to nothing"):
        await chunk_and_store_document(db_session, empty, ingest_run_id=ingest_run.id)

    assert await chunk_versions(db_session, "32023R1805") != set()


async def test_a_database_failure_is_reported_as_the_documents_own(
    db_session: AsyncSession, ingest_run: IngestRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rolling the flush back is the loop's job; the stage's is to name what could not be stored."""

    async def fail(session, *, celex, chunks, ingest_run_id):
        raise IntegrityError("INSERT", {}, Exception("duplicate key"))

    monkeypatch.setattr(stage, "sync_document_chunks", fail)

    with pytest.raises(DocumentFailed) as excinfo:
        await chunk_and_store_document(db_session, parsed("broken"), ingest_run_id=ingest_run.id)

    assert "IntegrityError" in excinfo.value.reason


async def test_chunking_one_document_leaves_outsiders_alone(
    db_session: AsyncSession,
    ingest_run: IngestRun,
    make_chunk_row: Callable[..., DocumentChunk],
) -> None:
    """Pruning needs the whole corpus, so a per-document chunk call must not attempt it."""
    db_session.add(make_chunk_row(ingest_run, celex="repealed", topic="fueleu"))
    await db_session.flush()

    counts = await chunk_and_store_document(
        db_session, parsed("32023R1805"), ingest_run_id=ingest_run.id
    )

    assert counts.deleted == 0
    assert await chunk_versions(db_session, "repealed") != set()
