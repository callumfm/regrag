"""The chunk stage over a parsed corpus: reconciliation in both directions."""

from collections.abc import Callable

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk import stage
from app.ingestion.chunk.chunker import chunk_document
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.chunk.stage import chunk_documents
from app.ingestion.enums import SectionKind
from app.ingestion.exceptions import ParseError
from app.ingestion.parse.models import ParsedDocument, Section

pytestmark = pytest.mark.anyio


async def test_reconciles_each_document_and_sums_one_result(db_session: AsyncSession) -> None:
    document = ParsedDocument(
        ref="32023R1805",
        topic="fueleu",
        sections=(Section(kind=SectionKind.PARAGRAPH, number="1", text="Text."),),
    )
    result = await chunk_documents(
        db_session, [document], corpus_version="v1", topics=["fueleu"], discovered=["32023R1805"]
    )
    assert (result.added, result.removed, result.unchanged) == (1, 0, 0)


async def test_a_second_identical_run_changes_nothing(
    db_session: AsyncSession,
) -> None:
    document = ParsedDocument(
        ref="32023R1805",
        topic="fueleu",
        sections=(Section(kind=SectionKind.PARAGRAPH, number="1", text="Text."),),
    )
    await chunk_documents(
        db_session, [document], corpus_version="v1", topics=["fueleu"], discovered=["32023R1805"]
    )
    result = await chunk_documents(
        db_session, [document], corpus_version="v1", topics=["fueleu"], discovered=["32023R1805"]
    )
    assert (result.added, result.removed, result.unchanged) == (0, 0, 1)


def parsed(ref: str) -> ParsedDocument:
    """A one-paragraph document, distinct per ref so its chunks are distinct too."""
    return ParsedDocument(
        ref=ref,
        topic="fueleu",
        sections=(Section(kind=SectionKind.PARAGRAPH, number="1", text=f"Text of {ref}."),),
    )


async def test_a_document_that_will_not_chunk_is_recorded_and_the_rest_persist(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    def chunk_one(document: ParsedDocument):
        if document.ref == "broken":
            raise ParseError("no sections")
        return chunk_document(document)

    monkeypatch.setattr("app.ingestion.chunk.stage.chunk_document", chunk_one)
    result = await chunk_documents(
        db_session,
        [parsed("broken"), parsed("32023R1805")],
        corpus_version="v1",
        topics=["fueleu"],
        discovered=["broken", "32023R1805"],
    )
    assert "broken" in result.failed
    assert result.added == 1
    assert not result.ok


async def test_a_database_failure_on_one_document_does_not_abort_the_rest(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed flush poisons the transaction, so each document reconciles in its own savepoint."""
    real = stage.upsert_document_chunks

    async def fail_one(session, *, ref, chunks, corpus_version):
        if ref == "broken":
            raise IntegrityError("INSERT", {}, Exception("duplicate key"))
        return await real(session, ref=ref, chunks=chunks, corpus_version=corpus_version)

    monkeypatch.setattr(stage, "upsert_document_chunks", fail_one)
    result = await chunk_documents(
        db_session,
        [parsed("broken"), parsed("32023R1805")],
        corpus_version="v1",
        topics=["fueleu"],
        discovered=["broken", "32023R1805"],
    )
    assert "IntegrityError" in result.failed["broken"]
    assert result.added == 1


async def test_chunks_of_a_ref_no_longer_discovered_are_dropped(
    db_session: AsyncSession, make_chunk_row: Callable[..., DocumentChunk]
) -> None:
    db_session.add(make_chunk_row(ref="repealed", topic="fueleu"))
    await db_session.flush()
    result = await chunk_documents(
        db_session, [], corpus_version="v1", topics=["fueleu"], discovered=["32023R1805"]
    )
    assert result.removed == 1
