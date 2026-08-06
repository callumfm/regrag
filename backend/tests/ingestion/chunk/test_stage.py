"""The chunk stage over a parsed corpus: reconciliation in both directions."""

from collections.abc import Callable

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.chunk.stage import chunk_documents
from app.ingestion.enums import SectionKind
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


async def test_chunks_of_a_ref_no_longer_discovered_are_dropped(
    db_session: AsyncSession, make_chunk_row: Callable[..., DocumentChunk]
) -> None:
    db_session.add(make_chunk_row(ref="repealed", topic="fueleu"))
    await db_session.flush()
    result = await chunk_documents(
        db_session, [], corpus_version="v1", topics=["fueleu"], discovered=["32023R1805"]
    )
    assert result.removed == 1
