"""The chunk stage over a parsed corpus: reconciliation in both directions."""

from collections.abc import Callable

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.corpus import chunk_corpus
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.enums import SectionKind
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.parse.models import ParsedDocument, Section

pytestmark = pytest.mark.anyio


async def test_reconciles_each_document_and_sums_one_delta(
    db_session: AsyncSession, make_document: Callable[..., RawDocument]
) -> None:
    document = ParsedDocument(
        ref="32023R1805",
        topic="fueleu",
        sections=(Section(kind=SectionKind.PARAGRAPH, number="1", text="Text."),),
    )
    delta = await chunk_corpus(db_session, [document], "v1", ["fueleu"], ["32023R1805"])
    assert (delta.added, delta.removed, delta.unchanged) == (1, 0, 0)


async def test_a_second_identical_run_changes_nothing(
    db_session: AsyncSession,
) -> None:
    document = ParsedDocument(
        ref="32023R1805",
        topic="fueleu",
        sections=(Section(kind=SectionKind.PARAGRAPH, number="1", text="Text."),),
    )
    await chunk_corpus(db_session, [document], "v1", ["fueleu"], ["32023R1805"])
    delta = await chunk_corpus(db_session, [document], "v1", ["fueleu"], ["32023R1805"])
    assert (delta.added, delta.removed, delta.unchanged) == (0, 0, 1)


async def test_chunks_of_a_ref_no_longer_discovered_are_dropped(
    db_session: AsyncSession, make_chunk_row: Callable[..., DocumentChunk]
) -> None:
    db_session.add(make_chunk_row(ref="repealed", topic="fueleu"))
    await db_session.flush()
    delta = await chunk_corpus(db_session, [], "v1", ["fueleu"], ["32023R1805"])
    assert delta.removed == 1
