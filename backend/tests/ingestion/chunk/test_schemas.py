"""Roundtrip tests for the document chunks table."""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.enums import SectionKind

pytestmark = pytest.mark.anyio


async def test_document_chunk_roundtrip(db_session: AsyncSession, make_chunk_row):
    chunk = make_chunk_row(article="4", paragraph="1", citation="Article 4(1)")
    db_session.add(chunk)
    await db_session.flush()
    db_session.expire_all()

    fetched = (await db_session.scalars(select(DocumentChunk))).one()
    assert fetched.citation == "Article 4(1)"
    assert fetched.kind is SectionKind.PARAGRAPH
    assert fetched.heading_path == ["Chapter I", "Section 2"]
    assert fetched.references == [{"raw": "Annex I", "annex": "I"}]
    assert fetched.created_at is not None


async def test_chunk_identity_unique_per_document(db_session: AsyncSession, make_chunk_row):
    db_session.add(make_chunk_row())
    db_session.add(make_chunk_row())
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_same_hash_at_a_later_occurrence_is_allowed(db_session: AsyncSession, make_chunk_row):
    db_session.add(make_chunk_row(occurrence=0))
    db_session.add(make_chunk_row(occurrence=1))
    await db_session.flush()

    assert len((await db_session.scalars(select(DocumentChunk))).all()) == 2
