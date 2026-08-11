"""Retrieval fixtures: a stored corpus, embedded by a deterministic stand-in for Voyage."""

import re
import zlib
from math import sqrt

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import config
from app.ingestion.chunk.chunker import chunk_document
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.chunk.service import upsert_document_chunks
from app.ingestion.parse.models import ParsedDocument
from app.ingestion.schemas import IngestRun
from tests.conftest import chunk_rows

TOKEN = re.compile(r"\w+")


def toy_embed(text: str) -> list[float]:
    """Token hashes into the real width, L2-normalised, so overlapping texts land near."""
    vector = [0.0] * config.EMBED_DIMENSIONS
    for token in TOKEN.findall(text.lower()):
        vector[zlib.crc32(token.encode()) % config.EMBED_DIMENSIONS] += 1.0
    norm = sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


@pytest.fixture
async def corpus(
    db_session: AsyncSession, ingest_run: IngestRun, fueleu: ParsedDocument, mrv: ParsedDocument
) -> list[DocumentChunk]:
    """Both fixture acts chunked, stored and embedded, without going near a provider."""
    for document in (fueleu, mrv):
        await upsert_document_chunks(
            db_session,
            celex=document.celex,
            chunks=chunk_document(document),
            ingest_run_id=ingest_run.id,
        )
    rows = await chunk_rows(db_session)
    for row in rows:
        row.embedding = toy_embed(row.text)
    await db_session.flush()
    return rows
