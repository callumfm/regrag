"""Embed stage: fill in the vector of every chunk that has none."""

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from itertools import batched, groupby
from operator import attrgetter
from typing import NamedTuple

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import config
from app.core.llm import EMBED_BATCH_SIZE, EmbedInput, LLMError, embed, llm_retry
from app.ingestion.chunk.models import ChunkQuery
from app.ingestion.chunk.service import count_chunks, get_chunks, update_chunks
from app.ingestion.embed.models import EmbedOutcome


class ChunkToEmbed(NamedTuple):
    """One vectorless chunk, detached from the ORM so a mid-page commit cannot expire it."""

    id: int
    celex: str
    text: str


Batch = tuple[str, Sequence[ChunkToEmbed]]
PendingVectors = asyncio.Task[list[list[float]]]


async def _unembedded_pages(session: AsyncSession) -> AsyncIterator[list[ChunkToEmbed]]:
    """Vectorless chunks a page at a time, the cursor read before the page can be rolled back."""
    after: tuple[str, int] | None = None
    while page := await get_chunks(
        session, ChunkQuery(has_embedding=False, after=after, limit=config.EMBED_PAGE_SIZE)
    ):
        after = (page[-1].celex, page[-1].id)
        yield [ChunkToEmbed(chunk.id, chunk.celex, chunk.text) for chunk in page]


def _batch_by_document(chunks: Sequence[ChunkToEmbed]) -> Iterator[Batch]:
    """Provider-sized batches that never span documents, each labelled with its document."""
    for celex, group in groupby(chunks, attrgetter("celex")):
        for batch in batched(group, EMBED_BATCH_SIZE):
            yield celex, batch


@llm_retry
async def _embed_batch(batch: Sequence[ChunkToEmbed]) -> list[list[float]]:
    """One provider call embedding one batch's texts, retrying transient failures."""
    return await embed([chunk.text for chunk in batch], input_type=EmbedInput.DOCUMENT)


@asynccontextmanager
async def _embed_batches_concurrently(
    batches: Sequence[Batch],
) -> AsyncIterator[list[PendingVectors]]:
    """Every batch's provider call in flight, at most EMBED_CONCURRENCY at once; leaving
    the block cancels whatever an abort left running."""
    semaphore = asyncio.Semaphore(config.EMBED_CONCURRENCY)

    async def paced(batch: Sequence[ChunkToEmbed]) -> list[list[float]]:
        async with semaphore:
            return await _embed_batch(batch)

    tasks = [asyncio.create_task(paced(batch)) for _, batch in batches]
    try:
        yield tasks
    finally:
        for task in tasks:
            task.cancel()


async def _store_batch(
    session: AsyncSession,
    celex: str,
    batch: Sequence[ChunkToEmbed],
    vectors: PendingVectors,
    result: EmbedOutcome,
) -> None:
    """Write one batch's vectors once embedded, commit, count — or record the failure against
    its document. The savepoint keeps a failed write from dragging down earlier uncommitted
    work, like the prune stage's deletes; anything but a provider or database error aborts."""
    try:
        embedded = await vectors
        async with session.begin_nested():
            await update_chunks(
                session,
                [
                    {"id": chunk.id, "embedding": vector}
                    for chunk, vector in zip(batch, embedded, strict=True)
                ],
            )
    except (LLMError, SQLAlchemyError) as exc:
        result.fail(celex, exc, chunks=len(batch))
    else:
        await session.commit()
        result.embedded += len(batch)


async def embed_chunks(session: AsyncSession) -> EmbedOutcome:
    """Fill in every missing chunk vector: provider calls run concurrently within a page, and
    each batch commits as it lands, so a failure costs at most one batch's paid embeddings."""
    result = EmbedOutcome(already_embedded=await count_chunks(session, has_embedding=True))
    async for page in _unembedded_pages(session):
        batches = list(_batch_by_document(page))
        async with _embed_batches_concurrently(batches) as pending:
            for (celex, batch), vectors in zip(batches, pending, strict=True):
                await _store_batch(session, celex, batch, vectors, result)
    return result
