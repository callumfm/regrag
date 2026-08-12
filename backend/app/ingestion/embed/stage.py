"""Embed stage: fill in the vector of every chunk that has none."""

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from itertools import batched, groupby
from operator import attrgetter

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import config
from app.core.llm import EMBED_BATCH_SIZE, EmbedInput, LLMError, embed, llm_retry
from app.ingestion.chunk.service import (
    ChunkToEmbed,
    count_embedded_chunks,
    get_unembedded_chunks,
    set_chunk_embeddings,
)
from app.ingestion.embed.models import EmbedOutcome

Batch = tuple[str, Sequence[ChunkToEmbed]]
PendingVectors = asyncio.Task[list[list[float]]]


def _batches(chunks: Sequence[ChunkToEmbed]) -> Iterator[Batch]:
    """Provider-sized batches that never span documents, each labelled with its document."""
    for celex, group in groupby(chunks, attrgetter("celex")):
        for batch in batched(group, EMBED_BATCH_SIZE):
            yield celex, batch


async def _pages(session: AsyncSession) -> AsyncIterator[Sequence[ChunkToEmbed]]:
    """Vectorless chunks a page at a time, the cursor read before the page can be rolled back."""
    after: tuple[str, int] | None = None
    while page := await get_unembedded_chunks(session, after=after, limit=config.EMBED_PAGE_SIZE):
        after = (page[-1].celex, page[-1].id)
        yield page


@llm_retry
async def _embed_batch(batch: Sequence[ChunkToEmbed]) -> list[list[float]]:
    """One provider call embedding one batch's texts, retrying transient failures."""
    return await embed([chunk.text for chunk in batch], input_type=EmbedInput.DOCUMENT)


@asynccontextmanager
async def _embedding(batches: Sequence[Batch]) -> AsyncIterator[list[PendingVectors]]:
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
    session: AsyncSession, batch: Sequence[ChunkToEmbed], vectors: list[list[float]]
) -> None:
    """Write one batch's vectors inside a savepoint: a plain rollback on failure would
    drag down earlier uncommitted work, like the prune stage's deletes."""
    async with session.begin_nested():
        await set_chunk_embeddings(
            session, {chunk.id: vector for chunk, vector in zip(batch, vectors, strict=True)}
        )


async def _process_batch(
    session: AsyncSession,
    celex: str,
    batch: Sequence[ChunkToEmbed],
    vectors: PendingVectors,
    result: EmbedOutcome,
) -> None:
    """One batch end to end: store its vectors once embedded, commit, count — or record
    the failure against its document. Anything but a provider or database error aborts."""
    try:
        await _store_batch(session, batch, await vectors)
    except (LLMError, SQLAlchemyError) as exc:
        result.fail(celex, exc, chunks=len(batch))
    else:
        await session.commit()
        result.embedded += len(batch)


async def embed_chunks(session: AsyncSession) -> EmbedOutcome:
    """Fill in every missing chunk vector: provider calls run concurrently within a page, and
    each batch commits as it lands, so a failure costs at most one batch's paid embeddings."""
    result = EmbedOutcome(already_embedded=await count_embedded_chunks(session))
    async for page in _pages(session):
        batches = list(_batches(page))
        async with _embedding(batches) as embeds:
            for (celex, batch), vectors in zip(batches, embeds, strict=True):
                await _process_batch(session, celex, batch, vectors, result)
    return result
