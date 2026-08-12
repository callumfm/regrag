"""Embed stage: fill in the vector of every chunk that has none."""

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from itertools import batched, groupby
from operator import attrgetter

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm import EMBED_BATCH_SIZE, EmbedInput, LLMError, embed, llm_retry
from app.ingestion.chunk.service import (
    ChunkToEmbed,
    count_embedded_chunks,
    get_unembedded_chunks,
    set_chunk_embeddings,
)
from app.ingestion.constants import EMBED_CONCURRENCY, EMBED_PAGE_SIZE
from app.ingestion.embed.models import EmbedOutcome

Batch = tuple[str, Sequence[ChunkToEmbed]]


def _batches(chunks: Sequence[ChunkToEmbed]) -> Iterator[Batch]:
    """Provider-sized batches that never span documents, each labelled with its document."""
    for celex, group in groupby(chunks, attrgetter("celex")):
        for batch in batched(group, EMBED_BATCH_SIZE):
            yield celex, batch


@llm_retry
async def _embed_texts(chunks: Sequence[ChunkToEmbed]) -> list[list[float]]:
    """Embed one batch, retrying transient provider failures."""
    return await embed([chunk.text for chunk in chunks], input_type=EmbedInput.DOCUMENT)


async def _embed_batches(batches: Sequence[Batch]) -> list[list[list[float]] | BaseException]:
    """Embed every batch, at most EMBED_CONCURRENCY provider calls in flight at once."""
    semaphore = asyncio.Semaphore(EMBED_CONCURRENCY)

    async def paced(batch: Sequence[ChunkToEmbed]) -> list[list[float]]:
        async with semaphore:
            return await _embed_texts(batch)

    return await asyncio.gather(*(paced(batch) for _, batch in batches), return_exceptions=True)


async def _pages(session: AsyncSession) -> AsyncIterator[Sequence[ChunkToEmbed]]:
    """Vectorless chunks a page at a time, the cursor read before the page can be rolled back."""
    after: tuple[str, int] | None = None
    while page := await get_unembedded_chunks(session, after=after, limit=EMBED_PAGE_SIZE):
        after = (page[-1].celex, page[-1].id)
        yield page


async def _store_batch(
    session: AsyncSession,
    celex: str,
    batch: Sequence[ChunkToEmbed],
    vectors: list[list[float]] | BaseException,
    result: EmbedOutcome,
) -> None:
    """Commit one batch's vectors inside a savepoint, or record why the batch has none:
    a plain rollback would drag down earlier uncommitted work, like the prune stage's deletes."""
    if isinstance(vectors, LLMError):
        result.fail(celex, vectors)
        return
    if isinstance(vectors, BaseException):
        raise vectors
    try:
        async with session.begin_nested():
            await set_chunk_embeddings(
                session, {chunk.id: vector for chunk, vector in zip(batch, vectors, strict=True)}
            )
    except SQLAlchemyError as exc:
        result.fail(celex, exc)
    else:
        await session.commit()
        result.embedded += len(batch)


async def embed_chunks(session: AsyncSession) -> EmbedOutcome:
    """Fill in every missing chunk vector: provider calls run concurrently within a page, and
    each batch commits as it lands, so a failure costs at most one batch's paid embeddings."""
    result = EmbedOutcome(already_embedded=await count_embedded_chunks(session))
    async for page in _pages(session):
        batches = list(_batches(page))
        vectors = await _embed_batches(batches)
        for (celex, batch), batch_vectors in zip(batches, vectors, strict=True):
            await _store_batch(session, celex, batch, batch_vectors, result)
    return result
