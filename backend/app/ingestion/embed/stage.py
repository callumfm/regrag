"""Embed stage: fill in the vector of every chunk that has none."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
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
Vectors = list[list[float]]


@asynccontextmanager
async def _run_concurrently[T, R](
    items: Sequence[T], fn: Callable[[T], Awaitable[R]], *, limit: int
) -> AsyncIterator[list[tuple[T, asyncio.Task[R]]]]:
    """Every item's `fn` call in flight, at most `limit` at once, each task paired with its
    item; leaving the block cancels whatever an abort left running."""
    semaphore = asyncio.Semaphore(limit)

    async def paced(item: T) -> R:
        async with semaphore:
            return await fn(item)

    pairs = [(item, asyncio.create_task(paced(item))) for item in items]
    try:
        yield pairs
    finally:
        for _, task in pairs:
            task.cancel()


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
async def _embed_batch(batch: Batch) -> Vectors:
    """One provider call embedding one batch's texts, retrying transient failures."""
    _, chunks = batch
    return await embed([chunk.text for chunk in chunks], input_type=EmbedInput.DOCUMENT)


async def _store_batch(
    session: AsyncSession, chunks: Sequence[ChunkToEmbed], vectors: Vectors
) -> None:
    """Write one batch's vectors inside a savepoint: a plain rollback on failure would drag
    down earlier uncommitted work, like the prune stage's deletes."""
    async with session.begin_nested():
        await update_chunks(
            session,
            [
                {"id": chunk.id, "embedding": vector}
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )


async def _embed_page(
    session: AsyncSession, page: Sequence[ChunkToEmbed], result: EmbedOutcome
) -> None:
    """Embed one page's batches concurrently, committing or recording each batch as it lands."""
    batches = list(_batch_by_document(page))
    async with _run_concurrently(batches, _embed_batch, limit=config.EMBED_CONCURRENCY) as pending:
        for (celex, chunks), vectors in pending:
            try:
                await _store_batch(session, chunks, await vectors)
            except (LLMError, SQLAlchemyError) as exc:
                result.fail(celex, exc, chunks=len(chunks))
            else:
                await session.commit()
                result.embedded += len(chunks)


async def embed_chunks(session: AsyncSession) -> EmbedOutcome:
    """Fill in every missing chunk vector: provider calls overlap within a page while writes
    serialize on the one session, and each batch commits as it lands, capping a failure's cost."""
    result = EmbedOutcome(already_embedded=await count_chunks(session, has_embedding=True))
    async for page in _unembedded_pages(session):
        await _embed_page(session, page, result)
    return result
