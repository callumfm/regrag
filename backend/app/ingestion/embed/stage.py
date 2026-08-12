"""Embed stage: fill in the vector of every chunk that has none."""

from collections.abc import AsyncIterator, Iterator, Sequence
from itertools import batched, groupby
from operator import attrgetter

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm import EMBED_BATCH_SIZE, EmbedInput, LLMError, embed, llm_retry
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.chunk.service import count_embedded_chunks, get_unembedded_chunks
from app.ingestion.constants import EMBED_PAGE_SIZE
from app.ingestion.embed.models import EmbedOutcome


def _batches(chunks: Sequence[DocumentChunk]) -> Iterator[tuple[str, Sequence[DocumentChunk]]]:
    """Provider-sized batches that never span documents, each labelled with its document."""
    for celex, group in groupby(chunks, attrgetter("celex")):
        for batch in batched(group, EMBED_BATCH_SIZE):
            yield celex, batch


@llm_retry
async def _embed_texts(chunks: Sequence[DocumentChunk]) -> list[list[float]]:
    """Embed one batch, retrying transient provider failures."""
    return await embed([chunk.text for chunk in chunks], input_type=EmbedInput.DOCUMENT)


async def _embed_batch(session: AsyncSession, chunks: Sequence[DocumentChunk]) -> None:
    """Write each chunk's vector onto its row."""
    for chunk, vector in zip(chunks, await _embed_texts(chunks), strict=True):
        chunk.embedding = vector
    await session.flush()


async def _pages(session: AsyncSession) -> AsyncIterator[Sequence[DocumentChunk]]:
    """Vectorless chunks a page at a time, the cursor read before the page can be rolled back."""
    after: tuple[str, int] | None = None
    while page := await get_unembedded_chunks(session, after=after, limit=EMBED_PAGE_SIZE):
        after = (page[-1].celex, page[-1].id)
        yield page


async def embed_chunks(session: AsyncSession) -> EmbedOutcome:
    """Fill in every missing chunk vector, committing each batch as it lands.

    The savepoint is what a failed batch rolls back to: a plain rollback would expire the
    page's remaining rows, and embedding is the part that costs money to redo.
    """
    result = EmbedOutcome(already_embedded=await count_embedded_chunks(session))
    async for page in _pages(session):
        for celex, batch in _batches(page):
            try:
                async with session.begin_nested():
                    await _embed_batch(session, batch)
            except (LLMError, SQLAlchemyError) as exc:
                result.fail(celex, exc, chunks=len(batch))
            else:
                result.embedded += len(batch)
                await session.commit()
    return result
