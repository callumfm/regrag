"""Embed stage: fill in the vector of every chunk that has none."""

from collections.abc import Sequence

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.concurrency import run_concurrently
from app.core.config import config
from app.core.llm import LLMError
from app.ingestion.chunk.service import count_chunks
from app.ingestion.embed.batch import Batch, embed_batch, iter_batch_pages, store_batch
from app.ingestion.embed.models import EmbedOutcome


async def _embed_page(
    session: AsyncSession, batches: Sequence[Batch], result: EmbedOutcome
) -> None:
    """Embed one page's batches concurrently, committing or recording each batch as it lands."""
    async with run_concurrently(batches, embed_batch, limit=config.EMBED_CONCURRENCY) as pending:
        for (celex, chunks), vectors in pending:
            try:
                await store_batch(session, chunks, await vectors)
            except (LLMError, SQLAlchemyError) as exc:
                result.fail(celex, exc, chunks=len(chunks))
            else:
                await session.commit()
                result.embedded += len(chunks)


async def embed_chunks(session: AsyncSession) -> EmbedOutcome:
    """Fill in every missing chunk vector: provider calls overlap within a page while writes
    serialize on the one session, and each batch commits as it lands, capping a failure's cost."""
    already_embedded_count = await count_chunks(session, has_embedding=True)
    result = EmbedOutcome(already_embedded=already_embedded_count)
    async for batches in iter_batch_pages(session):
        await _embed_page(session, batches, result)
    return result
