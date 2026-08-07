"""Embed stage: fill in the vector of every chunk that has none."""

from collections.abc import Iterator, Sequence
from itertools import batched, groupby
from operator import attrgetter

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm import EmbedInput, LLMError, embed
from app.core.retry import transient_retry
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.constants import EMBED_BATCH_SIZE
from app.ingestion.embed.models import EmbedRunResult
from app.ingestion.embed.service import count_embedded_chunks, get_unembedded_chunks


def batches(chunks: Sequence[DocumentChunk]) -> Iterator[tuple[str, Sequence[DocumentChunk]]]:
    """Provider-sized batches that never span documents, each labelled with its document."""
    for celex, group in groupby(chunks, attrgetter("celex")):
        for batch in batched(group, EMBED_BATCH_SIZE):
            yield celex, batch


@transient_retry(lambda exc: isinstance(exc, LLMError) and exc.transient)
async def embed_texts(chunks: Sequence[DocumentChunk]) -> list[list[float]]:
    """Embed one batch, retrying transient provider failures."""
    return await embed([chunk.text for chunk in chunks], input_type=EmbedInput.DOCUMENT)


async def _embed_batch(session: AsyncSession, chunks: Sequence[DocumentChunk]) -> None:
    """Write each chunk's vector onto its row."""
    for chunk, vector in zip(chunks, await embed_texts(chunks), strict=True):
        chunk.embedding = vector
    await session.flush()


async def embed_chunks(session: AsyncSession) -> EmbedRunResult:
    """Fill in every missing chunk vector; a batch that fails is recorded against its document."""
    result = EmbedRunResult(unchanged=await count_embedded_chunks(session))
    for celex, batch in batches(await get_unembedded_chunks(session)):
        try:
            async with session.begin_nested():
                await _embed_batch(session, batch)
            result.embedded += len(batch)
        except (LLMError, SQLAlchemyError) as exc:
            result.failed[celex] = f"{type(exc).__name__}: {exc}"
    return result
