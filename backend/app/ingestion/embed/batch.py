"""One batch of vectorless chunks: how they are grouped, embedded, and written back."""

from collections.abc import AsyncIterator, Sequence
from itertools import batched, groupby
from operator import attrgetter
from typing import NamedTuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import config
from app.core.llm import EMBED_BATCH_SIZE, EmbedInput, embed, llm_retry
from app.ingestion.chunk.models import ChunkQuery
from app.ingestion.chunk.service import get_chunks, update_chunks


class ChunkToEmbed(NamedTuple):
    """One vectorless chunk, detached from the ORM so a mid-page commit cannot expire it."""

    id: int
    celex: str
    text: str


Batch = tuple[str, Sequence[ChunkToEmbed]]
Vectors = list[list[float]]


def _batch_by_document(page: Sequence[ChunkToEmbed]) -> list[Batch]:
    """Provider-sized batches that never span documents, each labelled with its document."""
    return [
        (celex, batch)
        for celex, group in groupby(page, attrgetter("celex"))
        for batch in batched(group, EMBED_BATCH_SIZE)
    ]


async def iter_batch_pages(session: AsyncSession) -> AsyncIterator[list[Batch]]:
    """A page of vectorless chunks at a time as batches, the cursor read before the page can be
    rolled back."""
    after: tuple[str, int] | None = None
    while page := await get_chunks(
        session, ChunkQuery(has_embedding=False, after=after, limit=config.EMBED_PAGE_SIZE)
    ):
        after = (page[-1].celex, page[-1].id)
        yield _batch_by_document(
            [ChunkToEmbed(chunk.id, chunk.celex, chunk.text) for chunk in page]
        )


@llm_retry
async def embed_batch(batch: Batch) -> Vectors:
    """One provider call embedding one batch's texts, retrying transient failures."""
    _, chunks = batch
    return await embed([chunk.text for chunk in chunks], input_type=EmbedInput.DOCUMENT)


async def store_batch(
    session: AsyncSession, chunks: Sequence[ChunkToEmbed], vectors: Vectors
) -> None:
    """Write one batch's vectors inside a savepoint: a plain rollback on failure would drag
    down earlier uncommitted work, like the prune stage's deletes."""
    async with session.begin_nested():
        updates = [
            {"id": chunk.id, "embedding": vector}
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        await update_chunks(session, updates)
