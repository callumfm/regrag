"""Widen a hit to the whole section it was cut from, so the piece that ranked
arrives with the text that gives it meaning."""

from collections.abc import Sequence

from sqlalchemy import ColumnElement, and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import FrozenModel
from app.ingestion.chunk.schemas import DocumentChunk
from app.retrieval.models import CHUNK_COLUMNS, RetrievedChunk


class SectionKey(FrozenModel):
    """Which section a chunk was cut from: its article, the split it is a part of, or itself."""

    celex: str
    article: str | None = None
    split_start: int | None = None
    chunk_id: int | None = None

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk) -> "SectionKey":
        if chunk.article is not None:
            return cls(celex=chunk.celex, article=chunk.article)
        if chunk.parts > 1:
            return cls(celex=chunk.celex, split_start=chunk.position - chunk.part + 1)
        return cls(celex=chunk.celex, chunk_id=chunk.id)

    def query_filter(self) -> ColumnElement[bool]:
        """Matches every stored chunk of this section."""
        if self.article is not None:
            return and_(DocumentChunk.celex == self.celex, DocumentChunk.article == self.article)
        if self.split_start is not None:
            start = DocumentChunk.position - DocumentChunk.part + 1
            return and_(DocumentChunk.celex == self.celex, start == self.split_start)
        return DocumentChunk.id == self.chunk_id


async def expand_sections(
    session: AsyncSession, chunks: Sequence[RetrievedChunk]
) -> tuple[RetrievedChunk, ...]:
    """Each chunk widened to the section it was cut from, at the rank it was found.

    A paragraph rarely restates its own subject — "the limit referred to in paragraph 1"
    is unreachable by relevance — and a section split for length leaves its halves adrift
    of each other. Retrieval ranks the pieces; the section is the unit that answers.
    """
    if not chunks:
        return ()

    keys = list(dict.fromkeys(SectionKey.from_chunk(chunk) for chunk in chunks))
    stmt = (
        select(*CHUNK_COLUMNS)
        .where(or_(*(key.query_filter() for key in keys)))
        .order_by(DocumentChunk.position, DocumentChunk.part)
    )
    sections: dict[SectionKey, list[RetrievedChunk]] = {key: [] for key in keys}
    for row in await session.execute(stmt):
        chunk = RetrievedChunk.model_validate(row, from_attributes=True)
        sections[SectionKey.from_chunk(chunk)].append(chunk)
    return tuple(chunk for key in keys for chunk in sections[key])
