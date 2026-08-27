"""Widen a hit to the whole section it was cut from, so the piece that ranked
arrives with the text that gives it meaning."""

from collections.abc import Sequence

from sqlalchemy import ColumnElement, and_, case, func, or_, select
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
    session: AsyncSession, chunks: Sequence[RetrievedChunk], *, limit: int
) -> tuple[RetrievedChunk, ...]:
    """Each chunk widened to the section it was cut from, selected in rounds — every
    hit's own chunk first, then each section one chunk wider — so a hit is never
    evicted by a longer section above it, at most `limit` chunks in all.

    A paragraph rarely restates its own subject — "the limit referred to in paragraph 1"
    is unreachable by relevance — and a section split for length leaves its halves adrift
    of each other. Retrieval ranks the pieces; the section is the unit that answers.
    Sections still arrive contiguous and in rank order: rounds decide what is kept,
    not how it reads; a distance tie widens backward first, the chapeau before what
    follows.
    """
    if not chunks:
        return ()

    anchors: dict[SectionKey, int] = {}
    for chunk in chunks:
        anchors.setdefault(SectionKey.from_chunk(chunk), chunk.position)
    filters = [key.query_filter() for key in anchors]
    rank = case(*((matches, position) for position, matches in enumerate(filters)))
    anchor = case(
        *((matches, position) for matches, position in zip(filters, anchors.values(), strict=True))
    )
    distance = func.abs(DocumentChunk.position - anchor)
    round_number = func.row_number().over(
        partition_by=rank, order_by=(distance, DocumentChunk.position, DocumentChunk.part)
    )
    ranked = (
        select(*CHUNK_COLUMNS, rank.label("rank"), round_number.label("round"))
        .where(or_(*filters))
        .subquery()
    )
    kept = select(ranked).order_by(ranked.c.round, ranked.c.rank).limit(limit).subquery()
    stmt = select(*(kept.c[name] for name in RetrievedChunk.model_fields)).order_by(
        kept.c.rank, kept.c.position, kept.c.part
    )
    rows = await session.execute(stmt)
    return tuple(RetrievedChunk.model_validate(row) for row in rows)
