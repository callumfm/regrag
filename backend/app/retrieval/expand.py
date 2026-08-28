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


def _nearest_hit_distance(positions: Sequence[int]) -> ColumnElement[int]:
    """How far a stored chunk sits from the closest hit of its section."""
    gaps = [func.abs(DocumentChunk.position - position) for position in positions]
    return gaps[0] if len(gaps) == 1 else func.least(*gaps)


async def expand_sections(
    session: AsyncSession, chunks: Sequence[RetrievedChunk], *, limit: int
) -> tuple[RetrievedChunk, ...]:
    """Each chunk widened to the section it was cut from — every hit's own chunk kept
    first, then each section one chunk wider per round — so a hit is never evicted by
    the widening, at most `limit` chunks in all.

    A paragraph rarely restates its own subject — "the limit referred to in paragraph 1"
    is unreachable by relevance — and a section split for length leaves its halves adrift
    of each other. Retrieval ranks the pieces; the section is the unit that answers.
    What is kept still reads in rank then document order; distance is measured to the
    nearest hit, so a second hit in the same section counts as a hit, not as widening,
    though a tight cap may then leave a gap between two hits it kept.
    """
    if not chunks:
        return ()

    hit_positions: dict[SectionKey, list[int]] = {}
    for chunk in chunks:
        hit_positions.setdefault(SectionKey.from_chunk(chunk), []).append(chunk.position)

    filters = [key.query_filter() for key in hit_positions]
    section_rank = case(*((matches, order) for order, matches in enumerate(filters)))
    hit_distance = case(
        *(
            (matches, _nearest_hit_distance(positions))
            for matches, positions in zip(filters, hit_positions.values(), strict=True)
        )
    )
    labeled = (
        select(*CHUNK_COLUMNS, section_rank.label("rank"), hit_distance.label("distance"))
        .where(or_(*filters))
        .subquery()
    )
    round_number = func.row_number().over(
        partition_by=labeled.c.rank,
        order_by=(labeled.c.distance, labeled.c.position, labeled.c.part),
    )
    rounds = select(labeled, round_number.label("round")).subquery()
    stmt = (
        select(rounds).order_by(rounds.c.distance != 0, rounds.c.round, rounds.c.rank).limit(limit)
    )
    rows = await session.execute(stmt)
    reading_order = sorted(rows, key=lambda row: (row.rank, row.position, row.part))
    return tuple(RetrievedChunk.model_validate(row) for row in reading_order)
