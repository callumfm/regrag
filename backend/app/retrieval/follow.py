"""Follow a stored cross-reference to the text it cites, in reading order."""

from sqlalchemy import Integer, Select, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.schemas import DocumentChunk
from app.retrieval.models import CHUNK_COLUMNS, ReferenceTarget, RetrievedChunk

ARTICLE_ORDER = (
    DocumentChunk.paragraph.is_not(None),
    cast(func.substring(DocumentChunk.paragraph, "^[0-9]+"), Integer),
    func.substring(DocumentChunk.paragraph, "^[0-9]*(.*)$"),
    DocumentChunk.part,
)
"""The chapeau leads its paragraphs, since a NULL paragraph sorts False before their True;
the number sorts numerically and the letter after it, so 2 precedes 11 precedes 11a."""

ANNEX_ORDER = (DocumentChunk.position, DocumentChunk.part)
"""An annex numbers no paragraphs, so where it sits in the document is the only order it has."""


def _targeted(stmt: Select, target: ReferenceTarget) -> Select:
    """Narrow to the one division the target names."""
    stmt = stmt.where(DocumentChunk.celex == target.celex)
    if target.article is not None:
        stmt = stmt.where(func.lower(DocumentChunk.article) == target.article.lower())
    if target.paragraph is not None:
        stmt = stmt.where(DocumentChunk.paragraph == target.paragraph)
    if target.annex is not None:
        stmt = stmt.where(DocumentChunk.annex == target.annex)
    return stmt


async def follow_reference(
    session: AsyncSession, target: ReferenceTarget
) -> tuple[RetrievedChunk, ...]:
    """The text a stored cross-reference points at, in reading order."""
    order = ANNEX_ORDER if target.annex is not None else ARTICLE_ORDER
    stmt = _targeted(select(*CHUNK_COLUMNS), target).order_by(*order)

    rows = await session.execute(stmt)
    return tuple(RetrievedChunk.model_validate(row, from_attributes=True) for row in rows)
