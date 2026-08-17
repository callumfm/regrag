"""Text related to what search found: the division a citation names, or the whole
article a chunk sits in. Both read in reading order rather than by relevance."""

from collections.abc import Sequence

from sqlalchemy import Integer, Row, Select, cast, func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.schemas import DocumentChunk
from app.retrieval.models import ReferenceTarget, RetrievedChunk

CHUNK_COLUMNS = (
    DocumentChunk.id,
    DocumentChunk.celex,
    DocumentChunk.topic,
    DocumentChunk.citation,
    DocumentChunk.article,
    DocumentChunk.title,
    DocumentChunk.text,
    DocumentChunk.references,
)
"""What a caller sees of a chunk, without the vectors it is found by."""

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


def _article_of(chunk: RetrievedChunk) -> tuple[str, str] | None:
    """Which article a chunk sits in; None for an annex or anything outside one."""
    return (chunk.celex, chunk.article) if chunk.article is not None else None


async def _articles(
    session: AsyncSession, keys: Sequence[tuple[str, str]]
) -> dict[tuple[str, str], list[Row]]:
    """Every paragraph of each named article, grouped by article and in reading order."""
    stmt = (
        select(*CHUNK_COLUMNS)
        .where(tuple_(DocumentChunk.celex, DocumentChunk.article).in_(keys))
        .order_by(DocumentChunk.celex, DocumentChunk.article, *ARTICLE_ORDER)
    )
    grouped: dict[tuple[str, str], list[Row]] = {key: [] for key in keys}
    for row in await session.execute(stmt):
        grouped[(row.celex, row.article)].append(row)
    return grouped


async def expand_articles(
    session: AsyncSession, chunks: Sequence[RetrievedChunk]
) -> tuple[RetrievedChunk, ...]:
    """Each chunk widened to every paragraph of its article, at the rank it was found.

    A paragraph rarely restates its own subject — "the limit referred to in paragraph 1"
    is unreachable by relevance — so the article, not the paragraph, is the unit that
    answers. Chunks outside an article pass through as they came.
    """
    keys = list(dict.fromkeys(filter(None, map(_article_of, chunks))))
    grouped = await _articles(session, keys) if keys else {}
    expanded: list[RetrievedChunk] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        key = _article_of(chunk)
        if key is None:
            expanded.append(chunk)
        elif key not in seen:
            seen.add(key)
            expanded.extend(
                RetrievedChunk.model_validate(row, from_attributes=True) for row in grouped[key]
            )
    return tuple(expanded)
