"""Text related to what search found: the division a citation names, or the whole
article a chunk sits in. Both read in reading order rather than by relevance."""

from collections.abc import Sequence

from sqlalchemy import Integer, Row, Select, cast, func, or_, select, tuple_
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


SPLIT_GROUP = DocumentChunk.position - DocumentChunk.part
"""What the parts of one split section share: the chunker emits them at consecutive positions."""

SECTION_ORDER = (DocumentChunk.position, DocumentChunk.part)
"""Document order: it matches ARTICLE_ORDER for an article, and is a split's parts in turn."""


def _section_key(row: Row) -> tuple[str, str]:
    """The section a chunk was cut from: its article, the split it is a part of, or itself."""
    if row.article is not None:
        return ("article", f"{row.celex}:{row.article}")
    if row.parts > 1:
        return ("split", f"{row.celex}:{row.position - row.part}")
    return ("chunk", str(row.id))


async def _section_keys(session: AsyncSession, ids: Sequence[int]) -> list[tuple[str, str]]:
    """The section each hit belongs to, in the order the hits ranked and without repeats."""
    stmt = select(
        DocumentChunk.id,
        DocumentChunk.celex,
        DocumentChunk.article,
        DocumentChunk.position,
        DocumentChunk.part,
        DocumentChunk.parts,
    ).where(DocumentChunk.id.in_(ids))
    found = {row.id: _section_key(row) for row in await session.execute(stmt)}
    return list(dict.fromkeys(found[chunk_id] for chunk_id in ids if chunk_id in found))


async def _sections(
    session: AsyncSession, keys: Sequence[tuple[str, str]]
) -> dict[tuple[str, str], list[Row]]:
    """Every chunk of each named section, grouped by section and in document order."""
    by_kind: dict[str, list[str]] = {"article": [], "split": [], "chunk": []}
    for kind, value in keys:
        by_kind[kind].append(value)
    clauses = []
    if by_kind["article"]:
        pairs = [tuple(value.split(":", 1)) for value in by_kind["article"]]
        clauses.append(tuple_(DocumentChunk.celex, DocumentChunk.article).in_(pairs))
    if by_kind["split"]:
        pairs = [
            (celex, int(start))
            for celex, start in (value.split(":", 1) for value in by_kind["split"])
        ]
        clauses.append(tuple_(DocumentChunk.celex, SPLIT_GROUP).in_(pairs))
    if by_kind["chunk"]:
        clauses.append(DocumentChunk.id.in_([int(value) for value in by_kind["chunk"]]))
    stmt = (
        select(*CHUNK_COLUMNS, DocumentChunk.position, DocumentChunk.part, DocumentChunk.parts)
        .where(or_(*clauses))
        .order_by(*SECTION_ORDER)
    )
    grouped: dict[tuple[str, str], list[Row]] = {key: [] for key in keys}
    for row in await session.execute(stmt):
        grouped[_section_key(row)].append(row)
    return grouped


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
    keys = await _section_keys(session, [chunk.id for chunk in chunks])
    sections = await _sections(session, keys)
    return tuple(
        RetrievedChunk.model_validate(row, from_attributes=True)
        for key in keys
        for row in sections[key]
    )
