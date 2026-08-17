"""Widen a hit to the whole section it was cut from, so the piece that ranked
arrives with the text that gives it meaning."""

from collections.abc import Sequence

from sqlalchemy import Row, or_, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.schemas import DocumentChunk
from app.retrieval.models import CHUNK_COLUMNS, RetrievedChunk

DOCUMENT_ORDER = (DocumentChunk.position, DocumentChunk.part)
"""Reading order as the parser assigned it, which is the only order a section split for length
or an annex of prose has, and which matches an article's paragraph numbering throughout."""

SPLIT_GROUP = DocumentChunk.position - DocumentChunk.part
"""What the parts of one split section share: the chunker emits them at consecutive positions."""

SectionKey = tuple[str, str]
"""A section named by kind and value, so the three kinds cannot collide in one dict."""


def _section_key(row: Row) -> SectionKey:
    """The section a chunk was cut from: its article, the split it is a part of, or itself."""
    if row.article is not None:
        return ("article", f"{row.celex}:{row.article}")
    if row.parts > 1:
        return ("split", f"{row.celex}:{row.position - row.part}")
    return ("chunk", str(row.id))


async def _section_keys(session: AsyncSession, ids: Sequence[int]) -> list[SectionKey]:
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


def _matching(keys: Sequence[SectionKey]) -> list:
    """One clause per kind of section named, so each kind matches on the columns it is keyed by."""
    by_kind: dict[str, list[str]] = {"article": [], "split": [], "chunk": []}
    for kind, value in keys:
        by_kind[kind].append(value)
    clauses = []
    if by_kind["article"]:
        celex_article = [tuple(value.split(":", 1)) for value in by_kind["article"]]
        clauses.append(tuple_(DocumentChunk.celex, DocumentChunk.article).in_(celex_article))
    if by_kind["split"]:
        celex_start = [
            (celex, int(start))
            for celex, start in (value.split(":", 1) for value in by_kind["split"])
        ]
        clauses.append(tuple_(DocumentChunk.celex, SPLIT_GROUP).in_(celex_start))
    if by_kind["chunk"]:
        clauses.append(DocumentChunk.id.in_([int(value) for value in by_kind["chunk"]]))
    return clauses


async def _sections(
    session: AsyncSession, keys: Sequence[SectionKey]
) -> dict[SectionKey, list[Row]]:
    """Every chunk of each named section, grouped by section and in document order."""
    stmt = (
        select(*CHUNK_COLUMNS, DocumentChunk.position, DocumentChunk.part, DocumentChunk.parts)
        .where(or_(*_matching(keys)))
        .order_by(*DOCUMENT_ORDER)
    )
    grouped: dict[SectionKey, list[Row]] = {key: [] for key in keys}
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
