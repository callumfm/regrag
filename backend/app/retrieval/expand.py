"""Widen a hit to the whole section it was cut from, so the piece that ranked
arrives with the text that gives it meaning."""

from collections.abc import Sequence

from sqlalchemy import ColumnElement, Row, or_, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.schemas import DocumentChunk
from app.retrieval.models import CHUNK_COLUMNS, RetrievedChunk


def _section_key(row: Row) -> tuple:
    """The section a chunk was cut from: its article, the split it is a part of, or itself.

    Parts of one split share position - part, since the chunker emits them back to back.
    """
    if row.article is not None:
        return ("article", row.celex, row.article)
    if row.parts > 1:
        return ("split", row.celex, row.position - row.part)
    return ("chunk", row.id)


async def _section_keys(session: AsyncSession, ids: Sequence[int]) -> list[tuple]:
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


def _matching(keys: Sequence[tuple]) -> list[ColumnElement[bool]]:
    """One clause per kind of section named, each matching on the columns that kind is keyed by."""
    articles = [key[1:] for key in keys if key[0] == "article"]
    splits = [key[1:] for key in keys if key[0] == "split"]
    chunks = [key[1] for key in keys if key[0] == "chunk"]
    clauses: list[ColumnElement[bool]] = []
    if articles:
        clauses.append(tuple_(DocumentChunk.celex, DocumentChunk.article).in_(articles))
    if splits:
        group = DocumentChunk.position - DocumentChunk.part
        clauses.append(tuple_(DocumentChunk.celex, group).in_(splits))
    if chunks:
        clauses.append(DocumentChunk.id.in_(chunks))
    return clauses


async def _sections(session: AsyncSession, keys: Sequence[tuple]) -> dict[tuple, list[Row]]:
    """Every chunk of each named section, grouped by section and in the order the parser read it."""
    stmt = (
        select(*CHUNK_COLUMNS, DocumentChunk.position, DocumentChunk.part, DocumentChunk.parts)
        .where(or_(*_matching(keys)))
        .order_by(DocumentChunk.position, DocumentChunk.part)
    )
    grouped: dict[tuple, list[Row]] = {key: [] for key in keys}
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
