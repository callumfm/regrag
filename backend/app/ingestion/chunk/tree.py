"""Section tree -> chunks: one chunk per piece of a section's text, addressed by its ancestors."""

from collections.abc import Iterator

from app.core.config import config
from app.ingestion.chunk.models import Chunk, locator_for
from app.ingestion.chunk.references import extract_references
from app.ingestion.chunk.split import split_section_text
from app.ingestion.enums import SectionKind
from app.ingestion.parse.models import ParsedDocument, Section


def chunk_section_tree(
    section: Section, document: ParsedDocument, ancestors: tuple[Section, ...], max_chars: int
) -> Iterator[Chunk]:
    """This section's own text as chunks, then everything nested beneath it."""
    path = (*ancestors, section)
    locator = locator_for(path)
    pieces = split_section_text(section, max_chars)
    for part, piece in enumerate(pieces, start=1):
        yield Chunk(
            **locator.model_dump(),
            celex=document.celex,
            topic=document.topic,
            kind=section.kind,
            paragraph=section.number if section.kind is SectionKind.PARAGRAPH else None,
            text=piece,
            part=part,
            parts=len(pieces),
            references=extract_references(piece),
        )
    for child in section.children:
        yield from chunk_section_tree(child, document, path, max_chars)


def chunk_document(
    document: ParsedDocument, max_chars: int = config.MAX_CHARS
) -> tuple[Chunk, ...]:
    """Every text-bearing section of the parsed tree as a chunk, numbered in document order."""
    chunks = (
        chunk
        for section in document.sections
        for chunk in chunk_section_tree(section, document, (), max_chars)
    )
    numbered_chunks = (
        chunk.model_copy(update={"position": position}) for position, chunk in enumerate(chunks)
    )
    return tuple(numbered_chunks)
