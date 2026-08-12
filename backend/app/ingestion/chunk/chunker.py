"""Section tree -> chunks: one chunk per leaf, carrying the legal locator it inherits."""

from collections.abc import Iterator

from app.core.config import config
from app.ingestion.chunk.models import Chunk, Locator
from app.ingestion.chunk.references import extract_references
from app.ingestion.chunk.split import split_section_text
from app.ingestion.enums import SectionKind
from app.ingestion.parse.models import ParsedDocument, Section


def iter_section_chunks(
    section: Section, document: ParsedDocument, locator: Locator, max_chars: int
) -> Iterator[Chunk]:
    """Emit a chunk per piece of this section's text, then recurse into its children."""
    inherited = locator.with_section(section)
    pieces = split_section_text(section, max_chars)
    for index, piece in enumerate(pieces, start=1):
        yield Chunk(
            **inherited.model_dump(),
            celex=document.celex,
            topic=document.topic,
            kind=section.kind,
            paragraph=section.number if section.kind is SectionKind.PARAGRAPH else None,
            text=piece,
            part=index,
            parts=len(pieces),
            references=extract_references(piece),
        )
    for child in section.children:
        yield from iter_section_chunks(child, document, inherited, max_chars)


def chunk_document(
    document: ParsedDocument, max_chars: int = config.MAX_CHARS
) -> tuple[Chunk, ...]:
    """Every text-bearing section of the parsed tree as a chunk, in document order."""
    return tuple(
        chunk
        for section in document.sections
        for chunk in iter_section_chunks(section, document, Locator(), max_chars)
    )
