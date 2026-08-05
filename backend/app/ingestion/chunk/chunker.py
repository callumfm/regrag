"""Section tree -> chunks: one chunk per leaf, carrying the legal locator it inherits."""

import re
from collections.abc import Iterator

from app.core.models import FrozenModel
from app.ingestion.chunk.references import Reference, extract_references
from app.ingestion.enums import SectionKind
from app.ingestion.parse.base import ParsedDocument, Section

CELL_SEPARATOR = " | "
MAX_CHARS = 2000
SENTENCE = re.compile(r"(?<=[.;:])\s+")


class Locator(FrozenModel):
    """Where a chunk sits in the document, accumulated on the way down the tree."""

    article: str | None = None
    annex: str | None = None
    title: str | None = None
    heading_path: tuple[str, ...] = ()


class Chunk(FrozenModel):
    """One retrievable unit of a regulation, with its citation and cross-references."""

    ref: str
    topic: str
    kind: SectionKind
    text: str
    citation: str
    article: str | None = None
    annex: str | None = None
    paragraph: str | None = None
    title: str | None = None
    heading_path: tuple[str, ...] = ()
    part: int = 1
    parts: int = 1
    references: tuple[Reference, ...] = ()


def citation(locator: Locator, paragraph: str | None) -> str:
    """The locator as a lawyer would cite it: 'Article 6(2)', 'Annex I'."""
    if locator.article is not None:
        suffix = f"({paragraph})" if paragraph else ""
        return f"Article {locator.article}{suffix}"
    if locator.annex is not None:
        return f"Annex {locator.annex}"
    return locator.title or ""


def descend(section: Section, locator: Locator) -> Locator:
    """Fold a section's identity into the locator its children inherit."""
    if section.kind is SectionKind.ARTICLE:
        return locator.model_copy(update={"article": section.number, "title": section.title})
    if section.kind is SectionKind.ANNEX:
        return locator.model_copy(update={"annex": section.number, "title": section.title})
    if section.kind is SectionKind.HEADING and section.title:
        return locator.model_copy(update={"heading_path": (*locator.heading_path, section.title)})
    return locator


def body(section: Section) -> str:
    """A leaf's embeddable text; table rows are flattened one row per line."""
    if section.rows:
        return "\n".join(CELL_SEPARATOR.join(row) for row in section.rows)
    return section.text


def pack(parts: list[str], max_chars: int, joiner: str) -> list[str]:
    """Greedily join parts, starting a new piece when max_chars would be exceeded."""
    packed: list[str] = []
    for part in parts:
        if packed and len(packed[-1]) + len(joiner) + len(part) <= max_chars:
            packed[-1] += joiner + part
        else:
            packed.append(part)
    return packed


def segments(text: str, max_chars: int) -> list[str]:
    """Text as pieces within max_chars, splitting on line then sentence boundaries."""
    lines: list[str] = []
    for line in (line for line in text.split("\n") if line):
        too_long = len(line) > max_chars
        lines.extend(pack(SENTENCE.split(line), max_chars, " ") if too_long else [line])
    return pack(lines, max_chars, "\n")


def build(
    document: ParsedDocument,
    section: Section,
    locator: Locator,
    text: str,
    part: int,
    parts: int,
) -> Chunk:
    paragraph = section.number if section.kind is SectionKind.PARAGRAPH else None
    return Chunk(
        ref=document.ref,
        topic=document.topic,
        kind=section.kind,
        text=text,
        citation=citation(locator, paragraph),
        article=locator.article,
        annex=locator.annex,
        paragraph=paragraph,
        title=locator.title,
        heading_path=locator.heading_path,
        part=part,
        parts=parts,
        references=extract_references(text),
    )


def walk(
    section: Section, document: ParsedDocument, locator: Locator, max_chars: int
) -> Iterator[Chunk]:
    """Emit a chunk per piece of this section's text, then recurse into its children."""
    inherited = descend(section, locator)
    if text := body(section):
        pieces = segments(text, max_chars)
        for index, piece in enumerate(pieces, start=1):
            yield build(document, section, inherited, piece, index, len(pieces))
    for child in section.children:
        yield from walk(child, document, inherited, max_chars)


def chunk_document(document: ParsedDocument, max_chars: int = MAX_CHARS) -> tuple[Chunk, ...]:
    """Every text-bearing section of the parsed tree as a chunk, in document order."""
    return tuple(
        chunk
        for section in document.sections
        for chunk in walk(section, document, Locator(), max_chars)
    )
