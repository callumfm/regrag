"""Section tree -> chunks: one chunk per leaf, carrying the legal locator it inherits."""

import re
from collections.abc import Iterator

from app.ingestion.chunk.models import Chunk, Locator
from app.ingestion.chunk.references import extract_references
from app.ingestion.constants import MAX_CHARS
from app.ingestion.parse.models import ParsedDocument, Section

CELL_SEPARATOR = " | "
SENTENCE = re.compile(r"(?<=[.;:])\s+")


def wrap(part: str, max_chars: int) -> list[str]:
    """Last resort for a part with no boundary left to split on: cut it to length."""
    return [part[start : start + max_chars] for start in range(0, len(part), max_chars)]


def pack(parts: list[str], max_chars: int, joiner: str) -> list[str]:
    """Greedily join parts, starting a new piece when max_chars would be exceeded."""
    packed: list[str] = []
    for part in parts:
        if packed and len(packed[-1]) + len(joiner) + len(part) <= max_chars:
            packed[-1] += joiner + part
        elif len(part) > max_chars:
            packed.extend(wrap(part, max_chars))
        else:
            packed.append(part)
    return packed


def segments(text: str, max_chars: int) -> list[str]:
    """Text as pieces within max_chars, splitting on line then sentence boundaries."""
    lines: list[str] = []
    for line in filter(None, text.split("\n")):
        if len(line) > max_chars:
            lines.extend(pack(SENTENCE.split(line), max_chars, " "))
        else:
            lines.append(line)
    return pack(lines, max_chars, "\n")


def table_segments(rows: tuple[tuple[str, ...], ...], max_chars: int) -> list[str]:
    """Table rows as pieces within max_chars, every piece led by the header row."""
    header, *body = (CELL_SEPARATOR.join(row) for row in rows)
    budget = max_chars - len(header) - 1
    if not body or budget <= 0:
        return segments("\n".join([header, *body]), max_chars)
    return [f"{header}\n{piece}" for piece in pack(body, budget, "\n")]


def pieces(section: Section, max_chars: int) -> list[str]:
    """A leaf's embeddable text, split to fit; tables repeat their header on every piece."""
    if section.rows:
        return table_segments(section.rows, max_chars)
    return segments(section.text, max_chars) if section.text else []


def walk(
    section: Section, document: ParsedDocument, locator: Locator, max_chars: int
) -> Iterator[Chunk]:
    """Emit a chunk per piece of this section's text, then recurse into its children."""
    inherited = locator.descend(section)
    split = pieces(section, max_chars)
    for index, piece in enumerate(split, start=1):
        yield Chunk.build(
            document,
            section,
            locator=inherited,
            text=piece,
            part=index,
            parts=len(split),
            references=extract_references(piece),
        )
    for child in section.children:
        yield from walk(child, document, inherited, max_chars)


def chunk_document(document: ParsedDocument, max_chars: int = MAX_CHARS) -> tuple[Chunk, ...]:
    """Every text-bearing section of the parsed tree as a chunk, in document order."""
    return tuple(
        chunk
        for section in document.sections
        for chunk in walk(section, document, Locator(), max_chars)
    )
