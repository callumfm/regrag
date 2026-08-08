"""Text sizing: a section's text as pieces that fit the embedding budget."""

import re

from app.ingestion.parse.models import Section

CELL_SEPARATOR = " | "
SENTENCE = re.compile(r"(?<=[.;:])\s+")


def cut_to_max_chars(part: str, max_chars: int) -> list[str]:
    """Last resort for a part with no boundary left to split on."""
    return [part[start : start + max_chars] for start in range(0, len(part), max_chars)]


def pack_parts(parts: list[str], max_chars: int, joiner: str) -> list[str]:
    """Greedily join parts, starting a new one when max_chars would be exceeded."""
    packed: list[str] = []
    for part in parts:
        if packed and len(packed[-1]) + len(joiner) + len(part) <= max_chars:
            packed[-1] += joiner + part
        elif len(part) > max_chars:
            packed.extend(cut_to_max_chars(part, max_chars))
        else:
            packed.append(part)
    return packed


def split_text(text: str, max_chars: int) -> list[str]:
    """Split on line boundaries, falling back to sentence boundaries inside an overlong line."""
    lines: list[str] = []
    for line in filter(None, text.split("\n")):
        if len(line) > max_chars:
            lines.extend(pack_parts(SENTENCE.split(line), max_chars, " "))
        else:
            lines.append(line)
    return pack_parts(lines, max_chars, "\n")


def split_table_rows(rows: tuple[tuple[str, ...], ...], max_chars: int) -> list[str]:
    """Split on row boundaries, leading every piece with the header row."""
    header, *body = (CELL_SEPARATOR.join(row) for row in rows)
    budget = max_chars - len(header) - 1
    if not body or budget <= 0:
        return split_text("\n".join([header, *body]), max_chars)
    return [f"{header}\n{piece}" for piece in pack_parts(body, budget, "\n")]


def split_section_text(section: Section, max_chars: int) -> list[str]:
    """A leaf's embeddable text; a section with neither rows nor text yields nothing."""
    if section.rows:
        return split_table_rows(section.rows, max_chars)
    return split_text(section.text, max_chars) if section.text else []
