"""Format-neutral parser IR: every parser produces a ParsedDocument section tree."""

import re
from dataclasses import dataclass
from typing import Protocol

from app.ingestion.enums import SectionKind

FORMULA_PLACEHOLDER = "[formula]"
WHITESPACE = re.compile(r"\s+")


class ParseError(Exception):
    """A document could not be parsed into a section tree."""


@dataclass(frozen=True)
class Section:
    """One node of a document tree; rows is populated only for TABLE."""

    kind: SectionKind
    number: str | None = None
    title: str | None = None
    text: str = ""
    rows: tuple[tuple[str, ...], ...] = ()
    children: tuple["Section", ...] = ()


@dataclass(frozen=True)
class ParsedDocument:
    """A parsed act: identity from the ingest record, body as a section tree."""

    ref: str
    topic: str
    sections: tuple[Section, ...]


class Parser(Protocol):
    def __call__(self, html: str, ref: str, topic: str) -> ParsedDocument: ...


def normalise(text: str) -> str:
    """Collapse whitespace, including the non-breaking spaces EUR-Lex indents with."""
    return WHITESPACE.sub(" ", text.replace("\xa0", " ")).strip()
