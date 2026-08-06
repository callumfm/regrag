"""Format-neutral parser IR: every parser produces a ParsedDocument section tree."""

from typing import Protocol

from pydantic import Field

from app.core.models import FrozenModel
from app.ingestion.enums import SectionKind
from app.ingestion.stage import StageRunResult


class Section(FrozenModel):
    """One node of a document tree; rows is populated only for TABLE."""

    kind: SectionKind
    number: str | None = None
    title: str | None = None
    text: str = ""
    rows: tuple[tuple[str, ...], ...] = ()
    children: tuple["Section", ...] = ()


class ParsedDocument(FrozenModel):
    """A parsed act: identity from the ingest record, body as a section tree."""

    ref: str
    topic: str
    sections: tuple[Section, ...]


class Parser(Protocol):
    def __call__(self, html: str, ref: str, topic: str) -> ParsedDocument: ...


class ParseRunResult(StageRunResult):
    """Which fetched documents yielded a section tree."""

    parsed: list[str] = Field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {"parsed": len(self.parsed)}
