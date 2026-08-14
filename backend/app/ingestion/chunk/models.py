"""Chunk-stage values: the locator a chunk inherits, and the chunk itself."""

import hashlib
import json
from collections.abc import Sequence
from typing import ClassVar

from pydantic import computed_field

from app.core.models import FrozenModel
from app.ingestion.enums import SectionKind
from app.ingestion.parse.models import Section

DIVISIONS = (SectionKind.ARTICLE, SectionKind.ANNEX)
"""Section kinds that enclose a chunk: the innermost one is the division it belongs to."""


class Reference(FrozenModel):
    """One cross-reference; instrument is None when the target is this document."""

    raw: str
    instrument: str | None = None
    article: str | None = None
    paragraph: str | None = None
    annex: str | None = None


class Locator(FrozenModel):
    """Where a chunk sits: the division it belongs to, under the headings above it."""

    article: str | None = None
    annex: str | None = None
    title: str | None = None
    heading_path: tuple[str, ...] = ()


def locator_for(path: Sequence[Section]) -> Locator:
    """Where a section sits, read off its ancestors: the nearest article or annex encloses it,
    and every heading on the way down is part of the path."""
    headings = tuple(s.title for s in path if s.kind is SectionKind.HEADING and s.title)
    division = next((s for s in reversed(path) if s.kind in DIVISIONS), None)
    if division is None:
        return Locator(heading_path=headings)
    is_article = division.kind is SectionKind.ARTICLE
    return Locator(
        article=division.number if is_article else None,
        annex=None if is_article else division.number,
        title=division.title,
        heading_path=headings,
    )


class Chunk(Locator):
    """One retrievable unit of a regulation, with its citation and cross-references."""

    METADATA: ClassVar[set[str]] = {"citation", "position", "references", "topic"}
    """Fields outside content_hash: topic is provenance, position is placement, the rest
    derive from what is hashed."""

    celex: str
    topic: str
    kind: SectionKind
    text: str
    paragraph: str | None = None
    part: int = 1
    parts: int = 1
    position: int = 0
    references: tuple[Reference, ...] = ()

    @computed_field
    @property
    def citation(self) -> str:
        """The locator as a lawyer would cite it: 'Article 6(2)', 'Annex I'."""
        if self.article is not None:
            suffix = f"({self.paragraph})" if self.paragraph else ""
            return f"Article {self.article}{suffix}"
        if self.annex is not None:
            return f"Annex {self.annex}"
        return self.title or ""

    @property
    def content_hash(self) -> str:
        """Hash of every field bar the exclusions, so a new field counts towards identity."""
        return self._digest(self.model_dump(mode="json", exclude=self.METADATA))

    @property
    def metadata_hash(self) -> str:
        """Hash of the metadata fields, so drift detection compares one string per row."""
        return self._digest(self.model_dump(mode="json", include=self.METADATA))

    @staticmethod
    def _digest(payload: dict[str, object]) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ChunkQuery(FrozenModel):
    """Which chunks to select: by vector presence, keyset-paged for the embed sweep."""

    has_embedding: bool
    after: tuple[str, int] | None = None
    limit: int


class ChunkCounts(FrozenModel):
    """What reconciling chunks changed, for one document or summed across a run."""

    added: int = 0
    deleted: int = 0
    kept: int = 0
    updated: int = 0
