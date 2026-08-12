"""Chunk-stage values: the locator a chunk inherits, and the chunk itself."""

import hashlib
import json
from typing import ClassVar, Self

from pydantic import computed_field

from app.core.models import FrozenModel
from app.ingestion.enums import SectionKind
from app.ingestion.parse.models import Section


class Reference(FrozenModel):
    """One cross-reference; instrument is None when the target is this document."""

    raw: str
    instrument: str | None = None
    article: str | None = None
    paragraph: str | None = None
    annex: str | None = None


class Locator(FrozenModel):
    """Where a chunk sits in the document, accumulated on the way down the tree."""

    article: str | None = None
    annex: str | None = None
    title: str | None = None
    heading_path: tuple[str, ...] = ()

    def with_section(self, section: Section) -> Self:
        """This locator with a section's identity folded in; its children inherit the result."""
        if section.kind is SectionKind.ARTICLE:
            return self.model_copy(
                update={"article": section.number, "annex": None, "title": section.title}
            )
        if section.kind is SectionKind.ANNEX:
            return self.model_copy(
                update={"annex": section.number, "article": None, "title": section.title}
            )
        if section.kind is SectionKind.HEADING and section.title:
            return self.model_copy(update={"heading_path": (*self.heading_path, section.title)})
        return self


class Chunk(Locator):
    """One retrievable unit of a regulation, with its citation and cross-references."""

    NOT_IDENTITY: ClassVar[set[str]] = {"topic", "citation", "references"}
    """Fields outside content_hash: topic is provenance, the rest derive from what is hashed."""

    celex: str
    topic: str
    kind: SectionKind
    text: str
    paragraph: str | None = None
    part: int = 1
    parts: int = 1
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
        payload = self.model_dump(mode="json", exclude=self.NOT_IDENTITY)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ChunkCounts(FrozenModel):
    """What reconciling chunks changed, for one document or summed across a run."""

    added: int = 0
    deleted: int = 0
    kept: int = 0
