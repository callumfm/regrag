"""Chunk-stage values: the locator a chunk inherits, and the chunk itself."""

from typing import Self

from pydantic import computed_field

from app.core.models import FrozenModel
from app.ingestion.enums import SectionKind
from app.ingestion.parse.models import ParsedDocument, Section


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

    def descend(self, section: Section) -> Self:
        """Fold a section's identity into the locator its children inherit."""
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

    ref: str
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

    @classmethod
    def build(
        cls,
        document: ParsedDocument,
        section: Section,
        locator: Locator,
        text: str,
        part: int,
        parts: int,
        references: tuple[Reference, ...] = (),
    ) -> "Chunk":
        """One chunk of a section's text, carrying the locator it sits under."""
        return cls(
            **locator.model_dump(),
            ref=document.ref,
            topic=document.topic,
            kind=section.kind,
            text=text,
            paragraph=section.number if section.kind is SectionKind.PARAGRAPH else None,
            part=part,
            parts=parts,
            references=references,
        )
