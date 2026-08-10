"""Ingestion values every stage shares: what a stage changed, and what it could not do."""

from datetime import datetime
from typing import Any, ClassVar, Self

from pydantic import BaseModel, Field

from app.ingestion.enums import IngestRunStatus


class StageRunResult(BaseModel):
    """One ingest stage's outcome; every field a subclass declares is a reported bucket."""

    UNCOUNTED: ClassVar[frozenset[str]] = frozenset({"failed"})
    MAX_FAILURE_CHARS: ClassVar[int] = 500

    failed: dict[str, str] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed

    def fail(self, celex: str, exc: Exception) -> None:
        """Record why a document could not be processed, in the one format details() prints."""
        self.failed[celex] = f"{type(exc).__name__}: {exc}"

    def counts(self) -> dict[str, int]:
        """This stage's outcome buckets in declaration order; a list bucket counts its members."""
        buckets = ((name, getattr(self, name)) for name in type(self).model_fields)
        return {
            name: len(value) if isinstance(value, list) else value
            for name, value in buckets
            if name not in self.UNCOUNTED
        }

    def summary(self) -> str:
        """One line of counts, always closing with the failure count."""
        counted = [f"{value} {label}" for label, value in self.counts().items()]
        return ", ".join([*counted, f"{len(self.failed)} failed"])

    def details(self) -> list[str]:
        """The per-document lines the summary is too short to carry."""
        return [f"failed: {celex} ({error})" for celex, error in sorted(self.failed.items())]

    def report(self) -> dict[str, Any]:
        """This stage as the run row stores it: its counts, plus why each document failed."""
        return {
            **self.counts(),
            "failed": {
                celex: error[: self.MAX_FAILURE_CHARS] for celex, error in self.failed.items()
            },
        }

    def __add__(self, other: Self) -> Self:
        """Combine same-type operands field by field; fields must be int, list or dict."""
        if type(other) is not type(self):
            return NotImplemented
        merged: dict[str, Any] = {}
        for name in type(self).model_fields:
            mine, theirs = getattr(self, name), getattr(other, name)
            if isinstance(mine, dict):
                merged[name] = {**mine, **theirs}
            elif isinstance(mine, list | int) and not isinstance(mine, bool):
                merged[name] = mine + theirs
            else:
                raise TypeError(
                    f"{type(self).__name__}.{name}: result fields must be int, list or dict"
                )
        return type(self)(**merged)


class IngestRunUpdate(BaseModel):
    """Partial update body for an ingest run."""

    status: IngestRunStatus | None = None
    corpus_version: str | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
