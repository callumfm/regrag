"""Ingestion values every stage shares: what a stage changed, and what it could not do."""

from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, Field

from app.ingestion.enums import IngestRunStatus


class StageRunResult(BaseModel):
    """One ingest stage's outcome; subclasses declare their buckets and fill counts()."""

    failed: dict[str, str] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed

    def counts(self) -> dict[str, int]:
        """This stage's outcome buckets, in reporting order."""
        raise NotImplementedError

    def summary(self) -> str:
        """One line of counts, always closing with the failure count."""
        counted = [f"{value} {label}" for label, value in self.counts().items()]
        return ", ".join([*counted, f"{len(self.failed)} failed"])

    def details(self) -> list[str]:
        """The per-ref lines the summary is too short to carry."""
        return [f"failed: {ref} ({error})" for ref, error in sorted(self.failed.items())]

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
