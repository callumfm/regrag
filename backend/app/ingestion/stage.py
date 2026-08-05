"""The shape every ingest stage reports in: what it changed, and what it could not do."""

from typing import Any, Self

from pydantic import BaseModel, Field


class IngestStageDelta(BaseModel):
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
        merged: dict[str, Any] = {}
        for name in type(self).model_fields:
            mine, theirs = getattr(self, name), getattr(other, name)
            merged[name] = {**mine, **theirs} if isinstance(mine, dict) else mine + theirs
        return type(self)(**merged)
