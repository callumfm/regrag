"""The fetch stage's record of one downloaded document."""

from datetime import datetime

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db.schema import BaseSchema
from app.ingestion.schemas import IngestRun

SUFFIX = ".html"


def object_key(celex: str, resolved_celex: str, sha256: str) -> str:
    """The one definition of where a document's bytes live in object storage.

    Keyed by content hash under its version, so no fetch ever overwrites the bytes an
    earlier parse ran against, and re-storing unchanged content is a no-op.
    """
    return f"{celex}/{resolved_celex}/{sha256}{SUFFIX}"


class RawDocument(BaseSchema):
    """One source document as fetched: its provenance, its bytes' fingerprint, its stored object."""

    __tablename__ = "raw_documents"
    __table_args__ = (UniqueConstraint("ingest_run_id", "celex"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    ingest_run_id: Mapped[int] = mapped_column(ForeignKey("ingest_runs.id", ondelete="CASCADE"))
    source: Mapped[str]
    celex: Mapped[str]
    resolved_celex: Mapped[str]
    topic: Mapped[str]
    url: Mapped[str]
    sha256: Mapped[str]
    size_bytes: Mapped[int]
    fetched_at: Mapped[datetime]

    run: Mapped[IngestRun] = relationship()

    @property
    def key(self) -> str:
        """Where this document's bytes are stored."""
        return object_key(self.celex, self.resolved_celex, self.sha256)
