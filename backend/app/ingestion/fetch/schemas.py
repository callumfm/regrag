"""The fetch stage's record of one downloaded document."""

from datetime import datetime
from pathlib import Path

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db.schema import BaseSchema
from app.ingestion.schemas import IngestRun


class RawDocument(BaseSchema):
    """One source document as fetched: its provenance, its bytes' fingerprint, its file."""

    __tablename__ = "raw_documents"
    __table_args__ = (UniqueConstraint("ingest_run_id", "ref"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    ingest_run_id: Mapped[int] = mapped_column(ForeignKey("ingest_runs.id", ondelete="CASCADE"))
    source: Mapped[str]
    ref: Mapped[str]
    resolved_ref: Mapped[str]
    topic: Mapped[str]
    url: Mapped[str]
    sha256: Mapped[str]
    size_bytes: Mapped[int]
    fetched_at: Mapped[datetime]

    run: Mapped[IngestRun] = relationship()

    @staticmethod
    def filename(ref: str) -> str:
        """The one definition of what a fetched document is called on disk."""
        return f"{ref}.html"

    def path(self, data_dir: Path) -> Path:
        return data_dir / self.filename(self.ref)
