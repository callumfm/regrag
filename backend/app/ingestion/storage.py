"""Where a fetched document's bytes live: the only module that reads or writes them."""

import hashlib
from pathlib import Path

from app.ingestion.exceptions import EmptyDownloadError


def document_filename(celex: str) -> str:
    """The one definition of what a fetched document is called in storage."""
    return f"{celex}.html"


def _document_path(data_dir: Path, celex: str) -> Path:
    return data_dir / document_filename(celex)


def write_document(data_dir: Path, celex: str, content: bytes) -> tuple[str, int]:
    """Store the document's bytes and return their (sha256, size_bytes), refusing empty
    content so a failed download cannot overwrite the last good copy."""
    if not content:
        raise EmptyDownloadError(f"{celex}: download returned an empty body")
    data_dir.mkdir(parents=True, exist_ok=True)
    _document_path(data_dir, celex).write_bytes(content)
    return hashlib.sha256(content).hexdigest(), len(content)


def read_document(data_dir: Path, celex: str) -> bytes:
    """The bytes stored for a document the run recorded."""
    return _document_path(data_dir, celex).read_bytes()


def document_exists(data_dir: Path, celex: str) -> bool:
    """Whether a document the run recorded still has its bytes stored."""
    return _document_path(data_dir, celex).is_file()
