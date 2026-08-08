"""Where a fetched document's bytes live: the only module that reads or writes them."""

import hashlib
from pathlib import Path

from app.ingestion.exceptions import EmptyDownloadError
from app.ingestion.fetch.schemas import RawDocument


def write_document(data_dir: Path, celex: str, content: bytes) -> tuple[str, int]:
    """Store the document's bytes and return their (sha256, size_bytes).

    Empty content is refused: it would overwrite the last good copy with nothing.
    """
    if not content:
        raise EmptyDownloadError(f"{celex}: download returned an empty body")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / RawDocument.filename(celex)).write_bytes(content)
    return hashlib.sha256(content).hexdigest(), len(content)


def read_document(data_dir: Path, document: RawDocument) -> bytes:
    """The bytes stored for a document the run recorded."""
    return document.path(data_dir).read_bytes()


def document_exists(data_dir: Path, document: RawDocument) -> bool:
    """Whether a document the run recorded still has its bytes stored."""
    return document.path(data_dir).is_file()
