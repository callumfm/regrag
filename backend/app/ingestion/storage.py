"""Where a fetched document's bytes live: the only module that reads or writes them."""

import hashlib

from app.core.storage import ObjectStore
from app.ingestion.exceptions import EmptyDownloadError
from app.ingestion.fetch.schemas import RawDocument


def document_key(celex: str, resolved_celex: str, sha256: str) -> str:
    """Where a document's bytes live, keyed by content so a new version never overwrites a
    version an earlier parse ran against."""
    return f"{celex}/{resolved_celex}/{sha256}.html"


def write_document(
    store: ObjectStore, celex: str, resolved_celex: str, content: bytes
) -> tuple[str, int]:
    """Store the document's bytes and return their (sha256, size_bytes), refusing empty
    content so a failed download cannot be recorded as a version."""
    if not content:
        raise EmptyDownloadError(f"{celex}: download returned an empty body")
    sha256 = hashlib.sha256(content).hexdigest()
    store.put(document_key(celex, resolved_celex, sha256), content)
    return sha256, len(content)


def read_document(store: ObjectStore, document: RawDocument) -> bytes:
    """The bytes stored for a document the run recorded."""
    return store.get(document_key(document.celex, document.resolved_celex, document.sha256))
