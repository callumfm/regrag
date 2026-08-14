"""Where a fetched document's bytes live: the only module that reads or writes them."""

import hashlib

from app.core.storage import ObjectStore, StorageError
from app.ingestion.exceptions import EmptyDownloadError
from app.ingestion.fetch.schemas import RawDocument


class StoredBytesMismatchError(StorageError):
    """The object at a document's key is not the one the row recorded."""


def document_key(celex: str, resolved_celex: str, sha256: str) -> str:
    """Where a document's bytes live, keyed by content so a new version never overwrites a
    version an earlier parse ran against."""
    return f"{celex}/{resolved_celex}/{sha256}.html"


def write_document(
    store: ObjectStore, celex: str, resolved_celex: str, html: bytes
) -> tuple[str, int]:
    """Store the document's bytes and return their (sha256, size_bytes), refusing empty
    content so a failed download cannot be recorded as a version."""
    if not html:
        raise EmptyDownloadError(f"{celex}: download returned an empty body")
    sha256 = hashlib.sha256(html).hexdigest()
    store.put(document_key(celex, resolved_celex, sha256), html)
    return sha256, len(html)


def read_document(store: ObjectStore, document: RawDocument) -> bytes:
    """The bytes stored for a document, refusing any that are not the ones the row recorded.

    The row and the object are backed up separately, so a restore can leave them disagreeing;
    the reuse path treats that as bytes it does not have and downloads the version again.
    """
    key = document_key(document.celex, document.resolved_celex, document.sha256)
    html = store.get(key)
    if hashlib.sha256(html).hexdigest() != document.sha256:
        raise StoredBytesMismatchError("verify", key, "stored bytes do not match the recorded hash")
    return html
