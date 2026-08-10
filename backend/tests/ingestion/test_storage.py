"""Reading, writing and checking for a fetched document's stored bytes."""

import hashlib
from collections.abc import Callable

import pytest

from app.core.storage import LocalObjectStore, StorageError
from app.ingestion.enums import IngestRunStatus
from app.ingestion.exceptions import EmptyDownloadError
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.schemas import IngestRun
from app.ingestion.storage import document_key, read_document, write_document

HTML = b"<html>act</html>"


def run() -> IngestRun:
    return IngestRun(status=IngestRunStatus.COMPLETED)


def test_the_key_names_the_document_its_version_and_its_content():
    assert document_key("32023R1805", "02023R1805-20250101", "abc") == (
        "32023R1805/02023R1805-20250101/abc.html"
    )


def test_write_stores_the_bytes_and_returns_their_sha_and_size(local_store: LocalObjectStore):
    sha256, size = write_document(local_store, "32023R1805", "32023R1805", HTML)
    assert local_store.get(document_key("32023R1805", "32023R1805", sha256)) == HTML
    assert sha256 == hashlib.sha256(HTML).hexdigest()
    assert size == len(HTML)


def test_write_refuses_empty_content(local_store: LocalObjectStore):
    with pytest.raises(EmptyDownloadError, match="32023R2917"):
        write_document(local_store, "32023R2917", "32023R2917", b"")


def test_changed_content_is_stored_beside_the_bytes_an_earlier_parse_ran_against(
    local_store: LocalObjectStore,
):
    """The point of keying on the hash: a new version never overwrites the one already parsed."""
    old, _ = write_document(local_store, "32015R0757", "32015R0757", b"<html>v1</html>")
    new, _ = write_document(local_store, "32015R0757", "32015R0757", b"<html>v2</html>")

    assert local_store.get(document_key("32015R0757", "32015R0757", old)) == b"<html>v1</html>"
    assert local_store.get(document_key("32015R0757", "32015R0757", new)) == b"<html>v2</html>"


def test_restoring_unchanged_content_lands_on_the_same_key(local_store: LocalObjectStore):
    first, _ = write_document(local_store, "32023R1805", "32023R1805", HTML)
    second, _ = write_document(local_store, "32023R1805", "32023R1805", HTML)
    assert first == second


def test_read_returns_what_write_stored(
    local_store: LocalObjectStore, store_document: Callable[..., RawDocument]
):
    document = store_document(run(), HTML)
    assert read_document(local_store, document) == HTML


def test_read_raises_when_the_bytes_are_not_there(
    local_store: LocalObjectStore, make_document: Callable[..., RawDocument]
):
    with pytest.raises(StorageError, match="get failed"):
        read_document(local_store, make_document(run()))
