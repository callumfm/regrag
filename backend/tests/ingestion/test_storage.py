"""Reading, writing and checking for a fetched document's stored bytes."""

import hashlib
from pathlib import Path

import pytest

from app.ingestion.exceptions import EmptyDownloadError
from app.ingestion.storage import document_exists, read_document, write_document


def test_write_stores_the_bytes_and_returns_their_sha_and_size(tmp_path: Path) -> None:
    content = b"<html>act</html>"
    sha256, size = write_document(tmp_path / "raw", "32023R1805", content)
    assert (tmp_path / "raw" / "32023R1805.html").read_bytes() == content
    assert sha256 == hashlib.sha256(content).hexdigest()
    assert size == len(content)


def test_write_refuses_empty_content(tmp_path: Path) -> None:
    with pytest.raises(EmptyDownloadError, match="32023R2917"):
        write_document(tmp_path / "raw", "32023R2917", b"")


def test_write_leaves_the_previous_bytes_intact_when_content_is_empty(tmp_path: Path) -> None:
    """An empty body must not destroy the last good copy of the document."""
    write_document(tmp_path, "32023R2917", b"<html>act</html>")
    with pytest.raises(EmptyDownloadError):
        write_document(tmp_path, "32023R2917", b"")
    assert (tmp_path / "32023R2917.html").read_bytes() == b"<html>act</html>"


def test_read_returns_what_write_stored(tmp_path: Path) -> None:
    write_document(tmp_path, "32023R1805", b"<html>act</html>")
    assert read_document(tmp_path, "32023R1805") == b"<html>act</html>"


def test_read_raises_when_the_bytes_are_not_there(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_document(tmp_path, "32023R1805")


def test_exists_is_true_only_once_the_bytes_are_written(tmp_path: Path) -> None:
    assert not document_exists(tmp_path, "32023R1805")
    write_document(tmp_path, "32023R1805", b"<html>act</html>")
    assert document_exists(tmp_path, "32023R1805")
