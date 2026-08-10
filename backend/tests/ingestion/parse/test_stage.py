"""The parse stage over a fetched corpus: what parsed, and what would not."""

from collections.abc import Callable
from pathlib import Path

from app.core.storage import LocalObjectStore
from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.parse.stage import parse_documents
from app.ingestion.schemas import IngestRun

HTML = (Path(__file__).parent / "fixtures" / "32023R1805.html").read_text()


def unstored(make_document: Callable[..., RawDocument], *celexes: str) -> list[RawDocument]:
    """Rows whose bytes were never stored, so reading them fails."""
    run = IngestRun(status=IngestRunStatus.RUNNING)
    return [make_document(run, celex=celex) for celex in celexes]


def stored(
    store_document: Callable[..., RawDocument], content: bytes, *celexes: str
) -> list[RawDocument]:
    """Rows whose HTML is in the store, as the fetch stage would have left them."""
    run = IngestRun(status=IngestRunStatus.RUNNING)
    return [store_document(run, content, celex=celex) for celex in celexes]


def test_parses_every_document_that_has_stored_html(
    local_store: LocalObjectStore, store_document: Callable[..., RawDocument]
) -> None:
    documents = stored(store_document, HTML.encode("utf-8"), "32023R1805")
    parsed, result = parse_documents(documents, store=local_store)
    assert [document.celex for document in parsed] == ["32023R1805"]
    assert result.parsed == ["32023R1805"]
    assert result.ok


def test_missing_bytes_are_recorded_as_a_failure_not_raised(
    local_store: LocalObjectStore, make_document: Callable[..., RawDocument]
) -> None:
    parsed, result = parse_documents(unstored(make_document, "32023R1805"), store=local_store)
    assert parsed == []
    assert result.failed["32023R1805"].startswith("StorageError")
    assert not result.ok


def test_one_bad_document_does_not_stop_the_others(
    local_store: LocalObjectStore,
    make_document: Callable[..., RawDocument],
    store_document: Callable[..., RawDocument],
) -> None:
    documents = [
        *unstored(make_document, "32023R1805"),
        *stored(store_document, HTML.encode("utf-8"), "32015R0757"),
    ]
    parsed, result = parse_documents(documents, store=local_store)
    assert [document.celex for document in parsed] == ["32015R0757"]
    assert list(result.failed) == ["32023R1805"]


def test_html_that_is_not_utf8_is_recorded_not_raised(
    local_store: LocalObjectStore, store_document: Callable[..., RawDocument]
) -> None:
    documents = stored(store_document, b"\xff\xfe<html>", "32023R1805")
    _, result = parse_documents(documents, store=local_store)
    assert result.failed["32023R1805"].startswith("UnicodeDecodeError")
