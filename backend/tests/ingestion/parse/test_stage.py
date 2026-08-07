"""The parse stage over a fetched corpus: what parsed, and what would not."""

from collections.abc import Callable
from pathlib import Path

import pytest

from app.core.storage import LocalObjectStore
from app.ingestion.enums import IngestRunStatus
from app.ingestion.exceptions import ParseError
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.parse.stage import _parse_document, parse_documents
from app.ingestion.schemas import IngestRun

HTML = (Path(__file__).parent / "fixtures" / "32023R1805.html").read_text()


def documents(make_document: Callable[..., RawDocument], *celexes: str) -> list[RawDocument]:
    """Unpersisted rows — parse_documents reads each row's key, and never touches the session."""
    run = IngestRun(status=IngestRunStatus.RUNNING)
    return [make_document(run, celex=celex) for celex in celexes]


def test_parses_every_document_that_has_stored_html(
    store: LocalObjectStore, make_document: Callable[..., RawDocument]
) -> None:
    [document] = documents(make_document, "32023R1805")
    store.put(document.key, HTML.encode())
    parsed, result = parse_documents([document], store=store)
    assert [parsed_document.celex for parsed_document in parsed] == ["32023R1805"]
    assert result.parsed == ["32023R1805"]
    assert result.ok


def test_a_missing_object_is_recorded_as_a_failure_not_raised(
    store: LocalObjectStore, make_document: Callable[..., RawDocument]
) -> None:
    parsed, result = parse_documents(documents(make_document, "32023R1805"), store=store)
    assert parsed == []
    assert result.failed["32023R1805"].startswith("StorageError")
    assert not result.ok


def test_one_bad_document_does_not_stop_the_others(
    store: LocalObjectStore, make_document: Callable[..., RawDocument]
) -> None:
    rows = documents(make_document, "32023R1805", "32015R0757")
    store.put(rows[1].key, HTML.encode())
    parsed, result = parse_documents(rows, store=store)
    assert [document.celex for document in parsed] == ["32015R0757"]
    assert list(result.failed) == ["32023R1805"]


def test_html_that_is_not_utf8_is_recorded_not_raised(
    store: LocalObjectStore, make_document: Callable[..., RawDocument]
) -> None:
    [document] = documents(make_document, "32023R1805")
    store.put(document.key, b"\xff\xfe<html>")
    _, result = parse_documents([document], store=store)
    assert result.failed["32023R1805"].startswith("UnicodeDecodeError")


def test_an_unknown_file_type_is_a_parse_error(
    store: LocalObjectStore, make_document: Callable[..., RawDocument], monkeypatch
) -> None:
    [document] = documents(make_document, "32023R1805")
    monkeypatch.setattr(type(document), "key", property(lambda self: "32023R1805/v1/abc.pdf"))
    with pytest.raises(ParseError, match="pdf"):
        _parse_document(document, store=store)
