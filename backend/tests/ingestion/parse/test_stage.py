"""The parse stage over a fetched corpus: what parsed, and what would not."""

from collections.abc import Callable
from pathlib import Path

import pytest

from app.ingestion.enums import IngestRunStatus
from app.ingestion.exceptions import ParseError
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.parse.stage import parse_document, parse_documents
from app.ingestion.schemas import IngestRun

HTML = (Path(__file__).parent / "fixtures" / "32023R1805.html").read_text()


def documents(make_document: Callable[..., RawDocument], *refs: str) -> list[RawDocument]:
    """Unpersisted rows — parse_documents reads ref and topic, and never touches the session."""
    run = IngestRun(status=IngestRunStatus.RUNNING)
    return [make_document(run, ref=ref) for ref in refs]


def test_parses_every_document_that_has_stored_html(
    tmp_path: Path, make_document: Callable[..., RawDocument]
) -> None:
    (tmp_path / "32023R1805.html").write_text(HTML, encoding="utf-8")
    parsed, delta = parse_documents(documents(make_document, "32023R1805"), tmp_path)
    assert [document.ref for document in parsed] == ["32023R1805"]
    assert delta.parsed == ["32023R1805"]
    assert delta.ok


def test_a_missing_file_is_recorded_as_a_failure_not_raised(
    tmp_path: Path, make_document: Callable[..., RawDocument]
) -> None:
    parsed, delta = parse_documents(documents(make_document, "32023R1805"), tmp_path)
    assert parsed == []
    assert delta.failed["32023R1805"].startswith("FileNotFoundError")
    assert not delta.ok


def test_one_bad_document_does_not_stop_the_others(
    tmp_path: Path, make_document: Callable[..., RawDocument]
) -> None:
    (tmp_path / "32015R0757.html").write_text(HTML, encoding="utf-8")
    parsed, delta = parse_documents(documents(make_document, "32023R1805", "32015R0757"), tmp_path)
    assert [document.ref for document in parsed] == ["32015R0757"]
    assert list(delta.failed) == ["32023R1805"]


def test_html_that_is_not_utf8_is_recorded_not_raised(
    tmp_path: Path, make_document: Callable[..., RawDocument]
) -> None:
    (tmp_path / "32023R1805.html").write_bytes(b"\xff\xfe<html>")
    _, delta = parse_documents(documents(make_document, "32023R1805"), tmp_path)
    assert delta.failed["32023R1805"].startswith("UnicodeDecodeError")


def test_an_unknown_file_type_is_a_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "32023R1805.pdf"
    path.write_text("not html", encoding="utf-8")
    with pytest.raises(ParseError, match="pdf"):
        parse_document(path, "32023R1805", "fueleu")
