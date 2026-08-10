"""The parse stage over a fetched corpus: what parsed, and what would not."""

from collections.abc import Callable
from pathlib import Path

from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.parse.stage import parse_documents
from app.ingestion.schemas import IngestRun
from app.ingestion.storage import write_document

HTML = (Path(__file__).parent / "fixtures" / "32023R1805.html").read_text()


def documents(make_document: Callable[..., RawDocument], *celexes: str) -> list[RawDocument]:
    """Unpersisted rows — parse_documents reads celex and topic, and never touches the session."""
    run = IngestRun(status=IngestRunStatus.RUNNING)
    return [make_document(run, celex=celex) for celex in celexes]


def test_parses_every_document_that_has_stored_html(
    tmp_path: Path, make_document: Callable[..., RawDocument]
) -> None:
    write_document(tmp_path, "32023R1805", HTML.encode("utf-8"))
    parsed, result = parse_documents(documents(make_document, "32023R1805"), data_dir=tmp_path)
    assert [document.celex for document in parsed] == ["32023R1805"]
    assert result.parsed == ["32023R1805"]
    assert result.ok


def test_a_missing_file_is_recorded_as_a_failure_not_raised(
    tmp_path: Path, make_document: Callable[..., RawDocument]
) -> None:
    parsed, result = parse_documents(documents(make_document, "32023R1805"), data_dir=tmp_path)
    assert parsed == []
    assert result.failed["32023R1805"].startswith("FileNotFoundError")
    assert not result.ok


def test_one_bad_document_does_not_stop_the_others(
    tmp_path: Path, make_document: Callable[..., RawDocument]
) -> None:
    write_document(tmp_path, "32015R0757", HTML.encode("utf-8"))
    parsed, result = parse_documents(
        documents(make_document, "32023R1805", "32015R0757"), data_dir=tmp_path
    )
    assert [document.celex for document in parsed] == ["32015R0757"]
    assert list(result.failed) == ["32023R1805"]


def test_html_that_is_not_utf8_is_recorded_not_raised(
    tmp_path: Path, make_document: Callable[..., RawDocument]
) -> None:
    write_document(tmp_path, "32023R1805", b"\xff\xfe<html>")
    _, result = parse_documents(documents(make_document, "32023R1805"), data_dir=tmp_path)
    assert result.failed["32023R1805"].startswith("UnicodeDecodeError")
