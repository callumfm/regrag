"""The parse stage over a fetched corpus: what parsed, and what would not."""

from collections.abc import Callable
from pathlib import Path

from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.models import FetchedDocument
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.parse.stage import parse_documents
from app.ingestion.schemas import IngestRun

HTML = (Path(__file__).parent / "fixtures" / "32023R1805.html").read_text()


def fetched(
    make_document: Callable[..., RawDocument], content: bytes, *celexes: str
) -> list[FetchedDocument]:
    """Rows paired with their bytes, as the fetch stage would have handed them over."""
    run = IngestRun(status=IngestRunStatus.RUNNING)
    return [
        FetchedDocument(document=make_document(run, celex=celex), content=content)
        for celex in celexes
    ]


def test_parses_every_document_it_was_handed(
    make_document: Callable[..., RawDocument],
) -> None:
    documents = fetched(make_document, HTML.encode("utf-8"), "32023R1805")
    parsed, result = parse_documents(documents)
    assert [document.celex for document in parsed] == ["32023R1805"]
    assert result.parsed == ["32023R1805"]
    assert result.ok


def test_html_that_will_not_parse_is_recorded_as_a_failure_not_raised(
    make_document: Callable[..., RawDocument],
) -> None:
    parsed, result = parse_documents(fetched(make_document, b"<html></html>", "32023R1805"))
    assert parsed == []
    assert result.failed["32023R1805"].startswith("ParseError")
    assert not result.ok


def test_one_bad_document_does_not_stop_the_others(
    make_document: Callable[..., RawDocument],
) -> None:
    documents = [
        *fetched(make_document, b"<html></html>", "32023R1805"),
        *fetched(make_document, HTML.encode("utf-8"), "32015R0757"),
    ]
    parsed, result = parse_documents(documents)
    assert [document.celex for document in parsed] == ["32015R0757"]
    assert list(result.failed) == ["32023R1805"]


def test_html_that_is_not_utf8_is_recorded_not_raised(
    make_document: Callable[..., RawDocument],
) -> None:
    _, result = parse_documents(fetched(make_document, b"\xff\xfe<html>", "32023R1805"))
    assert result.failed["32023R1805"].startswith("UnicodeDecodeError")
