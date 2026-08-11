"""The parse stage on one fetched document: what parsed, and what would not.

One bad document not stopping the others is the pipeline's loop now, and is covered there.
"""

from collections.abc import Callable
from pathlib import Path

from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.models import FetchedDocument
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.parse.stage import parse_document
from app.ingestion.schemas import IngestRun

HTML = (Path(__file__).parent / "fixtures" / "32023R1805.html").read_text()


def fetched(
    make_document: Callable[..., RawDocument], content: bytes, celex: str = "32023R1805"
) -> FetchedDocument:
    """A row paired with its bytes, as the fetch stage would have handed it over."""
    run = IngestRun(status=IngestRunStatus.RUNNING)
    return FetchedDocument(document=make_document(run, celex=celex), content=content)


def test_parses_the_document_it_was_handed(make_document: Callable[..., RawDocument]) -> None:
    parsed, result = parse_document(fetched(make_document, HTML.encode("utf-8")))
    assert parsed is not None
    assert parsed.celex == "32023R1805"
    assert result.parsed == ["32023R1805"]
    assert result.ok


def test_html_that_will_not_parse_is_recorded_as_a_failure_not_raised(
    make_document: Callable[..., RawDocument],
) -> None:
    parsed, result = parse_document(fetched(make_document, b"<html></html>"))
    assert parsed is None
    assert result.failed["32023R1805"].startswith("ParseError")
    assert not result.ok


def test_html_that_is_not_utf8_is_recorded_not_raised(
    make_document: Callable[..., RawDocument],
) -> None:
    parsed, result = parse_document(fetched(make_document, b"\xff\xfe<html>"))
    assert parsed is None
    assert result.failed["32023R1805"].startswith("UnicodeDecodeError")
