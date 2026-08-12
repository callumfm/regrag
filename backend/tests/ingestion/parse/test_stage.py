"""The parse stage on one fetched document: what parsed, and what would not.

One bad document not stopping the others is the pipeline's loop now, and is covered there.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from app.ingestion.enums import IngestRunStatus, Stage
from app.ingestion.exceptions import DocumentFailed
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
    parsed = parse_document(fetched(make_document, HTML.encode("utf-8")))
    assert parsed.celex == "32023R1805"


def test_html_that_will_not_parse_fails_the_document_at_the_parse_stage(
    make_document: Callable[..., RawDocument],
) -> None:
    with pytest.raises(DocumentFailed) as excinfo:
        parse_document(fetched(make_document, b"<html></html>"))

    assert (excinfo.value.stage, excinfo.value.celex) == (Stage.PARSE, "32023R1805")
    assert excinfo.value.reason.startswith("ParseError")


def test_html_that_is_not_utf8_fails_the_document_too(
    make_document: Callable[..., RawDocument],
) -> None:
    with pytest.raises(DocumentFailed) as excinfo:
        parse_document(fetched(make_document, b"\xff\xfe<html>"))

    assert excinfo.value.reason.startswith("UnicodeDecodeError")
