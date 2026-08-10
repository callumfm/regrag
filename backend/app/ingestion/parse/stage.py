"""Parse stage: each fetched document's stored HTML as a section tree."""

from collections.abc import Sequence
from pathlib import Path

from app.ingestion.exceptions import ParseError
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.parse.html.parser import parse_eurlex_html
from app.ingestion.parse.models import ParsedDocument, ParseRunResult
from app.ingestion.storage import read_document


def _parse_document(document: RawDocument, *, data_dir: Path) -> ParsedDocument:
    """Parse one stored document's HTML, the one format storage keeps, into a section tree."""
    content = read_document(data_dir, document.celex).decode("utf-8")
    return parse_eurlex_html(content, document.celex, document.topic)


def parse_documents(
    documents: Sequence[RawDocument], *, data_dir: Path
) -> tuple[list[ParsedDocument], ParseRunResult]:
    """Parse every fetched document, recording the ones that would not parse."""
    parsed: list[ParsedDocument] = []
    result = ParseRunResult()
    for document in documents:
        try:
            parsed.append(_parse_document(document, data_dir=data_dir))
        except (ParseError, OSError, UnicodeDecodeError) as exc:
            result.fail(document.celex, exc)
            continue
        result.parsed.append(document.celex)
    return parsed, result
