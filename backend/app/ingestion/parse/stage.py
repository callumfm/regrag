"""Parse stage: each fetched document's HTML as a section tree."""

from collections.abc import Sequence

from app.ingestion.exceptions import ParseError
from app.ingestion.fetch.models import FetchedDocument
from app.ingestion.parse.html.parser import parse_eurlex_html
from app.ingestion.parse.models import ParsedDocument, ParseRunResult


def _parse_document(fetched: FetchedDocument) -> ParsedDocument:
    """Parse one document's HTML, the one format storage keeps, into a section tree."""
    document = fetched.document
    content = fetched.content.decode("utf-8")
    return parse_eurlex_html(content, document.celex, document.topic)


def parse_documents(
    fetched: Sequence[FetchedDocument],
) -> tuple[list[ParsedDocument], ParseRunResult]:
    """Parse every fetched document, recording the ones that would not parse."""
    parsed: list[ParsedDocument] = []
    result = ParseRunResult()
    for item in fetched:
        try:
            parsed.append(_parse_document(item))
        except (ParseError, UnicodeDecodeError) as exc:
            result.fail(item.document.celex, exc)
            continue
        result.parsed.append(item.document.celex)
    return parsed, result
