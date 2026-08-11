"""Parse stage: one fetched document's HTML as a section tree."""

from app.ingestion.exceptions import ParseError
from app.ingestion.fetch.models import FetchedDocument
from app.ingestion.parse.html.parser import parse_eurlex_html
from app.ingestion.parse.models import ParsedDocument, ParseRunResult


def parse_document(fetched: FetchedDocument) -> tuple[ParsedDocument | None, ParseRunResult]:
    """Parse one document's HTML, the one format storage keeps, or record why it would not."""
    result = ParseRunResult()
    document = fetched.document
    try:
        parsed = parse_eurlex_html(fetched.content.decode("utf-8"), document.celex, document.topic)
    except (ParseError, UnicodeDecodeError) as exc:
        result.fail(document.celex, exc)
        return None, result
    result.parsed.append(document.celex)
    return parsed, result
