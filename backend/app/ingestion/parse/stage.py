"""Parse stage: one fetched document's HTML as a section tree."""

from app.ingestion.enums import Stage
from app.ingestion.exceptions import DocumentFailed, ParseError
from app.ingestion.fetch.models import FetchedDocument
from app.ingestion.parse.html.parser import parse_eurlex_html
from app.ingestion.parse.models import ParsedDocument


def parse_document(fetched: FetchedDocument) -> ParsedDocument:
    """Parse one document's HTML, the one format storage keeps, or say why it would not."""
    document = fetched.document
    try:
        return parse_eurlex_html(fetched.content.decode("utf-8"), document.celex, document.topic)
    except (ParseError, UnicodeDecodeError) as exc:
        raise DocumentFailed(Stage.PARSE, document.celex, exc) from exc
