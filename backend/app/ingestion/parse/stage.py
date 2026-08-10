"""Parse stage: each fetched document's stored HTML as a section tree."""

from collections.abc import Sequence

from app.core.storage import ObjectStore, StorageError
from app.ingestion.exceptions import ParseError
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.parse.html.parser import parse_eurlex_html
from app.ingestion.parse.models import ParsedDocument, ParseRunResult
from app.ingestion.storage import read_document


def _parse_document(document: RawDocument, *, store: ObjectStore) -> ParsedDocument:
    """Parse one stored document's HTML, the one format storage keeps, into a section tree."""
    content = read_document(store, document).decode("utf-8")
    return parse_eurlex_html(content, document.celex, document.topic)


def parse_documents(
    documents: Sequence[RawDocument], *, store: ObjectStore
) -> tuple[list[ParsedDocument], ParseRunResult]:
    """Parse every fetched document, recording the ones that would not parse."""
    parsed: list[ParsedDocument] = []
    result = ParseRunResult()
    for document in documents:
        try:
            parsed.append(_parse_document(document, store=store))
        except (ParseError, StorageError, UnicodeDecodeError) as exc:
            result.fail(document.celex, exc)
            continue
        result.parsed.append(document.celex)
    return parsed, result
