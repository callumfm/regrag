"""Parse stage: each fetched document's stored HTML as a section tree."""

from collections.abc import Sequence
from pathlib import PurePosixPath

from app.core.storage import ObjectStore, StorageError
from app.ingestion.exceptions import ParseError
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.parse.eurlex_html import parse_eurlex_html
from app.ingestion.parse.models import ParsedDocument, Parser, ParseRunResult

PARSERS: dict[str, Parser] = {".html": parse_eurlex_html}


def _parse_document(document: RawDocument, *, store: ObjectStore) -> ParsedDocument:
    """Parse one stored document, choosing the parser its file type calls for."""
    suffix = PurePosixPath(document.key).suffix
    parser = PARSERS.get(suffix)
    if parser is None:
        raise ParseError(f"{document.celex}: no parser for {suffix or 'an object with no suffix'}")
    content = store.get(document.key).decode("utf-8")
    return parser(content, document.celex, document.topic)


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
