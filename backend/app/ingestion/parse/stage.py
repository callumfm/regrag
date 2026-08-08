"""Parse stage: each fetched document's stored HTML as a section tree."""

from collections.abc import Sequence
from pathlib import Path

from app.ingestion.exceptions import ParseError
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.fetch.storage import read_document
from app.ingestion.parse.html.parser import parse_eurlex_html
from app.ingestion.parse.models import ParsedDocument, Parser, ParseRunResult

PARSERS: dict[str, Parser] = {".html": parse_eurlex_html}


def _parse_document(document: RawDocument, *, data_dir: Path) -> ParsedDocument:
    """Parse one stored document, choosing the parser its file type calls for."""
    suffix = document.path(data_dir).suffix
    parser = PARSERS.get(suffix)
    if parser is None:
        raise ParseError(f"{document.celex}: no parser for {suffix or 'a file with no extension'}")
    content = read_document(data_dir, document).decode("utf-8")
    return parser(content, document.celex, document.topic)


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
