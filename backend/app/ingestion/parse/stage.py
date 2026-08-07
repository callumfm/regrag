"""Parse stage: each fetched document's stored HTML as a section tree."""

from collections.abc import Sequence
from pathlib import Path

from app.ingestion.exceptions import ParseError
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.parse.eurlex_html import parse_eurlex_html
from app.ingestion.parse.models import ParsedDocument, Parser, ParseRunResult

PARSERS: dict[str, Parser] = {".html": parse_eurlex_html}


def _parse_document(path: Path, *, celex: str, topic: str) -> ParsedDocument:
    """Parse one stored document, choosing the parser its file type calls for."""
    parser = PARSERS.get(path.suffix)
    if parser is None:
        raise ParseError(f"{celex}: no parser for {path.suffix or 'a file with no extension'}")
    return parser(path.read_text(encoding="utf-8"), celex, topic)


def parse_documents(
    documents: Sequence[RawDocument], *, data_dir: Path
) -> tuple[list[ParsedDocument], ParseRunResult]:
    """Parse every fetched document, recording the ones that would not parse."""
    parsed: list[ParsedDocument] = []
    result = ParseRunResult()
    for document in documents:
        try:
            parsed.append(
                _parse_document(document.path(data_dir), celex=document.celex, topic=document.topic)
            )
        except (ParseError, OSError, UnicodeDecodeError) as exc:
            result.fail(document.celex, exc)
            continue
        result.parsed.append(document.celex)
    return parsed, result
