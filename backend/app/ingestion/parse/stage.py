"""Parse stage: each fetched document's stored HTML as a section tree."""

from collections.abc import Sequence
from pathlib import Path

from app.ingestion.exceptions import ParseError
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.models import ParseDelta
from app.ingestion.parse.eurlex_html import parse_eurlex_html
from app.ingestion.parse.models import ParsedDocument, Parser

PARSERS: dict[str, Parser] = {".html": parse_eurlex_html}


def parse_document(path: Path, ref: str, topic: str) -> ParsedDocument:
    """Parse one stored document, choosing the parser its file type calls for."""
    parser = PARSERS.get(path.suffix)
    if parser is None:
        raise ParseError(f"{ref}: no parser for {path.suffix or 'a file with no extension'}")
    return parser(path.read_text(encoding="utf-8"), ref, topic)


def parse_documents(
    documents: Sequence[RawDocument], data_dir: Path
) -> tuple[list[ParsedDocument], ParseDelta]:
    """Parse every fetched document, recording the ones that would not parse."""
    parsed: list[ParsedDocument] = []
    delta = ParseDelta()
    for document in documents:
        try:
            parsed.append(parse_document(document.path(data_dir), document.ref, document.topic))
        except (ParseError, OSError, UnicodeDecodeError) as exc:
            delta.failed[document.ref] = f"{type(exc).__name__}: {exc}"
            continue
        delta.parsed.append(document.ref)
    return parsed, delta
