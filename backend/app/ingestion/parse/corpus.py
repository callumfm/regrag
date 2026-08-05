"""Parse stage: each fetched document's stored HTML as a section tree."""

from collections.abc import Sequence
from pathlib import Path

from app.ingestion.exceptions import ParseError
from app.ingestion.parse.eurlex_html import parse_eurlex_html
from app.ingestion.parse.models import ParsedDocument, ParseDelta
from app.ingestion.schemas import IngestedDocument
from app.ingestion.storage import raw_html_path


def parse_corpus(
    documents: Sequence[IngestedDocument], data_dir: Path
) -> tuple[list[ParsedDocument], ParseDelta]:
    """Parse every fetched document, recording the ones that would not parse."""
    parsed: list[ParsedDocument] = []
    delta = ParseDelta()
    for document in documents:
        try:
            html = raw_html_path(data_dir, document.ref).read_text(encoding="utf-8")
            parsed.append(parse_eurlex_html(html, document.ref, document.topic))
        except (ParseError, OSError, UnicodeDecodeError) as exc:
            delta.failed[document.ref] = f"{type(exc).__name__}: {exc}"
            continue
        delta.parsed.append(document.ref)
    return parsed, delta
