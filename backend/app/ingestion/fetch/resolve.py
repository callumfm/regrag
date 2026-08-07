"""EUR-Lex resolution: verify graph-fed candidates, falling back to the original act."""

import httpx

from app.core.http import http_retry
from app.ingestion.exceptions import DocumentNotReadyError, ResolutionError
from app.ingestion.fetch.models import DiscoveredDocument, Resolution

HTML_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"
MISSING_MARKER = "The requested document does not exist."


def is_missing_document(html: str) -> bool:
    return MISSING_MARKER in html


def is_still_rendering(response: httpx.Response) -> bool:
    """EUR-Lex answers 202 with an empty body while it generates a document on demand."""
    return response.status_code == httpx.codes.ACCEPTED


@http_retry
def resolve_version(client: httpx.Client, spec: DiscoveredDocument) -> Resolution:
    """Return a verified Resolution for the latest consolidated version, else the original act."""
    candidates = [spec.candidate_celex, spec.celex] if spec.candidate_celex else [spec.celex]
    for candidate in candidates:
        url = HTML_URL.format(celex=candidate)
        response = client.get(url)
        if is_still_rendering(response):
            raise DocumentNotReadyError(
                f"{spec.topic}:{spec.celex}: EUR-Lex is still rendering {candidate}"
            )
        if response.status_code == httpx.codes.NOT_FOUND or (
            response.is_success and is_missing_document(response.text)
        ):
            continue
        response.raise_for_status()
        return Resolution(resolved_celex=candidate, url=url)
    raise ResolutionError(f"{spec.topic}:{spec.celex}: no fetchable HTML, tried {candidates}")
