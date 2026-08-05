"""EUR-Lex resolution: verify graph-fed candidates, falling back to the original act."""

import httpx

from app.core.http import transient_retry
from app.ingestion.exceptions import ResolutionError
from app.ingestion.fetch.models import DiscoveredDocument, Resolution

HTML_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{ref}"
MISSING_MARKER = "The requested document does not exist."


def is_missing_document(html: str) -> bool:
    return MISSING_MARKER in html


@transient_retry
def resolve_version(client: httpx.Client, spec: DiscoveredDocument) -> Resolution:
    """Return a verified Resolution for the latest consolidated version, else the original act."""
    candidates = [spec.candidate_ref, spec.ref] if spec.candidate_ref else [spec.ref]
    for candidate in candidates:
        url = HTML_URL.format(ref=candidate)
        response = client.get(url)
        if response.status_code == httpx.codes.NOT_FOUND or (
            response.is_success and is_missing_document(response.text)
        ):
            continue
        response.raise_for_status()
        return Resolution(resolved_ref=candidate, url=url)
    raise ResolutionError(f"{spec.topic}:{spec.ref}: no fetchable HTML, tried {candidates}")
