"""The EUR-Lex HTML endpoint and its quirks: which version it will actually serve, and its bytes."""

import httpx

from app.core.http import http_retry
from app.ingestion.exceptions import DocumentStillRenderingError, NoFetchableVersionError
from app.ingestion.fetch.discover import version_candidates
from app.ingestion.fetch.models import DiscoveredDocument, ResolvedVersion

HTML_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"
MISSING_MARKER = "The requested document does not exist."


def html_url(celex: str) -> str:
    """The EUR-Lex HTML endpoint for one version."""
    return HTML_URL.format(celex=celex)


def is_missing_document_page(html: str) -> bool:
    """The page EUR-Lex serves, with a success status, in place of a version it does not have."""
    return MISSING_MARKER in html


def is_missing(response: httpx.Response) -> bool:
    """EUR-Lex denies a version two ways: a hard 404, or a 200 serving its 'does not exist' page."""
    return response.status_code == httpx.codes.NOT_FOUND or (
        response.is_success and is_missing_document_page(response.text)
    )


def is_still_rendering(response: httpx.Response) -> bool:
    """EUR-Lex answers 202 with an empty body while it generates a document on demand."""
    return response.status_code == httpx.codes.ACCEPTED


def expected_version(spec: DiscoveredDocument) -> ResolvedVersion:
    """Where the download will land if EUR-Lex still serves the version discovery wants first."""
    celex = version_candidates(spec)[0]
    return ResolvedVersion(resolved_celex=celex, url=html_url(celex))


@http_retry
def download_fetchable_version(
    client: httpx.Client, spec: DiscoveredDocument
) -> tuple[ResolvedVersion, bytes]:
    """The newest version EUR-Lex will serve, and the HTML it served for it."""
    candidates = version_candidates(spec)
    for candidate in candidates:
        url = html_url(candidate)
        response = client.get(url)
        if is_still_rendering(response):
            raise DocumentStillRenderingError(
                f"{spec.topic}:{spec.celex}: EUR-Lex is still rendering {candidate}"
            )
        if is_missing(response):
            continue
        response.raise_for_status()
        return ResolvedVersion(resolved_celex=candidate, url=url), response.content
    raise NoFetchableVersionError(
        f"{spec.topic}:{spec.celex}: no fetchable HTML, tried {candidates}"
    )
