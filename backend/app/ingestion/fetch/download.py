"""The EUR-Lex HTML endpoint and its quirks: which version it will actually serve, and its bytes."""

import httpx

from app.core.http import is_transient
from app.core.retry import transient_retry
from app.ingestion.discover.models import DiscoveredDocument
from app.ingestion.exceptions import DocumentStillRenderingError, NoFetchableVersionError
from app.ingestion.fetch.models import ResolvedVersion

HTML_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"
MISSING_MARKER = "The requested document does not exist."


def _is_retryable(exc: BaseException) -> bool:
    """A version EUR-Lex is rendering on demand resolves itself, like any transient failure."""
    return isinstance(exc, DocumentStillRenderingError) or is_transient(exc)


download_retry = transient_retry(_is_retryable)
"""Decorator retrying one EUR-Lex request, including the 202 it answers while rendering."""


def version_candidates(document: DiscoveredDocument) -> list[str]:
    """The versions to try in order: the consolidation discovery found, then the original act."""
    if document.candidate_celex:
        return [document.candidate_celex, document.celex]
    return [document.celex]


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


@download_retry
async def download_version(
    client: httpx.AsyncClient, document: DiscoveredDocument, celex: str
) -> bytes | None:
    """The HTML EUR-Lex serves for one version, or None if it denies having that version."""
    response = await client.get(html_url(celex))
    if is_still_rendering(response):
        raise DocumentStillRenderingError(
            f"{document.topic}:{document.celex}: EUR-Lex is still rendering {celex}"
        )
    if is_missing(response):
        return None
    response.raise_for_status()
    return response.content


async def download_fetchable_version(
    client: httpx.AsyncClient, document: DiscoveredDocument
) -> tuple[ResolvedVersion, bytes]:
    """The newest version EUR-Lex will serve, and the HTML it served for it."""
    candidates = version_candidates(document)
    for candidate in candidates:
        content = await download_version(client, document, candidate)
        if content is not None:
            return ResolvedVersion(resolved_celex=candidate, url=html_url(candidate)), content
    raise NoFetchableVersionError(
        f"{document.topic}:{document.celex}: no fetchable HTML, tried {candidates}"
    )
