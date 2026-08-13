"""The EUR-Lex HTML endpoint and its quirks: which version it will actually serve, and its bytes."""

import httpx

from app.core.http import is_transient
from app.core.retry import transient_retry
from app.ingestion.discover.models import DiscoveredDocument
from app.ingestion.exceptions import DocumentStillRenderingError, NoFetchableVersionError
from app.ingestion.fetch.models import ResolvedVersion

HTML_URL_TEMPLATE = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"
MISSING_PAGE_MARKER = "The requested document does not exist."


def eurlex_html_url(celex: str) -> str:
    """The EUR-Lex HTML endpoint for one version."""
    return HTML_URL_TEMPLATE.format(celex=celex)


def is_version_missing(response: httpx.Response) -> bool:
    """EUR-Lex denies a version two ways: a hard 404, or a 200 serving its 'does not exist' page."""
    return response.status_code == httpx.codes.NOT_FOUND or (
        response.is_success and (MISSING_PAGE_MARKER in response.text)
    )


def is_still_rendering(response: httpx.Response) -> bool:
    """EUR-Lex answers 202 with an empty body while it generates a document on demand."""
    return response.status_code == httpx.codes.ACCEPTED


def _is_retryable(exc: BaseException) -> bool:
    """A version EUR-Lex is rendering on demand resolves itself, like any transient failure."""
    return isinstance(exc, DocumentStillRenderingError) or is_transient(exc)


download_retry = transient_retry(_is_retryable)
"""Decorator retrying one EUR-Lex request, including the 202 it answers while rendering."""


@download_retry
async def download_version(
    client: httpx.AsyncClient, document: DiscoveredDocument, version: str
) -> bytes | None:
    """The HTML EUR-Lex serves for one version, or None if it denies having that version."""
    response = await client.get(eurlex_html_url(version))
    if is_still_rendering(response):
        raise DocumentStillRenderingError(
            f"{document.topic}:{document.celex}: EUR-Lex is still rendering {version}"
        )
    if is_version_missing(response):
        return None

    response.raise_for_status()
    return response.content


async def download_fetchable_version(
    client: httpx.AsyncClient, document: DiscoveredDocument
) -> tuple[ResolvedVersion, bytes]:
    """The newest version EUR-Lex will serve, and the HTML it served for it."""
    for version in document.versions:
        content = await download_version(client, document, version)
        if content is not None:
            return ResolvedVersion(resolved_celex=version, url=eurlex_html_url(version)), content

    raise NoFetchableVersionError(
        f"{document.topic}:{document.celex}: no fetchable HTML, tried {document.versions}"
    )
