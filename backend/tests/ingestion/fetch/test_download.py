"""The EUR-Lex HTML endpoint: which version it serves, and how it denies the ones it has not."""

from pathlib import Path

import httpx
import pytest

from app.ingestion.constants import SEEDS
from app.ingestion.discover.stage import discover_topic
from app.ingestion.exceptions import DocumentStillRenderingError, NoFetchableVersionError
from app.ingestion.fetch.download import (
    MISSING_MARKER,
    download_fetchable_version,
    is_missing,
    is_missing_document_page,
    version_candidates,
)
from app.ingestion.fetch.models import ResolvedVersion
from tests.conftest import discovered_document

pytestmark = pytest.mark.anyio

FIXTURES = Path(__file__).parent / "fixtures"
DOC_HTML = (FIXTURES / "doc.html").read_text()
MISSING_HTML_PAGE = (FIXTURES / "missing.html").read_text()


def transport(responses):
    def handler(request):
        celex = request.url.params["uri"].removeprefix("CELEX:")
        return responses[celex]

    return httpx.MockTransport(handler)


def doc_response():
    return httpx.Response(200, text=DOC_HTML)


def missing_response(status=404):
    return httpx.Response(status, text=MISSING_HTML_PAGE)


def rendering_response():
    """What EUR-Lex answers while it generates a document: 202, no body."""
    return httpx.Response(202, content=b"")


def queued(responses):
    """Transport answering each request with the next queued response, recording the celexes."""
    calls: list[str] = []

    def handler(request):
        calls.append(request.url.params["uri"].removeprefix("CELEX:"))
        return responses.pop(0)

    return httpx.MockTransport(handler), calls


def test_missing_document_page_detected():
    assert is_missing_document_page(f"<html><body>{MISSING_MARKER}</body></html>")


def test_real_document_not_flagged_missing():
    assert not is_missing_document_page("<html><body>Article 1</body></html>")


def test_hard_404_is_missing():
    assert is_missing(missing_response(404))


def test_soft_404_is_missing():
    """A success status is not enough: EUR-Lex serves its 'does not exist' page with one."""
    assert is_missing(missing_response(200))


def test_served_document_is_not_missing():
    assert not is_missing(doc_response())


async def test_downloads_candidate_when_html_exists():
    responses = {"02015R0757-20250101": doc_response()}
    async with httpx.AsyncClient(transport=transport(responses)) as client:
        resolution, content = await download_fetchable_version(
            client, discovered_document(candidate="02015R0757-20250101")
        )
    assert resolution == ResolvedVersion(
        resolved_celex="02015R0757-20250101",
        url="https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02015R0757-20250101",
    )
    assert content == DOC_HTML.encode()


async def test_the_served_body_comes_back_so_the_caller_need_not_ask_again():
    """The point of the tuple: one GET per document, not one to check and one to download."""
    calls: list[str] = []

    def handler(request):
        calls.append(str(request.url))
        return doc_response()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        document = discovered_document(celex="32023R2449")
        _, content = await download_fetchable_version(client, document)
    assert content == DOC_HTML.encode()
    assert len(calls) == 1


async def test_falls_back_to_celex_on_hard_404():
    responses = {"02023R2917-20231229": missing_response(404), "32023R2917": doc_response()}
    async with httpx.AsyncClient(transport=transport(responses)) as client:
        resolution, _ = await download_fetchable_version(
            client, discovered_document(celex="32023R2917", candidate="02023R2917-20231229")
        )
    assert resolution.resolved_celex == "32023R2917"


async def test_falls_back_to_celex_on_soft_404():
    responses = {"02023R2917-20231229": missing_response(200), "32023R2917": doc_response()}
    async with httpx.AsyncClient(transport=transport(responses)) as client:
        resolution, _ = await download_fetchable_version(
            client, discovered_document(celex="32023R2917", candidate="02023R2917-20231229")
        )
    assert resolution.resolved_celex == "32023R2917"


async def test_no_candidate_downloads_the_celex_directly():
    responses = {"32023R2449": doc_response()}
    async with httpx.AsyncClient(transport=transport(responses)) as client:
        document = discovered_document(celex="32023R2449")
        resolution, _ = await download_fetchable_version(client, document)
    assert resolution.resolved_celex == "32023R2449"


async def test_raises_when_all_candidates_missing():
    responses = {"02023R2917-20231229": missing_response(404), "32023R2917": missing_response(200)}
    async with httpx.AsyncClient(transport=transport(responses)) as client:
        with pytest.raises(NoFetchableVersionError, match="32023R2917"):
            await download_fetchable_version(
                client, discovered_document(celex="32023R2917", candidate="02023R2917-20231229")
            )


async def test_still_rendering_document_raises():
    responses = {"32023R2917": rendering_response()}
    async with httpx.AsyncClient(transport=transport(responses)) as client:
        with pytest.raises(DocumentStillRenderingError, match="32023R2917"):
            await download_fetchable_version(client, discovered_document(celex="32023R2917"))


async def test_still_rendering_is_retried_until_eurlex_has_rendered_the_document():
    """202 is the normal answer for a version nobody asked for recently, not a failed run."""
    handler, calls = queued([rendering_response(), doc_response()])
    async with httpx.AsyncClient(transport=handler) as client:
        resolution, content = await download_fetchable_version(
            client, discovered_document(celex="32023R2917")
        )
    assert resolution.resolved_celex == "32023R2917"
    assert content == DOC_HTML.encode()
    assert calls == ["32023R2917", "32023R2917"]


async def test_still_rendering_candidate_does_not_fall_back_to_the_original_act():
    """Falling back would swap a consolidated version for the original act and call it changed."""
    responses = {"02023R2917-20231229": rendering_response(), "32023R2917": doc_response()}
    async with httpx.AsyncClient(transport=transport(responses)) as client:
        with pytest.raises(DocumentStillRenderingError):
            await download_fetchable_version(
                client, discovered_document(celex="32023R2917", candidate="02023R2917-20231229")
            )


async def test_a_denied_candidate_is_not_requested_again_when_a_later_one_retries():
    """The retry covers one request: restarting the loop re-asks for a version already denied."""
    handler, calls = queued([missing_response(404), httpx.Response(503), doc_response()])
    async with httpx.AsyncClient(transport=handler) as client:
        resolution, _ = await download_fetchable_version(
            client, discovered_document(celex="32023R2917", candidate="02023R2917-20231229")
        )
    assert resolution.resolved_celex == "32023R2917"
    assert calls == ["02023R2917-20231229", "32023R2917", "32023R2917"]


async def test_unexpected_error_status_raises():
    responses = {"32023R2449": httpx.Response(503, text="maintenance")}
    async with httpx.AsyncClient(transport=transport(responses)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await download_fetchable_version(client, discovered_document(celex="32023R2449"))


def test_version_candidates_try_the_consolidation_before_the_original_act():
    consolidated = discovered_document(candidate="02015R0757-20250101")
    assert version_candidates(consolidated) == ["02015R0757-20250101", "32015R0757"]


def test_version_candidates_without_a_consolidation_is_the_act_alone():
    assert version_candidates(discovered_document("32023R2449")) == ["32023R2449"]


EXPECTED_RESOLVED = {
    "fueleu:32023R1805": "32023R1805",
    "fueleu:32024R2027": "32024R2027",
    "fueleu:32024R2031": "32024R2031",
    "fueleu:32025R0192": "32025R0192",
    "fueleu:32025R1127": "32025R1127",
    "fueleu:32026R0394": "32026R0394",
    "mrv:32015R0757": "02015R0757-20250101",
    "mrv:32016R1928": "32016R1928",
    "mrv:32023R2449": "32023R2449",
    "mrv:32023R2849": "32023R2849",
    "mrv:32023R2917": "32023R2917",
}

MISSING_HTML = {"02023R1805-20230922", "02023R2917-20231229", "02024R2027-20240729"}


def corpus_sparql() -> dict[str, httpx.Response]:
    """The recorded SPARQL response for every topic seed."""
    return {
        topic: httpx.Response(200, text=(FIXTURES / f"sparql-{topic}.json").read_text())
        for topic in SEEDS
    }


def corpus_docs() -> dict[str, httpx.Response]:
    """A 200 for every version EUR-Lex serves, a 404 for the consolidations it does not."""
    return {celex: httpx.Response(200, text=DOC_HTML) for celex in EXPECTED_RESOLVED.values()} | {
        celex: httpx.Response(404, text=MISSING_HTML_PAGE) for celex in MISSING_HTML
    }


@pytest.mark.parametrize("topic", sorted(SEEDS))
async def test_every_discovered_document_resolves_to_the_version_eurlex_serves(
    topic, corpus_client
):
    """The handshake between the two packages: what discovery points at is what download gets."""
    client, _ = corpus_client(corpus_sparql(), corpus_docs())
    discovered = await discover_topic(client, topic, SEEDS[topic])
    resolved = {
        f"{topic}:{d.celex}": (await download_fetchable_version(client, d))[0].resolved_celex
        for d in discovered
    }
    expected = {k: v for k, v in EXPECTED_RESOLVED.items() if k.startswith(f"{topic}:")}
    assert resolved == expected
