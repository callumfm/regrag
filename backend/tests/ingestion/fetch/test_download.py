"""The EUR-Lex HTML endpoint: which version it serves, and how it denies the ones it has not."""

from pathlib import Path

import httpx
import pytest

from app.ingestion.exceptions import DocumentStillRenderingError, NoFetchableVersionError
from app.ingestion.fetch.download import (
    MISSING_MARKER,
    download_fetchable_version,
    expected_version,
    is_missing,
    is_missing_document_page,
)
from app.ingestion.fetch.models import DiscoveredDocument, ResolvedVersion

FIXTURES = Path(__file__).parent / "fixtures"
DOC_HTML = (FIXTURES / "doc.html").read_text()
MISSING_HTML_PAGE = (FIXTURES / "missing.html").read_text()


def spec(celex="32015R0757", candidate=None):
    return DiscoveredDocument(topic="mrv", source="eurlex", celex=celex, candidate_celex=candidate)


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


def test_expected_version_points_at_the_candidate_without_a_request():
    assert expected_version(spec(candidate="02015R0757-20250101")) == ResolvedVersion(
        resolved_celex="02015R0757-20250101",
        url="https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02015R0757-20250101",
    )


def test_expected_version_falls_back_to_the_act_when_discovery_found_no_consolidation():
    assert expected_version(spec(celex="32023R2449")).resolved_celex == "32023R2449"


def test_downloads_candidate_when_html_exists():
    responses = {"02015R0757-20250101": doc_response()}
    with httpx.Client(transport=transport(responses)) as client:
        resolution, content = download_fetchable_version(
            client, spec(candidate="02015R0757-20250101")
        )
    assert resolution == ResolvedVersion(
        resolved_celex="02015R0757-20250101",
        url="https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02015R0757-20250101",
    )
    assert content == DOC_HTML.encode()


def test_the_served_body_comes_back_so_the_caller_need_not_ask_again():
    """The point of the tuple: one GET per document, not one to check and one to download."""
    calls: list[str] = []

    def handler(request):
        calls.append(str(request.url))
        return doc_response()

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        _, content = download_fetchable_version(client, spec(celex="32023R2449"))
    assert content == DOC_HTML.encode()
    assert len(calls) == 1


def test_falls_back_to_celex_on_hard_404():
    responses = {"02023R2917-20231229": missing_response(404), "32023R2917": doc_response()}
    with httpx.Client(transport=transport(responses)) as client:
        resolution, _ = download_fetchable_version(
            client, spec(celex="32023R2917", candidate="02023R2917-20231229")
        )
    assert resolution.resolved_celex == "32023R2917"


def test_falls_back_to_celex_on_soft_404():
    responses = {"02023R2917-20231229": missing_response(200), "32023R2917": doc_response()}
    with httpx.Client(transport=transport(responses)) as client:
        resolution, _ = download_fetchable_version(
            client, spec(celex="32023R2917", candidate="02023R2917-20231229")
        )
    assert resolution.resolved_celex == "32023R2917"


def test_no_candidate_downloads_the_celex_directly():
    responses = {"32023R2449": doc_response()}
    with httpx.Client(transport=transport(responses)) as client:
        resolution, _ = download_fetchable_version(client, spec(celex="32023R2449"))
    assert resolution.resolved_celex == "32023R2449"


def test_raises_when_all_candidates_missing():
    responses = {"02023R2917-20231229": missing_response(404), "32023R2917": missing_response(200)}
    with httpx.Client(transport=transport(responses)) as client:
        with pytest.raises(NoFetchableVersionError, match="32023R2917"):
            download_fetchable_version(
                client, spec(celex="32023R2917", candidate="02023R2917-20231229")
            )


def test_still_rendering_document_raises():
    responses = {"32023R2917": rendering_response()}
    with httpx.Client(transport=transport(responses)) as client:
        with pytest.raises(DocumentStillRenderingError, match="32023R2917"):
            download_fetchable_version(client, spec(celex="32023R2917"))


def test_still_rendering_candidate_does_not_fall_back_to_the_original_act():
    """Falling back would swap a consolidated version for the original act and call it changed."""
    responses = {"02023R2917-20231229": rendering_response(), "32023R2917": doc_response()}
    with httpx.Client(transport=transport(responses)) as client:
        with pytest.raises(DocumentStillRenderingError):
            download_fetchable_version(
                client, spec(celex="32023R2917", candidate="02023R2917-20231229")
            )


def test_unexpected_error_status_raises():
    responses = {"32023R2449": httpx.Response(503, text="maintenance")}
    with httpx.Client(transport=transport(responses)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            download_fetchable_version(client, spec(celex="32023R2449"))
