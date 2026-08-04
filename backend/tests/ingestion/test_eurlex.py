"""EUR-Lex resolver: consolidated-version selection, soft-404 detection, resolve()."""

from pathlib import Path

import httpx
import pytest

from app.ingestion.eurlex import (
    MISSING_MARKER,
    ResolutionError,
    is_missing_document,
    latest_consolidated_ref,
    resolve,
)
from app.ingestion.registry import CORPUS, DocumentSpec

ALL_PAGE_SNIPPET = """
<html><body>
<a href="./?uri=CELEX:02015R0757-20240101">Consolidated text 01/01/2024</a>
<a href="./?uri=CELEX:02015R0757-20250101">Consolidated text 01/01/2025</a>
<a href="./?uri=CELEX:02015R0757-20160701">Consolidated text 01/07/2016</a>
</body></html>
"""


def test_picks_max_consolidation_date():
    assert latest_consolidated_ref(ALL_PAGE_SNIPPET, "32015R0757") == "02015R0757-20250101"


def test_returns_none_when_no_consolidated_versions():
    html = '<html><body><a href="./?uri=CELEX:32023R1805">Original act</a></body></html>'
    assert latest_consolidated_ref(html, "32023R1805") is None


def test_ignores_other_acts_consolidated_refs():
    assert latest_consolidated_ref(ALL_PAGE_SNIPPET, "32023R1805") is None


def test_missing_document_page_detected():
    assert is_missing_document(f"<html><body>{MISSING_MARKER}</body></html>")


def test_real_document_not_flagged_missing():
    assert not is_missing_document("<html><body>Article 1</body></html>")


FIXTURES = Path(__file__).parent / "fixtures"

EXPECTED_RESOLVED = {
    "fueleu-maritime": "32023R1805",
    "ets-directive": "02003L0087-20240301",
    "mrv-regulation": "02015R0757-20250101",
    "fueleu-verification": "32024R2027",
    "fueleu-monitoring-plan": "32024R2031",
    "fueleu-verifier-accreditation": "32025R0192",
    "fueleu-transhipment-ports": "32025R1127",
    "fueleu-database": "32026R0394",
    "ets-company-administration": "32023R2599",
    "ets-administering-authorities": "02024D0411-20260101",
    "ets-administering-authorities-correction": "32026D1453",
    "ets-transhipment-ports": "32023R2297",
    "ets-derogation-lists": "02023D2895-20250101",
    "mrv-templates": "32023R2449",
    "mrv-verification": "32023R2917",
    "mrv-company-emissions": "32023R2849",
    "mrv-cargo-determination": "32016R1928",
}

MISSING_HTML = {
    "02016R1928-20161105",
    "02023R1805-20230922",
    "02024R2027-20240729",
    "02023R2917-20231229",
}


def fixture_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        celex = request.url.params["uri"].removeprefix("CELEX:")
        if "/ALL/" in request.url.path:
            return httpx.Response(200, text=(FIXTURES / f"{celex}-all.html").read_text())
        if celex in MISSING_HTML:
            return httpx.Response(404, text=(FIXTURES / "missing.html").read_text())
        return httpx.Response(200, text=(FIXTURES / "doc.html").read_text())

    return httpx.MockTransport(handler)


@pytest.mark.parametrize("spec", CORPUS, ids=[spec.name for spec in CORPUS])
def test_every_corpus_entry_resolves(spec: DocumentSpec):
    with httpx.Client(transport=fixture_transport()) as client:
        resolution = resolve(client, spec)
    assert resolution.resolved_ref == EXPECTED_RESOLVED[spec.name]
    assert resolution.url.endswith(f"CELEX:{resolution.resolved_ref}")


def test_falls_back_to_original_when_consolidated_html_missing():
    spec = next(s for s in CORPUS if s.name == "fueleu-verification")
    with httpx.Client(transport=fixture_transport()) as client:
        resolution = resolve(client, spec)
    assert resolution.resolved_ref == spec.ref


def test_raises_when_original_also_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        if "/ALL/" in request.url.path:
            return httpx.Response(200, text="<html><body>no versions</body></html>")
        return httpx.Response(200, text=(FIXTURES / "missing.html").read_text())

    spec = DocumentSpec(name="ghost", source="eurlex", ref="32099R9999")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ResolutionError, match="ghost"):
            resolve(client, spec)
