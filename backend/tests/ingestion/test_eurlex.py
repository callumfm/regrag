"""EUR-Lex resolver: consolidated-version selection, soft-404 detection."""

from app.ingestion.eurlex import (
    MISSING_MARKER,
    is_missing_document,
    latest_consolidated_ref,
)

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
