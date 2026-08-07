"""The two EUR-Lex dialect fixtures, parsed once per test module."""

from pathlib import Path

import pytest

from app.ingestion.parse.html.parser import parse_eurlex_html
from app.ingestion.parse.models import ParsedDocument

FIXTURES = Path(__file__).parent.parent / "fixtures"
FUELEU_HTML = (FIXTURES / "32023R1805.html").read_text()
MRV_HTML = (FIXTURES / "32015R0757.html").read_text()


@pytest.fixture(scope="module")
def fueleu() -> ParsedDocument:
    return parse_eurlex_html(FUELEU_HTML, "32023R1805", "fueleu")


@pytest.fixture(scope="module")
def mrv() -> ParsedDocument:
    return parse_eurlex_html(MRV_HTML, "32015R0757", "mrv")
