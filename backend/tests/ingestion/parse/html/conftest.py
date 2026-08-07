"""The two EUR-Lex dialect fixtures, parsed once per test module."""

import pytest

from app.ingestion.parse.html.parser import parse_eurlex_html
from app.ingestion.parse.models import ParsedDocument
from tests.ingestion.parse.html.helpers import FUELEU_HTML, MRV_HTML


@pytest.fixture(scope="module")
def fueleu() -> ParsedDocument:
    return parse_eurlex_html(FUELEU_HTML, "32023R1805", "fueleu")


@pytest.fixture(scope="module")
def mrv() -> ParsedDocument:
    return parse_eurlex_html(MRV_HTML, "32015R0757", "mrv")
