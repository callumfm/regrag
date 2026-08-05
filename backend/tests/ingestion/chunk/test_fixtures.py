from pathlib import Path

import pytest

from app.ingestion.chunk.chunker import Chunk, chunk_document
from app.ingestion.enums import SectionKind
from app.ingestion.parse.eurlex_html import parse_eurlex_html

FIXTURES = Path(__file__).parents[1] / "parse" / "fixtures"


def chunks_for(ref: str, topic: str) -> tuple[Chunk, ...]:
    document = parse_eurlex_html((FIXTURES / f"{ref}.html").read_text(), ref, topic)
    return chunk_document(document)


@pytest.fixture(scope="module")
def fueleu() -> tuple[Chunk, ...]:
    return chunks_for("32023R1805", "fueleu")


@pytest.fixture(scope="module")
def mrv() -> tuple[Chunk, ...]:
    return chunks_for("32015R0757", "mrv")


def test_fueleu_article_boundaries_follow_the_regulation(fueleu: tuple[Chunk, ...]) -> None:
    boundaries = [(c.article, c.paragraph) for c in fueleu if c.article]
    assert boundaries == [("4", str(n)) for n in range(1, 5)] + [
        ("5", str(n)) for n in range(1, 11)
    ]


def test_fueleu_annex_table_is_chunked_separately(fueleu: tuple[Chunk, ...]) -> None:
    tables = [c for c in fueleu if c.kind is SectionKind.TABLE]
    assert len(tables) == 1
    assert tables[0].citation == "Annex II"


def test_fueleu_long_annex_prose_is_split_into_parts(fueleu: tuple[Chunk, ...]) -> None:
    prose = [c for c in fueleu if c.annex and c.kind is SectionKind.PARAGRAPH]
    assert len(prose) > 1
    assert {c.parts for c in prose} == {len(prose)}
    assert {c.citation for c in prose} == {"Annex II"}


def test_every_chunk_is_within_the_length_limit(fueleu: tuple[Chunk, ...]) -> None:
    assert max(len(c.text) for c in fueleu) <= 2000


def test_mrv_letter_suffixed_article_is_preserved(mrv: tuple[Chunk, ...]) -> None:
    assert [c.citation for c in mrv if c.article == "11a"] == [
        f"Article 11a({n})" for n in range(1, 5)
    ]


def test_mrv_definitions_article_has_no_paragraph_number(mrv: tuple[Chunk, ...]) -> None:
    definitions = [c for c in mrv if c.article == "3"]
    assert {c.paragraph for c in definitions} == {None}
    assert {c.citation for c in definitions} == {"Article 3"}


def test_mrv_resolves_external_instruments_to_celex(mrv: tuple[Chunk, ...]) -> None:
    instruments = {r.instrument for c in mrv for r in c.references if r.instrument}
    assert instruments == {
        "32003L0087",
        "32008R0765",
        "32009L0016",
        "32012R0601",
        "32018R2066",
        "32023R1805",
    }


@pytest.mark.xfail(
    strict=True,
    reason="parser leaves consolidated annex prose in a flat sibling, not under its headings",
)
def test_annex_heading_path_is_recorded_for_consolidated_annexes(mrv: tuple[Chunk, ...]) -> None:
    assert any(c.heading_path for c in mrv if c.annex == "I")
