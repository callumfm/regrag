import pytest

from app.core.config import config
from app.ingestion.chunk.models import Chunk
from app.ingestion.chunk.tree import chunk_document
from app.ingestion.enums import SectionKind
from app.ingestion.parse.html.document import parse_eurlex_html
from app.ingestion.parse.models import ParsedDocument


@pytest.fixture(scope="session")
def fueleu_chunks(fueleu: ParsedDocument) -> tuple[Chunk, ...]:
    """Chunked once for the whole session: Chunk is frozen, so tests share the tuple."""
    return chunk_document(fueleu)


@pytest.fixture(scope="session")
def mrv_chunks(mrv: ParsedDocument) -> tuple[Chunk, ...]:
    """Chunked once and shared like fueleu_chunks."""
    return chunk_document(mrv)


def test_fueleu_article_boundaries_follow_the_regulation(fueleu_chunks: tuple[Chunk, ...]) -> None:
    boundaries = [(c.article, c.paragraph) for c in fueleu_chunks if c.article]
    assert boundaries == [("4", str(n)) for n in range(1, 5)] + [
        ("5", str(n)) for n in range(1, 11)
    ]


def test_fueleu_annex_table_is_chunked_separately(fueleu_chunks: tuple[Chunk, ...]) -> None:
    tables = [c for c in fueleu_chunks if c.kind is SectionKind.TABLE]
    assert len(tables) == 1
    assert tables[0].citation == "Annex II"


def test_fueleu_long_annex_prose_is_split_into_parts(fueleu_chunks: tuple[Chunk, ...]) -> None:
    prose = [c for c in fueleu_chunks if c.annex and c.kind is SectionKind.PARAGRAPH]
    assert len(prose) > 1
    assert {c.parts for c in prose} == {len(prose)}
    assert {c.citation for c in prose} == {"Annex II"}


def test_every_chunk_is_within_the_length_limit(fueleu_chunks: tuple[Chunk, ...]) -> None:
    assert max(len(c.text) for c in fueleu_chunks) <= config.MAX_CHARS


def test_mrv_letter_suffixed_article_is_preserved(mrv_chunks: tuple[Chunk, ...]) -> None:
    assert [c.citation for c in mrv_chunks if c.article == "11a"] == [
        f"Article 11a({n})" for n in range(1, 5)
    ]


def test_mrv_definitions_article_has_no_paragraph_number(mrv_chunks: tuple[Chunk, ...]) -> None:
    definitions = [c for c in mrv_chunks if c.article == "3"]
    assert {c.paragraph for c in definitions} == {None}
    assert {c.citation for c in definitions} == {"Article 3"}


def test_mrv_resolves_external_instruments_to_celex(mrv_chunks: tuple[Chunk, ...]) -> None:
    instruments = {r.instrument for c in mrv_chunks for r in c.references if r.instrument}
    assert instruments == {
        "32003L0087",
        "32008R0765",
        "32009L0016",
        "32012R0601",
        "32018R2066",
        "32023R1805",
    }


def test_annex_heading_path_is_recorded_for_consolidated_annexes(
    mrv_chunks: tuple[Chunk, ...],
) -> None:
    assert any(c.heading_path for c in mrv_chunks if c.annex == "I")


def test_an_act_whose_sole_annex_is_unnumbered_still_addresses_every_chunk() -> None:
    """32024R2031 carries its substance in one annex labelled 'ANNEX', with no numeral to read."""
    sections = parse_eurlex_html(
        "<html><body>"
        '<div class="eli-subdivision" id="art_1">'
        '<p class="oj-ti-art">Article 1</p><p class="oj-normal">Subject matter.</p></div>'
        '<div id="anx_1"><p class="oj-doc-ti">ANNEX</p>'
        '<div class="oj-normal">Template body.</div></div>'
        "</body></html>"
    )
    document = ParsedDocument(celex="32024R2031", topic="fueleu", sections=sections)
    assert [c.citation for c in chunk_document(document)] == ["Article 1", "Annex"]
