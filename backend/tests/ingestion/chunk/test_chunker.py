from app.ingestion.chunk.chunker import chunk_document
from app.ingestion.chunk.references import Reference
from app.ingestion.enums import SectionKind
from app.ingestion.parse.base import ParsedDocument, Section


def paragraph(number: str | None, text: str) -> Section:
    return Section(kind=SectionKind.PARAGRAPH, number=number, text=text)


def document(*sections: Section) -> ParsedDocument:
    return ParsedDocument(ref="32023R1805", topic="fueleu", sections=sections)


def article(number: str, title: str, *children: Section) -> Section:
    return Section(kind=SectionKind.ARTICLE, number=number, title=title, children=children)


def test_emits_one_chunk_per_numbered_paragraph() -> None:
    doc = document(
        article("4", "GHG intensity limit", paragraph("1", "First."), paragraph("2", "Second."))
    )
    chunks = chunk_document(doc)
    assert [c.text for c in chunks] == ["First.", "Second."]


def test_chunk_carries_article_number_and_title() -> None:
    doc = document(article("4", "GHG intensity limit", paragraph("1", "First.")))
    chunk = chunk_document(doc)[0]
    assert (chunk.article, chunk.title, chunk.paragraph) == ("4", "GHG intensity limit", "1")


def test_chunk_carries_document_identity() -> None:
    doc = document(article("4", "Limits", paragraph("1", "First.")))
    chunk = chunk_document(doc)[0]
    assert (chunk.ref, chunk.topic) == ("32023R1805", "fueleu")


def test_citation_combines_article_and_paragraph() -> None:
    doc = document(article("11a", "Reporting", paragraph("3", "Text.")))
    assert chunk_document(doc)[0].citation == "Article 11a(3)"


def test_citation_omits_paragraph_when_article_is_unnumbered() -> None:
    doc = document(article("3", "Definitions", paragraph(None, "For the purposes of...")))
    chunk = chunk_document(doc)[0]
    assert (chunk.citation, chunk.paragraph) == ("Article 3", None)


def test_annex_table_becomes_its_own_chunk() -> None:
    table = Section(kind=SectionKind.TABLE, rows=(("Fuel", "Factor"), ("LNG", "2.75")))
    doc = document(
        Section(kind=SectionKind.ANNEX, number="II", title="Emission factors", children=(table,))
    )
    chunk = chunk_document(doc)[0]
    assert chunk.kind is SectionKind.TABLE
    assert chunk.text == "Fuel | Factor\nLNG | 2.75"
    assert (chunk.annex, chunk.citation) == ("II", "Annex II")


def test_records_nested_heading_path_within_an_annex() -> None:
    inner = Section(
        kind=SectionKind.HEADING, title="1. Formulae", children=(paragraph(None, "Body."),)
    )
    outer = Section(kind=SectionKind.HEADING, title="A. CALCULATION", children=(inner,))
    doc = document(Section(kind=SectionKind.ANNEX, number="I", title="Methods", children=(outer,)))
    chunk = chunk_document(doc)[0]
    assert chunk.heading_path == ("A. CALCULATION", "1. Formulae")
    assert chunk.annex == "I"


def test_attaches_references_found_in_the_chunk_text() -> None:
    doc = document(article("4", "Limits", paragraph("1", "as set out in Annex I")))
    assert chunk_document(doc)[0].references == (Reference(raw="Annex I", annex="I"),)


def test_skips_sections_with_no_text() -> None:
    doc = document(article("4", "Limits", paragraph("1", ""), paragraph("2", "Second.")))
    assert [c.paragraph for c in chunk_document(doc)] == ["2"]


def test_does_not_split_a_section_within_the_limit() -> None:
    doc = document(article("4", "Limits", paragraph("1", "Short.")))
    chunk = chunk_document(doc, max_chars=100)[0]
    assert (chunk.part, chunk.parts) == (1, 1)


def test_splits_a_long_section_on_line_boundaries() -> None:
    doc = document(article("4", "Limits", paragraph("1", "aaaa\nbbbb\ncccc")))
    chunks = chunk_document(doc, max_chars=9)
    assert [c.text for c in chunks] == ["aaaa\nbbbb", "cccc"]


def test_numbers_each_part_of_a_split_section() -> None:
    doc = document(article("4", "Limits", paragraph("1", "aaaa\nbbbb\ncccc")))
    chunks = chunk_document(doc, max_chars=9)
    assert [(c.part, c.parts) for c in chunks] == [(1, 2), (2, 2)]


def test_split_parts_keep_the_same_citation_and_metadata() -> None:
    doc = document(article("4", "Limits", paragraph("2", "aaaa\nbbbb\ncccc")))
    chunks = chunk_document(doc, max_chars=9)
    assert {c.citation for c in chunks} == {"Article 4(2)"}
    assert {c.paragraph for c in chunks} == {"2"}


def test_splits_a_single_overlong_line_on_sentence_boundaries() -> None:
    doc = document(article("4", "Limits", paragraph("1", "One is here. Two is here. Three.")))
    chunks = chunk_document(doc, max_chars=26)
    assert [c.text for c in chunks] == ["One is here. Two is here.", "Three."]
