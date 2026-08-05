from app.ingestion.chunk.models import Chunk, Locator, Reference
from app.ingestion.enums import SectionKind
from app.ingestion.parse.models import ParsedDocument, Section


def test_descend_into_an_article_records_its_number_and_title() -> None:
    section = Section(kind=SectionKind.ARTICLE, number="6", title="Monitoring")
    locator = Locator().descend(section)
    assert (locator.article, locator.title, locator.annex) == ("6", "Monitoring", None)


def test_descend_into_an_annex_clears_the_article() -> None:
    locator = Locator(article="6").descend(Section(kind=SectionKind.ANNEX, number="I"))
    assert (locator.annex, locator.article) == ("I", None)


def test_descend_into_a_heading_extends_the_heading_path() -> None:
    first = Locator().descend(Section(kind=SectionKind.HEADING, title="Part A"))
    second = first.descend(Section(kind=SectionKind.HEADING, title="Part B"))
    assert second.heading_path == ("Part A", "Part B")


def test_descend_into_a_paragraph_changes_nothing() -> None:
    locator = Locator(article="6")
    assert locator.descend(Section(kind=SectionKind.PARAGRAPH, number="2")) == locator


def test_build_combines_document_identity_locator_and_text() -> None:
    document = ParsedDocument(ref="32023R1805", topic="fueleu", sections=())
    section = Section(kind=SectionKind.PARAGRAPH, number="2")
    chunk = Chunk.build(document, section, Locator(article="6"), "The yearly average", 1, 3)
    assert (chunk.ref, chunk.topic) == ("32023R1805", "fueleu")
    assert (chunk.article, chunk.paragraph, chunk.citation) == ("6", "2", "Article 6(2)")
    assert (chunk.part, chunk.parts, chunk.text) == (1, 3, "The yearly average")


def test_build_leaves_paragraph_unset_for_a_non_paragraph_section() -> None:
    document = ParsedDocument(ref="32023R1805", topic="fueleu", sections=())
    section = Section(kind=SectionKind.ARTICLE, number="6")
    assert Chunk.build(document, section, Locator(), "text", 1, 1).paragraph is None


def test_build_carries_the_references_it_is_given() -> None:
    document = ParsedDocument(ref="32023R1805", topic="fueleu", sections=())
    section = Section(kind=SectionKind.PARAGRAPH, number="2")
    reference = Reference(raw="Annex I", annex="I")
    chunk = Chunk.build(document, section, Locator(), "text", 1, 1, (reference,))
    assert chunk.references == (reference,)


def test_build_defaults_to_no_references() -> None:
    document = ParsedDocument(ref="32023R1805", topic="fueleu", sections=())
    section = Section(kind=SectionKind.PARAGRAPH, number="2")
    assert Chunk.build(document, section, Locator(), "text", 1, 1).references == ()
