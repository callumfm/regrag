from app.ingestion.chunk.models import Locator
from app.ingestion.enums import SectionKind
from app.ingestion.parse.models import Section
from tests.conftest import chunk


def test_folding_in_an_article_records_its_number_and_title() -> None:
    section = Section(kind=SectionKind.ARTICLE, number="6", title="Monitoring")
    locator = Locator().with_section(section)
    assert (locator.article, locator.title, locator.annex) == ("6", "Monitoring", None)


def test_folding_in_an_annex_clears_the_article() -> None:
    locator = Locator(article="6").with_section(Section(kind=SectionKind.ANNEX, number="I"))
    assert (locator.annex, locator.article) == ("I", None)


def test_folding_in_a_heading_extends_the_heading_path() -> None:
    first = Locator().with_section(Section(kind=SectionKind.HEADING, title="Part A"))
    second = first.with_section(Section(kind=SectionKind.HEADING, title="Part B"))
    assert second.heading_path == ("Part A", "Part B")


def test_folding_in_a_paragraph_changes_nothing() -> None:
    locator = Locator(article="6")
    assert locator.with_section(Section(kind=SectionKind.PARAGRAPH, number="2")) == locator


def test_hash_is_stable_for_identical_content():
    assert chunk().content_hash == chunk().content_hash


def test_hash_is_sixty_four_hex_chars():
    digest = chunk().content_hash
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_differing_text_hashes_differently():
    assert chunk().content_hash != chunk(text="Something else entirely.").content_hash


def test_same_text_under_a_different_article_hashes_differently():
    assert chunk().content_hash != chunk(article="5").content_hash


def test_same_text_in_a_different_document_hashes_differently():
    assert chunk().content_hash != chunk(celex="32015R0757").content_hash


def test_topic_does_not_affect_the_hash():
    assert chunk().content_hash == chunk(topic="mrv").content_hash


def test_heading_path_affects_the_hash():
    assert chunk().content_hash != chunk(heading_path=("Chapter I",)).content_hash


def test_position_does_not_affect_the_hash():
    """An inserted paragraph shifts every position after it; none of those chunks may churn."""
    shifted = chunk(position=7)
    assert shifted.position == 7
    assert shifted.content_hash == chunk().content_hash
