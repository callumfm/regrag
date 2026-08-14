from app.ingestion.chunk.models import Locator, locator_for
from app.ingestion.enums import SectionKind
from app.ingestion.parse.models import Section
from tests.conftest import chunk

ARTICLE_6 = Section(kind=SectionKind.ARTICLE, number="6", title="Monitoring")
ANNEX_I = Section(kind=SectionKind.ANNEX, number="I", title="Methods")
PARAGRAPH_2 = Section(kind=SectionKind.PARAGRAPH, number="2")


def heading(title: str) -> Section:
    return Section(kind=SectionKind.HEADING, title=title)


def test_an_article_ancestor_gives_its_number_and_title() -> None:
    locator = locator_for((ARTICLE_6, PARAGRAPH_2))
    assert (locator.article, locator.title, locator.annex) == ("6", "Monitoring", None)


def test_the_nearest_division_wins_so_an_annex_inside_an_article_is_not_an_article() -> None:
    locator = locator_for((ARTICLE_6, ANNEX_I, PARAGRAPH_2))
    assert (locator.annex, locator.article, locator.title) == ("I", None, "Methods")


def test_headings_stack_in_the_order_they_were_entered() -> None:
    locator = locator_for((ANNEX_I, heading("Part A"), heading("Part B"), PARAGRAPH_2))
    assert locator.heading_path == ("Part A", "Part B")


def test_a_paragraph_contributes_nothing_to_the_address() -> None:
    assert locator_for((ARTICLE_6, PARAGRAPH_2)) == locator_for((ARTICLE_6,))


def test_a_path_with_no_article_or_annex_has_an_empty_address() -> None:
    assert locator_for((heading("Part A"),)) == Locator(heading_path=("Part A",))


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


def test_metadata_hash_is_stable_for_identical_content():
    assert chunk().metadata_hash == chunk().metadata_hash


def test_topic_affects_the_metadata_hash_not_the_content_hash():
    assert chunk().metadata_hash != chunk(topic="mrv").metadata_hash


def test_position_affects_the_metadata_hash_not_the_content_hash():
    assert chunk().metadata_hash != chunk(position=7).metadata_hash


def test_text_affects_the_content_hash_not_the_metadata_hash():
    """Identity and metadata fields partition the chunk: each change lands in exactly one hash."""
    assert chunk().metadata_hash == chunk(text="Something else entirely.").metadata_hash
