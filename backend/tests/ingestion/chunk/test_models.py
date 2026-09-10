from app.ingestion.chunk.models import Chunk
from tests.conftest import chunk


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


def test_points_are_what_the_text_opens_lines_with():
    """A definitions article lists its terms one to a line, '(e)' or '(15)' first."""
    text = "For the purposes of this Regulation:\n(a) ‘ship’ means a vessel;\n(15) ‘berth’ means"
    assert chunk(text=text).points == ("a", "15")


def test_a_point_named_mid_line_is_not_a_point_of_the_chunk():
    assert chunk(text="as defined in Article 3, point (e), of Regulation X").points == ()


def test_points_land_in_the_metadata_hash_not_the_content_hash():
    """Points derive from the text, so a chunk that grows one changes only its metadata."""
    assert chunk(text="(a) ‘ship’ means").metadata_hash != chunk(text="‘ship’ means").metadata_hash
    assert "points" not in chunk().model_dump(exclude=Chunk.METADATA)
