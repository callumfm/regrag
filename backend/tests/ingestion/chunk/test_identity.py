"""Chunk identity: content hashing and duplicate disambiguation."""

from typing import Any

from app.ingestion.chunk.chunker import Chunk
from app.ingestion.chunk.identity import content_hash, keyed
from app.ingestion.enums import SectionKind


def chunk(**overrides: Any) -> Chunk:
    defaults: dict[str, Any] = {
        "ref": "32023R1805",
        "topic": "fueleu",
        "kind": SectionKind.PARAGRAPH,
        "text": "The greenhouse gas intensity limit.",
        "article": "4",
        "paragraph": "1",
    }
    return Chunk(**{**defaults, **overrides})


def test_hash_is_stable_for_identical_content():
    assert content_hash(chunk()) == content_hash(chunk())


def test_hash_is_sixty_four_hex_chars():
    digest = content_hash(chunk())
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_differing_text_hashes_differently():
    assert content_hash(chunk()) != content_hash(chunk(text="Something else entirely."))


def test_same_text_under_a_different_article_hashes_differently():
    assert content_hash(chunk()) != content_hash(chunk(article="5"))


def test_same_text_in_a_different_document_hashes_differently():
    assert content_hash(chunk()) != content_hash(chunk(ref="32015R0757"))


def test_topic_does_not_affect_the_hash():
    assert content_hash(chunk()) == content_hash(chunk(topic="mrv"))


def test_heading_path_affects_the_hash():
    assert content_hash(chunk()) != content_hash(chunk(heading_path=("Chapter I",)))


def test_occurrence_counts_up_for_duplicates():
    keys = [(digest, n) for _, digest, n in keyed([chunk(), chunk(), chunk()])]
    assert [n for _, n in keys] == [0, 1, 2]
    assert len({digest for digest, _ in keys}) == 1


def test_distinct_chunks_each_start_at_occurrence_zero():
    keys = [(digest, n) for _, digest, n in keyed([chunk(), chunk(article="5")])]
    assert [n for _, n in keys] == [0, 0]
    assert len({digest for digest, _ in keys}) == 2


def test_keyed_yields_the_original_chunks_in_order():
    chunks = [chunk(), chunk(article="5")]
    assert [c for c, _, _ in keyed(chunks)] == chunks
