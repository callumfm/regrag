"""Citation markers: how an answer's [n] markers are read, stripped, and bound to blocks."""

from app.core.citations import find_cited_markers, find_cited_sources, strip_markers


def test_strip_markers_removes_every_citation_marker_and_nothing_else() -> None:
    assert strip_markers("Ships must report.[1] Yearly.[2][3] Done.") == (
        "Ships must report. Yearly. Done."
    )
    assert strip_markers("No markers here.") == "No markers here."


def test_markers_are_read_in_first_cited_order_without_repeats() -> None:
    assert find_cited_markers("A [2] B [1][2] C [10]") == (2, 1, 10)


def test_no_markers_when_the_answer_cites_nothing() -> None:
    assert find_cited_markers("I cannot answer that from the documents I have.") == ()


def test_cited_sources_pair_each_marker_with_its_block_in_cited_order() -> None:
    sources = ("first", "second", "third")

    assert find_cited_sources("So [3], then [1] and [7].", sources) == ((3, "third"), (1, "first"))
