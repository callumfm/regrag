"""Retrieval CLI: one line per hit, every retrieval signal side by side."""

from app.retrieval.cli import format_result
from tests.conftest import search_result


def test_a_line_shows_every_signal_then_the_citation() -> None:
    result = search_result(rrf_score=0.0328, cosine_similarity=0.7512, reranker_relevance=0.6)

    line = format_result(result)

    assert line.startswith("rrf 0.0328  cos 0.7512  rel 0.6000  Article 4(1)")
    assert "32023R1805" in line


def test_a_missing_signal_prints_as_a_dash() -> None:
    result = search_result(cosine_similarity=None, reranker_relevance=None)

    line = format_result(result)

    assert line.startswith("rrf 0.9000  cos -       rel -       Article 4(1)")
