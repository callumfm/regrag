import pytest

from app.retrieval.cli import build_parser, main
from app.retrieval.models import RetrievedChunk, SearchResult

RESULT = SearchResult(
    id=1,
    celex="32015R0757",
    topic="mrv",
    citation="Article 4(3)",
    title="General principles",
    text="Companies shall monitor emissions per voyage.",
    score=0.0328,
    vector_rank=1,
    text_rank=2,
)
CHUNK = RetrievedChunk(
    id=1,
    celex="32015R0757",
    topic="mrv",
    citation="Article 11a(1)",
    title=None,
    text="The company shall report.",
)


def test_a_query_and_an_article_lookup_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["a query", "--article", "32015R0757", "4"])


def test_asking_for_neither_a_query_nor_an_article_is_an_error() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_a_search_prints_each_result_with_its_rank_in_both_legs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def _search(*args: object, **kwargs: object) -> tuple[SearchResult, ...]:
        return (RESULT,)

    monkeypatch.setattr("app.retrieval.cli.search", _search)

    assert main(["verification period", "--topic", "mrv"]) == 0
    printed = capsys.readouterr().out
    assert "Article 4(3)" in printed
    assert "32015R0757" in printed
    assert "v1" in printed
    assert "t2" in printed


def test_a_leg_that_missed_a_result_prints_a_dash(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def _search(*args: object, **kwargs: object) -> tuple[SearchResult, ...]:
        return (RESULT.model_copy(update={"text_rank": None}),)

    monkeypatch.setattr("app.retrieval.cli.search", _search)
    main(["verification period"])

    assert "t-" in capsys.readouterr().out


def test_an_article_lookup_prints_the_article(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def _get_article(*args: object, **kwargs: object) -> tuple[RetrievedChunk, ...]:
        return (CHUNK,)

    monkeypatch.setattr("app.retrieval.cli.get_article", _get_article)

    assert main(["--article", "32015R0757", "11a"]) == 0
    assert "Article 11a(1)" in capsys.readouterr().out


def test_finding_nothing_says_so_and_still_succeeds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def _search(*args: object, **kwargs: object) -> tuple[SearchResult, ...]:
        return ()

    monkeypatch.setattr("app.retrieval.cli.search", _search)

    assert main(["nothing matches this"]) == 0
    assert "no results" in capsys.readouterr().out
