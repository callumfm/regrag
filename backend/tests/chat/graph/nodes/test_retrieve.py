"""retrieve: what each query brings back, how the hits interleave, and section widening."""

import pytest

from app.chat.graph.nodes.retrieve import interleave_by_rank, retrieve
from app.chat.models import ChatState
from app.core.config import config
from app.retrieval.models import SearchRequest
from tests.chat.conftest import QUESTION, hits_for, run_graph
from tests.conftest import junk_result, search_result

pytestmark = pytest.mark.anyio


async def test_retrieve_widens_what_search_found_to_whole_sections(
    one_result, answer_model, monkeypatch
):
    """The graph hands search's hits to expansion, so the prompt sees whole sections."""
    widened = (search_result(id=2, citation="Article 4(2)", text="The limit is 91,16 gCO2e/MJ."),)

    async def fake_expand(session, chunks, *, limit):
        assert tuple(chunks) == (search_result(),)
        assert limit == config.CHAT_CONTEXT_CHUNKS
        return widened

    monkeypatch.setattr(config, "EXPAND_SECTIONS", True)
    monkeypatch.setattr("app.chat.graph.nodes.retrieve.expand_sections", fake_expand)

    state = await run_graph()

    assert state.sources == widened
    assert state.hits == (search_result(),)


async def test_retrieve_leaves_search_alone_when_expansion_is_off(
    one_result, answer_model, monkeypatch
):
    """The off switch skips the widening query, not just its result."""

    async def refuse(session, chunks, *, limit):
        raise AssertionError("expansion ran with EXPAND_SECTIONS off")

    monkeypatch.setattr("app.chat.graph.nodes.retrieve.expand_sections", refuse)

    state = await run_graph()

    assert state.sources == (search_result(),)


class TestInterleaveByRank:
    def test_takes_every_querys_first_hit_before_any_querys_second(self):
        first = (search_result(id=1), search_result(id=2))
        second = (search_result(id=3), search_result(id=4))

        merged = interleave_by_rank([first, second])

        assert tuple(chunk.id for chunk in merged) == (1, 3, 2, 4)

    def test_a_chunk_two_queries_found_is_kept_once_at_its_earliest_place(self):
        first = (search_result(id=1), search_result(id=2))
        second = (search_result(id=2), search_result(id=3))

        merged = interleave_by_rank([first, second])

        assert tuple(chunk.id for chunk in merged) == (1, 2, 3)

    def test_a_shorter_list_runs_out_without_ending_the_longer(self):
        first = (search_result(id=1),)
        second = (search_result(id=2), search_result(id=3), search_result(id=4))

        merged = interleave_by_rank([first, second])

        assert tuple(chunk.id for chunk in merged) == (1, 2, 3, 4)

    def test_one_list_comes_back_as_it_was(self):
        only = (search_result(id=1), search_result(id=2))

        assert interleave_by_rank([only]) == only


class TestRetrieveOverQueries:
    async def test_each_query_is_searched_and_the_hits_interleaved(self, monkeypatch):
        requests = hits_for(
            monkeypatch, a=(search_result(id=1), search_result(id=2)), b=(search_result(id=3),)
        )

        update = await retrieve(ChatState(question="A and B?", queries=("a", "b")))

        assert {r.query for r in requests} == {"a", "b"}
        assert all(r.limit == config.CHAT_SOURCES for r in requests)
        assert tuple(chunk.id for chunk in update["hits"]) == (1, 3, 2)
        assert tuple(chunk.id for chunk in update["sources"]) == (1, 3, 2)
        assert update["retrieved_sources"] == 3

    async def test_a_query_below_the_bar_keeps_its_hits_but_adds_no_sources(self, monkeypatch):
        """The out-of-corpus part cannot admit sub-bar hits to the context, yet what search
        found for it stays on the state so the split can be read against it."""
        hits_for(monkeypatch, a=(search_result(id=1),), b=(junk_result(id=9),))

        update = await retrieve(ChatState(question="A and B?", queries=("a", "b")))

        assert tuple(chunk.id for chunk in update["hits"]) == (1, 9)
        assert tuple(chunk.id for chunk in update["sources"]) == (1,)

    async def test_no_query_clearing_the_bar_leaves_the_context_empty(self, monkeypatch):
        junk = junk_result()
        hits_for(monkeypatch, a=(junk,), b=(junk,))

        update = await retrieve(ChatState(question="A and B?", queries=("a", "b")))

        assert update["sources"] == ()
        assert update["retrieved_sources"] == 0
        assert update["hits"] == (junk,)

    async def test_no_queries_searches_the_question_as_asked(self, one_result):
        update = await retrieve(ChatState(question=QUESTION))

        assert one_result == [SearchRequest(query=QUESTION, limit=config.CHAT_SOURCES)]
        assert update["sources"] == (search_result(),)

    async def test_expansion_widens_the_interleaved_survivors(self, monkeypatch):
        hits_for(monkeypatch, a=(search_result(id=1),), b=(search_result(id=2),))
        widened: list[tuple[int, ...]] = []

        async def fake_expand(session, chunks, *, limit):
            widened.append(tuple(chunk.id for chunk in chunks))
            return (*chunks, search_result(id=3))

        monkeypatch.setattr(config, "EXPAND_SECTIONS", True)
        monkeypatch.setattr("app.chat.graph.nodes.retrieve.expand_sections", fake_expand)

        update = await retrieve(ChatState(question="A and B?", queries=("a", "b")))

        assert widened == [(1, 2)]
        assert tuple(chunk.id for chunk in update["sources"]) == (1, 2, 3)
