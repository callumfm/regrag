"""Chat graph: the compiled shape, and the routing that ends a run in an answer or the
fixed refusal."""

import pytest

from app.chat.enums import RefusalReason
from app.chat.graph import GRAPH_EDGES, chat_graph
from app.chat.models import ChatState, Refusal
from app.chat.prompts import REFUSAL_ANSWER
from app.core.config import config
from app.retrieval.models import SearchRequest
from tests.chat.conftest import QUESTION, fake_chat_model
from tests.conftest import install_chat_model, search_result

pytestmark = pytest.mark.anyio


async def test_graph_retrieves_then_answers(one_result, monkeypatch):
    model = fake_chat_model("Yes, Article 4 [1].")
    install_chat_model(monkeypatch, lambda *_: model)

    state = await chat_graph.ainvoke(ChatState(question=QUESTION))

    assert state["answer"] == "Yes, Article 4 [1]."
    assert state["sources"] == (search_result(),)
    assert one_result == [SearchRequest(query=QUESTION, limit=config.CHAT_SOURCES)]


# The refusal gate


async def test_a_question_the_corpus_does_not_cover_is_refused_before_any_model_call(monkeypatch):
    searches: list[SearchRequest] = []

    async def junk_search(session, request):
        searches.append(request)
        return (search_result(cosine_similarity=0.2, reranker_relevance=0.3),)

    model = fake_chat_model()
    monkeypatch.setattr("app.chat.nodes.retrieve.search", junk_search)
    install_chat_model(monkeypatch, lambda *_: model)

    state = await chat_graph.ainvoke(ChatState(question="What is the best pizza topping?"))

    assert state["answer"] == REFUSAL_ANSWER
    assert state["sources"] == ()
    assert state["refusal"] == Refusal(reason=RefusalReason.NOTHING_RETRIEVED)
    assert model.received == []
    assert len(searches) == 1


async def test_a_refused_question_still_keeps_what_search_found(monkeypatch):
    """The hits the gate judged stay on the state, so a refusal can be told from a miss:
    what search found, and how it scored, is what an eval reads a too-tight gate from."""
    junk = (search_result(cosine_similarity=0.2, reranker_relevance=0.3),)

    async def junk_search(session, request):
        return junk

    monkeypatch.setattr("app.chat.nodes.retrieve.search", junk_search)
    install_chat_model(monkeypatch, lambda *_: fake_chat_model())

    state = await chat_graph.ainvoke(ChatState(question="What is the best pizza topping?"))

    assert state["hits"] == junk
    assert state["sources"] == ()


async def test_an_empty_search_is_refused_before_any_model_call(monkeypatch):
    async def nothing(session, request):
        return ()

    model = fake_chat_model()
    monkeypatch.setattr("app.chat.nodes.retrieve.search", nothing)
    install_chat_model(monkeypatch, lambda *_: model)

    state = await chat_graph.ainvoke(ChatState(question=QUESTION))

    assert state["answer"] == REFUSAL_ANSWER
    assert model.received == []


async def test_a_refused_question_is_not_widened_to_sections(monkeypatch):
    async def junk_search(session, request):
        return (search_result(cosine_similarity=0.2, reranker_relevance=0.3),)

    async def refuse_to_expand(session, chunks, *, limit):
        raise AssertionError("expansion ran for a question the gate refused")

    monkeypatch.setattr(config, "EXPAND_SECTIONS", True)
    monkeypatch.setattr("app.chat.nodes.retrieve.search", junk_search)
    monkeypatch.setattr("app.chat.nodes.retrieve.expand_sections", refuse_to_expand)
    install_chat_model(monkeypatch, lambda *_: fake_chat_model())

    state = await chat_graph.ainvoke(ChatState(question=QUESTION))

    assert state["answer"] == REFUSAL_ANSWER


def test_the_compiled_graph_has_the_edges_the_readme_draws():
    """The README's diagram is hand-drawn, so the edge list it was drawn from is asserted
    here: an edge added to the graph fails this until the drawing catches up."""
    edges = {(edge.source, edge.target) for edge in chat_graph.get_graph().edges}

    assert edges == {(source, target) for source, target in GRAPH_EDGES}
