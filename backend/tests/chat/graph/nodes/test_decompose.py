"""decompose: the split it makes, the model it is bound to, and its best-effort failure."""

import json

import pytest
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableBinding
from langchain_litellm import ChatLiteLLM
from pydantic import ValidationError

from app.chat.enums import ChatNode
from app.chat.graph.nodes.decompose import (
    DECOMPOSE_SYSTEM_PROMPT,
    DecomposedQuestion,
    decompose,
    decompose_model,
)
from app.chat.graph.nodes.refuse import REFUSAL_ANSWER
from app.chat.graph.service import chat_graph
from app.chat.models import ChatState
from app.core.config import config
from app.core.llm.models import TokenUsage
from app.retrieval.models import SearchRequest
from tests.chat.conftest import (
    QUESTION,
    TOKEN_USAGE,
    FailingModel,
    fake_chat_model,
    hits_for,
    litellm_completion,
    run_graph,
    split_message,
)
from tests.conftest import install_chat_model, search_result

pytestmark = pytest.mark.anyio


async def test_the_decompose_client_sends_the_output_format_and_the_node_records_usage(
    monkeypatch,
):
    """The format is bound on the client and must reach litellm as response_format; the
    blocking answer carries usage on the message, which becomes the step's tokens."""
    calls = litellm_completion(
        monkeypatch,
        json.dumps({"queries": ["what is A", "what is B"]}),
        usage={"prompt_tokens": 120, "completion_tokens": 20, "total_tokens": 140},
    )

    update = await decompose(ChatState(question="What are A and B?"))

    assert calls[0]["response_format"] is DecomposedQuestion
    assert calls[0]["stream"] is False
    assert update["queries"] == ("what is A", "what is B")
    [step] = update["steps"]
    assert step.usage == TokenUsage(input_tokens=120, output_tokens=20)


class TestDecomposeInTheGraph:
    async def test_off_records_no_decompose_step_and_searches_the_question(
        self, one_result, answer_model
    ):
        state = await run_graph()

        assert [r.step for r in state.steps] == [ChatNode.RETRIEVE, ChatNode.SYNTHESIZE]
        assert one_result == [SearchRequest(query=QUESTION, limit=config.CHAT_SOURCES)]
        assert state.queries == ()

    async def test_on_a_split_question_searches_each_part_then_answers_the_whole(
        self, decompose_on, answer_model, decompose_turns, monkeypatch
    ):
        decompose_turns(split_message("what is A", "what is B"))
        fake_search, requests = hits_for(
            **{"what is A": (search_result(id=1),), "what is B": (search_result(id=2),)}
        )
        monkeypatch.setattr("app.chat.graph.nodes.retrieve.search", fake_search)

        state = ChatState(question="What are A and B?")
        state.sync_from_snapshot(await chat_graph.ainvoke(state))

        assert [r.step for r in state.steps] == [
            ChatNode.DECOMPOSE,
            ChatNode.RETRIEVE,
            ChatNode.SYNTHESIZE,
        ]
        assert state.queries == ("what is A", "what is B")
        assert {r.query for r in requests} == {"what is A", "what is B"}
        assert tuple(chunk.id for chunk in state.sources) == (1, 2)
        [messages] = answer_model.received
        assert messages[1].content.endswith("Question: What are A and B?")

    async def test_on_a_single_part_question_is_searched_as_asked(
        self, decompose_on, one_result, answer_model, decompose_turns
    ):
        """The only difference from the switch being off is the recorded step."""
        decompose_turns(split_message(QUESTION))

        state = await run_graph()

        assert [r.step for r in state.steps] == [
            ChatNode.DECOMPOSE,
            ChatNode.RETRIEVE,
            ChatNode.SYNTHESIZE,
        ]
        assert one_result == [SearchRequest(query=QUESTION, limit=config.CHAT_SOURCES)]
        assert state.queries == ()

    async def test_on_a_split_whose_every_part_misses_the_bar_is_refused(
        self, decompose_on, decompose_turns, monkeypatch
    ):
        decompose_turns(split_message("pizza", "pasta"))
        junk = search_result(cosine_similarity=0.2, reranker_relevance=0.3)
        fake_search, _ = hits_for(pizza=(junk,), pasta=(junk,))
        model = fake_chat_model()
        monkeypatch.setattr("app.chat.graph.nodes.retrieve.search", fake_search)
        install_chat_model(monkeypatch, lambda *_: model)

        state = await chat_graph.ainvoke(ChatState(question="Best pizza and pasta?"))

        assert state["answer"] == REFUSAL_ANSWER
        assert model.received == []


class TestDecompose:
    async def test_a_multi_part_question_becomes_one_query_per_part(self, decompose_turns):
        model = decompose_turns(split_message("what is A", "what is B"))

        update = await decompose(ChatState(question="What are A and B?"))

        assert update["queries"] == ("what is A", "what is B")
        [messages] = model.received
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == DECOMPOSE_SYSTEM_PROMPT
        assert messages[1].content == "What are A and B?"

    async def test_a_single_part_question_leaves_queries_empty(self, decompose_turns):
        """One query back means the question asked one thing; the original text is what
        retrieve searches, so a lightly rephrased echo cannot change retrieval."""
        decompose_turns(split_message("What is the GHG intensity limit, rephrased?"))

        update = await decompose(ChatState(question=QUESTION))

        assert update["queries"] == ()

    async def test_surplus_parts_are_truncated_to_the_cap(self, decompose_turns, monkeypatch):
        monkeypatch.setattr(config, "DECOMPOSE_MAX_PARTS", 2)
        decompose_turns(split_message("a", "b", "c"))

        update = await decompose(ChatState(question="a, b and c?"))

        assert update["queries"] == ("a", "b")

    async def test_an_answer_off_the_schema_falls_back_to_the_question(
        self, decompose_turns, caplog
    ):
        decompose_turns(AIMessage(content="I'd split this into two."))

        update = await decompose(ChatState(question=QUESTION))

        assert update["queries"] == ()
        assert "decompose answered off its schema" in caplog.text

    async def test_a_failing_call_falls_back_to_the_question(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "app.chat.graph.nodes.decompose.decompose_model",
            lambda: FailingModel(messages=iter([]), failures=9),
        )

        update = await decompose(ChatState(question=QUESTION))

        assert update["queries"] == ()
        assert "decompose call failed" in caplog.text

    async def test_the_step_records_the_calls_usage(self, decompose_turns):
        decompose_turns(split_message("what is A", "what is B"))

        update = await decompose(ChatState(question="What are A and B?"))

        [step] = update["steps"]
        assert step.step is ChatNode.DECOMPOSE
        assert step.usage == TOKEN_USAGE


def test_decompose_is_built_on_its_own_model_with_the_output_format_bound(monkeypatch):
    """The split is asked for in the DecomposedQuestion shape, on a model set apart from
    the answer's and assess's, as one setting per role requires."""
    monkeypatch.setattr(config, "CHAT_MODEL", "anthropic/answer-model")
    monkeypatch.setattr(config, "DECOMPOSE_MODEL", "anthropic/decompose-model")

    binding = decompose_model()
    assert isinstance(binding, RunnableBinding)
    model = binding.bound
    assert isinstance(model, ChatLiteLLM)
    assert model.model == "anthropic/decompose-model"
    assert model.streaming is False
    assert binding.kwargs["response_format"] is DecomposedQuestion


def test_a_decomposed_question_is_frozen_and_holds_its_queries_in_order():
    split = DecomposedQuestion(queries=("what is A", "what is B"))

    assert split.queries == ("what is A", "what is B")
    with pytest.raises(ValidationError):
        split.queries = ()  # type: ignore
