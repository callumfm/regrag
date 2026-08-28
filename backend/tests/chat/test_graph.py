"""Chat graph: state flow, search passthrough, prompt assembly, error wrapping."""

from collections.abc import AsyncIterator
from typing import Any

import httpx
import litellm
import openai
import pytest
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatResult

from app.chat.enums import ChatNode, ToolStep
from app.chat.graph import GRAPH_EDGES, chat_graph, merge_sources
from app.chat.models import ChatState, ToolCall
from app.chat.prompts import ASSESS_SYSTEM_PROMPT, REFUSAL_ANSWER
from app.core.config import config
from app.core.llm import LLMError
from app.retrieval.models import SearchRequest
from tests.chat.conftest import (
    ANSWER,
    THINKING,
    USAGE,
    RecordingChatModel,
    fake_chat_model,
    tool_call_message,
)
from tests.conftest import search_result

pytestmark = pytest.mark.anyio

QUESTION = "What is the GHG intensity limit?"


def rate_limited() -> openai.RateLimitError:
    request = httpx.Request("POST", "https://api.anthropic.example")
    return openai.RateLimitError(
        message="provider said no", response=httpx.Response(429, request=request), body=None
    )


class FailingModel(RecordingChatModel):
    """Refuses the first `failures` prompts as a rate limit, then answers."""

    failures: int = 1

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        if len(self.received) < self.failures:
            self.received.append(list(messages))
            raise rate_limited()
        return super()._generate(messages, *args, **kwargs)


async def test_graph_retrieves_then_answers(one_result, monkeypatch):
    model = fake_chat_model("Yes, Article 4 [1].")
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    state = await chat_graph.ainvoke(ChatState(question=QUESTION))

    assert state["answer"] == "Yes, Article 4 [1]."
    assert state["sources"] == (search_result(),)
    assert one_result == [SearchRequest(query=QUESTION, limit=config.CHAT_SOURCES)]


async def test_model_receives_system_prompt_and_numbered_context(monkeypatch):
    async def fake_search(session, request):
        return (search_result(text="A very specific clause."),)

    model = fake_chat_model()
    monkeypatch.setattr("app.chat.graph.search", fake_search)
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    await chat_graph.ainvoke(ChatState(question=QUESTION))

    (prompt,) = model.received
    assert isinstance(prompt[0], SystemMessage)
    assert "[1] (32023R1805, Article 4(1))" in prompt[1].content
    assert "A very specific clause." in prompt[1].content


async def test_a_transient_provider_failure_is_retried(one_result, monkeypatch):
    model = FailingModel(messages=iter(["Second time lucky [1]."]), failures=1)
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    state = await chat_graph.ainvoke(ChatState(question=QUESTION))

    assert state["answer"] == "Second time lucky [1]."
    assert len(model.received) == 2


async def test_a_persistent_provider_failure_becomes_a_transient_llm_error(one_result, monkeypatch):
    model = FailingModel(messages=iter([]), failures=10)
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    with pytest.raises(LLMError) as exc_info:
        await chat_graph.ainvoke(ChatState(question=QUESTION))

    assert exc_info.value.transient is True
    assert str(exc_info.value) == "chat call failed"
    assert len(model.received) == 3


def streamed_text(data: Any) -> str:
    """The text of one messages-mode stream item, a (chunk, metadata) pair."""
    chunk, _ = data
    return chunk.text


def litellm_stream(
    monkeypatch, *deltas: dict[str, Any], usage: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    """Stand litellm's completion call in with these deltas — and, as litellm reports it
    when asked, a trailing usage-only chunk; the calls made are returned."""
    calls: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        calls.append(kwargs)

        async def chunks() -> AsyncIterator[dict[str, Any]]:
            for delta in deltas:
                yield {"choices": [{"delta": delta, "finish_reason": None}]}
            if usage:
                yield {"choices": [], "usage": usage}

        return chunks()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    return calls


async def test_the_chat_client_streams_one_token_per_litellm_delta(one_result, monkeypatch):
    """The seam below the fakes: chat_model()'s ChatLiteLLM asks litellm to stream, and
    the graph's message stream sees each delta as it lands."""
    calls = litellm_stream(
        monkeypatch, {"role": "assistant", "content": "Ships must "}, {"content": "comply [1]."}
    )

    texts = [
        streamed_text(data)
        async for mode, data in chat_graph.astream(
            ChatState(question=QUESTION), stream_mode=["updates", "messages"]
        )
        if mode == "messages"
    ]

    assert calls[0]["stream"] is True
    assert calls[0]["model"] == config.CHAT_MODEL
    assert [text for text in texts if text] == ["Ships must ", "comply [1]."]


async def test_the_chat_client_asks_litellm_for_usage_and_the_node_records_it(
    one_result, monkeypatch
):
    """litellm strips usage from streamed chunks unless asked for it in stream_options;
    asked, it sends one usage-only chunk last, which becomes synthesize's tokens."""
    calls = litellm_stream(
        monkeypatch,
        {"role": "assistant", "content": "Ships must comply [1]."},
        usage={"prompt_tokens": 1500, "completion_tokens": 40, "total_tokens": 1540},
    )

    state = ChatState.model_validate(await chat_graph.ainvoke(ChatState(question=QUESTION)))

    assert calls[0]["stream_options"] == {"include_usage": True}
    [_retrieve, synthesize] = state.steps
    assert (synthesize.input_tokens, synthesize.output_tokens) == (1500, 40)


async def test_the_chat_client_answers_with_the_text_of_a_reasoning_response(
    one_result, monkeypatch
):
    """litellm's reasoning_content becomes a thinking block ahead of the text; the answer
    is the text alone, not the repr of the block list."""
    litellm_stream(
        monkeypatch,
        {"role": "assistant", "content": "", "reasoning_content": THINKING},
        {"content": ANSWER},
    )

    state = await chat_graph.ainvoke(ChatState(question=QUESTION))

    assert state["answer"] == ANSWER


async def test_retrieve_widens_what_search_found_to_whole_sections(one_result, monkeypatch):
    """The graph hands search's hits to expansion, so the prompt sees whole sections."""
    widened = (search_result(id=2, citation="Article 4(2)", text="The limit is 91,16 gCO2e/MJ."),)

    async def fake_expand(session, chunks, *, limit):
        assert tuple(chunks) == (search_result(),)
        assert limit == config.CHAT_CONTEXT_CHUNKS
        return widened

    monkeypatch.setattr(config, "EXPAND_SECTIONS", True)
    monkeypatch.setattr("app.chat.graph.expand_sections", fake_expand)
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: fake_chat_model())

    state = await chat_graph.ainvoke(ChatState(question=QUESTION))

    assert state["sources"] == widened
    assert state["hits"] == (search_result(),)


async def test_retrieve_leaves_search_alone_when_expansion_is_off(one_result, monkeypatch):
    """The off switch skips the widening query, not just its result."""

    async def refuse(session, chunks, *, limit):
        raise AssertionError("expansion ran with EXPAND_SECTIONS off")

    monkeypatch.setattr("app.chat.graph.expand_sections", refuse)
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: fake_chat_model())

    state = await chat_graph.ainvoke(ChatState(question=QUESTION))

    assert state["sources"] == (search_result(),)


# The refusal gate


async def test_a_question_the_corpus_does_not_cover_is_refused_before_any_model_call(monkeypatch):
    searches: list[SearchRequest] = []

    async def junk_search(session, request):
        searches.append(request)
        return (search_result(cosine_similarity=0.2, reranker_relevance=0.3),)

    model = fake_chat_model()
    monkeypatch.setattr("app.chat.graph.search", junk_search)
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    state = await chat_graph.ainvoke(ChatState(question="What is the best pizza topping?"))

    assert state["answer"] == REFUSAL_ANSWER
    assert state["sources"] == ()
    assert model.received == []
    assert len(searches) == 1


async def test_a_refused_question_still_keeps_what_search_found(monkeypatch):
    """The hits the gate judged stay on the state, so a refusal can be told from a miss:
    what search found, and how it scored, is what an eval reads a too-tight gate from."""
    junk = (search_result(cosine_similarity=0.2, reranker_relevance=0.3),)

    async def junk_search(session, request):
        return junk

    monkeypatch.setattr("app.chat.graph.search", junk_search)
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: fake_chat_model())

    state = await chat_graph.ainvoke(ChatState(question="What is the best pizza topping?"))

    assert state["hits"] == junk
    assert state["sources"] == ()


async def test_an_empty_search_is_refused_before_any_model_call(monkeypatch):
    async def nothing(session, request):
        return ()

    model = fake_chat_model()
    monkeypatch.setattr("app.chat.graph.search", nothing)
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    state = await chat_graph.ainvoke(ChatState(question=QUESTION))

    assert state["answer"] == REFUSAL_ANSWER
    assert model.received == []


async def test_a_refused_question_is_not_widened_to_sections(monkeypatch):
    async def junk_search(session, request):
        return (search_result(cosine_similarity=0.2, reranker_relevance=0.3),)

    async def refuse_to_expand(session, chunks, *, limit):
        raise AssertionError("expansion ran for a question the gate refused")

    monkeypatch.setattr(config, "EXPAND_SECTIONS", True)
    monkeypatch.setattr("app.chat.graph.search", junk_search)
    monkeypatch.setattr("app.chat.graph.expand_sections", refuse_to_expand)
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: fake_chat_model())

    state = await chat_graph.ainvoke(ChatState(question=QUESTION))

    assert state["answer"] == REFUSAL_ANSWER


@pytest.fixture
def answer_model(monkeypatch):
    model = fake_chat_model("Answered [1].")
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)
    return model


async def run_graph() -> ChatState:
    """The graph run, folded back onto the state it started from."""
    state = ChatState(question=QUESTION)
    state.sync_from_snapshot(await chat_graph.ainvoke(state))
    return state


class TestMergeSources:
    def test_appends_new_chunks_after_existing_in_arrival_order(self):
        existing = (search_result(id=1),)
        additions = [search_result(id=2), search_result(id=3)]

        merged = merge_sources(existing, additions, cap=10)

        assert tuple(chunk.id for chunk in merged) == (1, 2, 3)

    def test_deduplicates_by_chunk_id_keeping_the_earlier_chunk(self):
        existing = (search_result(id=1, text="first form"),)
        additions = [search_result(id=1, text="refetched form"), search_result(id=2)]

        merged = merge_sources(existing, additions, cap=10)

        assert tuple(chunk.id for chunk in merged) == (1, 2)
        assert merged[0].text == "first form"

    def test_stops_appending_at_the_cap_so_earlier_context_wins(self):
        existing = (search_result(id=1), search_result(id=2))
        additions = [search_result(id=3), search_result(id=4)]

        merged = merge_sources(existing, additions, cap=3)

        assert tuple(chunk.id for chunk in merged) == (1, 2, 3)

    def test_existing_beyond_the_cap_is_kept_but_nothing_is_added(self):
        existing = (search_result(id=1), search_result(id=2))

        merged = merge_sources(existing, [search_result(id=3)], cap=2)

        assert tuple(chunk.id for chunk in merged) == (1, 2)


class TestAssessLoop:
    async def test_no_tool_calls_goes_straight_to_synthesize(
        self, loop_on, one_result, answer_model, assess_turns
    ):
        assess_turns(AIMessage(content=""))

        state = await run_graph()

        assert state.answer == "Answered [1]."
        assert [r.step for r in state.steps] == [
            ChatNode.RETRIEVE,
            ChatNode.ASSESS,
            ChatNode.SYNTHESIZE,
        ]

    async def test_a_tool_round_merges_its_chunks_then_answers(
        self, loop_on, one_result, answer_model, assess_turns, tool_results
    ):
        assess_turns(
            tool_call_message("follow_reference", {"celex": "32023R1805", "article": "6"}),
            AIMessage(content=""),
        )
        run_calls = tool_results(search_result(id=42, citation="Article 6"))

        state = await run_graph()

        assert [r.step for r in state.steps] == [
            ChatNode.RETRIEVE,
            ChatNode.ASSESS,
            ToolStep.FOLLOW_REFERENCE,
            ChatNode.ASSESS,
            ChatNode.SYNTHESIZE,
        ]
        assert run_calls == [
            ToolCall(name="follow_reference", args={"celex": "32023R1805", "article": "6"})
        ]
        assert tuple(chunk.id for chunk in state.sources) == (1, 42)
        assert state.pending_calls == ()

    async def test_the_round_cap_forces_synthesis_with_calls_still_pending(
        self, loop_on, one_result, answer_model, assess_turns, tool_results, monkeypatch
    ):
        monkeypatch.setattr(config, "ASSESS_MAX_ROUNDS", 1)
        assess_turns(
            tool_call_message("search", {"query": "first gap"}),
            tool_call_message("search", {"query": "never runs"}),
        )
        tool_results(search_result(id=2))

        state = await run_graph()

        assert [r.step for r in state.steps] == [
            ChatNode.RETRIEVE,
            ChatNode.ASSESS,
            ToolStep.SEARCH,
            ChatNode.SYNTHESIZE,
        ]

    async def test_a_call_to_a_tool_the_surface_does_not_have_is_still_a_step(
        self, loop_on, one_result, answer_model, assess_turns, tool_results
    ):
        """A model asking for a tool that does not exist is worth reading off the path,
        and the round it spent still shows there."""
        assess_turns(tool_call_message("summarize", {"query": "gap"}), AIMessage(content=""))
        tool_results()

        state = await run_graph()

        assert [r.step for r in state.steps] == [
            ChatNode.RETRIEVE,
            ChatNode.ASSESS,
            ToolStep.UNKNOWN,
            ChatNode.ASSESS,
            ChatNode.SYNTHESIZE,
        ]
        assert state.answer == "Answered [1]."

    async def test_each_tool_call_of_a_round_is_timed_as_its_own_step(
        self, loop_on, one_result, answer_model, assess_turns, tool_results
    ):
        """Two calls in one round leave two steps, so a slow round names the call that
        was slow rather than reporting the pair as one number."""
        assess_turns(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search", "args": {"query": "a"}, "id": "c1", "type": "tool_call"},
                    {
                        "name": "follow_reference",
                        "args": {"celex": "32023R1805", "article": "6"},
                        "id": "c2",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content=""),
        )
        tool_results(search_result(id=7))

        state = await run_graph()

        assert [r.step for r in state.steps] == [
            ChatNode.RETRIEVE,
            ChatNode.ASSESS,
            ToolStep.SEARCH,
            ToolStep.FOLLOW_REFERENCE,
            ChatNode.ASSESS,
            ChatNode.SYNTHESIZE,
        ]
        assert state.assess_rounds() == 2

    async def test_each_assess_visit_records_its_own_usage(
        self, loop_on, one_result, answer_model, assess_turns, tool_results
    ):
        assess_turns(tool_call_message("search", {"query": "gap"}), AIMessage(content=""))
        tool_results()

        state = await run_graph()

        assesses = [r for r in state.steps if r.step is ChatNode.ASSESS]
        assert len(assesses) == 2
        assert all(r.input_tokens == USAGE["input_tokens"] for r in assesses)

    async def test_assess_sees_the_question_and_numbered_context(
        self, loop_on, one_result, answer_model, assess_turns
    ):
        assess = assess_turns(AIMessage(content=""))

        await run_graph()

        (prompt,) = assess.received
        assert prompt[0].content == ASSESS_SYSTEM_PROMPT
        assert "[1] (32023R1805" in prompt[1].content
        assert str(prompt[1].content).endswith(f"Question: {QUESTION}")

    async def test_a_gated_question_still_refuses_without_any_model_call(
        self, loop_on, monkeypatch
    ):
        async def empty_search(session, request):
            return ()

        monkeypatch.setattr("app.chat.graph.search", empty_search)

        state = await run_graph()

        assert state.answer == REFUSAL_ANSWER
        assert [r.step for r in state.steps] == [ChatNode.RETRIEVE, ChatNode.REFUSE]

    async def test_a_persistently_failing_assess_call_still_synthesizes_from_the_context(
        self, loop_on, one_result, answer_model, monkeypatch
    ):
        """An assess round is best-effort: it must never destroy a request that already
        has answerable context, even when the model keeps failing."""
        assess = FailingModel(messages=iter([]), failures=10, usage=USAGE)
        monkeypatch.setattr("app.chat.graph.assess_model", lambda: assess)

        state = await run_graph()

        assert state.answer == "Answered [1]."
        assert [r.step for r in state.steps] == [
            ChatNode.RETRIEVE,
            ChatNode.ASSESS,
            ChatNode.SYNTHESIZE,
        ]

    async def test_an_assess_turn_asking_for_more_than_the_cap_runs_only_the_cap(
        self, loop_on, one_result, answer_model, assess_turns, tool_results, monkeypatch
    ):
        monkeypatch.setattr(config, "ASSESS_MAX_CALLS", 1)
        assess_turns(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search", "args": {"query": "a"}, "id": "call_1", "type": "tool_call"},
                    {"name": "search", "args": {"query": "b"}, "id": "call_2", "type": "tool_call"},
                ],
            ),
            AIMessage(content=""),
        )
        run_calls = tool_results()

        await run_graph()

        assert run_calls == [ToolCall(name="search", args={"query": "a"})]

    async def test_growth_is_budgeted_from_the_context_retrieve_left(
        self, loop_on, one_result, answer_model, assess_turns, tool_results, monkeypatch
    ):
        """The budget is what the loop may add, not a ceiling the initial context is assumed
        to already fill: retrieve left one chunk, so two more is all two rounds may append."""
        monkeypatch.setattr(config, "ASSESS_EXTRA_CHUNKS", 2)
        monkeypatch.setattr(config, "CHAT_CONTEXT_CHUNKS", 15)
        assess_turns(
            tool_call_message("search", {"query": "gap"}),
            tool_call_message("search", {"query": "more"}),
        )
        tool_results(search_result(id=2), search_result(id=3), search_result(id=4))

        state = await run_graph()

        assert state.retrieved_sources == 1
        assert tuple(chunk.id for chunk in state.sources) == (1, 2, 3)

    async def test_a_zero_budget_reads_the_context_without_growing_it(
        self, loop_on, one_result, answer_model, assess_turns, tool_results, monkeypatch
    ):
        monkeypatch.setattr(config, "ASSESS_EXTRA_CHUNKS", 0)
        assess_turns(tool_call_message("search", {"query": "gap"}), AIMessage(content=""))
        tool_results(search_result(id=2))

        state = await run_graph()

        assert tuple(chunk.id for chunk in state.sources) == (1,)


def test_the_compiled_graph_has_the_edges_the_readme_draws():
    """The README's diagram is hand-drawn, so the edge list it was drawn from is asserted
    here: an edge added to the graph fails this until the drawing catches up."""
    edges = {(edge.source, edge.target) for edge in chat_graph.get_graph().edges}

    assert edges == {(source, target) for source, target in GRAPH_EDGES}
