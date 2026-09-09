"""Chat graph: state flow, search passthrough, prompt assembly, error wrapping."""

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import litellm
import openai
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableBinding
from langchain_litellm import ChatLiteLLM

from app.chat.enums import ChatNode, RefusalReason, ToolStep
from app.chat.graph import GRAPH_EDGES, chat_graph
from app.chat.models import (
    ChatState,
    ChatTurn,
    DecomposedQuestion,
    Refusal,
    StandaloneQuestion,
    ToolCall,
)
from app.chat.nodes.assess import (
    ASSESS_SYSTEM_PROMPT,
    assess_model,
    build_assess_system_prompt,
    merge_sources,
)
from app.chat.nodes.decompose import DECOMPOSE_SYSTEM_PROMPT, decompose, decompose_model
from app.chat.nodes.retrieve import interleave_by_rank, retrieve
from app.chat.nodes.rewrite import REWRITE_SYSTEM_PROMPT, rewrite, rewrite_model
from app.chat.nodes.synthesize import SYSTEM_PROMPT
from app.chat.prompts import REFUSAL_ANSWER, THREAD_NOTE, system_prompt
from app.core.config import config
from app.core.llm import LLMError
from app.retrieval.models import SearchRequest
from tests.chat.conftest import (
    ANSWER,
    THINKING,
    USAGE,
    RecordingChatModel,
    fake_chat_model,
    restated_message,
    split_message,
    tool_call_message,
)
from tests.conftest import install_chat_model, search_result

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
    install_chat_model(monkeypatch, lambda *_: model)

    state = await chat_graph.ainvoke(ChatState(question=QUESTION))

    assert state["answer"] == "Yes, Article 4 [1]."
    assert state["sources"] == (search_result(),)
    assert one_result == [SearchRequest(query=QUESTION, limit=config.CHAT_SOURCES)]


async def test_model_receives_system_prompt_and_numbered_context(monkeypatch):
    async def fake_search(session, request):
        return (search_result(text="A very specific clause."),)

    model = fake_chat_model()
    monkeypatch.setattr("app.chat.nodes.retrieve.search", fake_search)
    install_chat_model(monkeypatch, lambda *_: model)

    await chat_graph.ainvoke(ChatState(question=QUESTION))

    (prompt,) = model.received
    assert isinstance(prompt[0], SystemMessage)
    assert "[1] (32023R1805, Article 4(1))" in prompt[1].content
    assert "A very specific clause." in prompt[1].content


async def test_a_transient_provider_failure_is_retried(one_result, monkeypatch):
    model = FailingModel(messages=iter(["Second time lucky [1]."]), failures=1)
    install_chat_model(monkeypatch, lambda *_: model)

    state = await chat_graph.ainvoke(ChatState(question=QUESTION))

    assert state["answer"] == "Second time lucky [1]."
    assert len(model.received) == 2


async def test_a_persistent_provider_failure_becomes_a_transient_llm_error(one_result, monkeypatch):
    model = FailingModel(messages=iter([]), failures=10)
    install_chat_model(monkeypatch, lambda *_: model)

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


def test_assess_is_built_on_its_own_model_rather_than_the_answer_model(monkeypatch):
    """The two calls do different jobs and are measured against different things, so the
    model reviewing the context is set apart from the one writing the answer."""
    monkeypatch.setattr(config, "CHAT_MODEL", "anthropic/answer-model")
    monkeypatch.setattr(config, "ASSESS_MODEL", "anthropic/assess-model")

    binding = assess_model()
    assert isinstance(binding, RunnableBinding)
    model = binding.bound
    assert isinstance(model, ChatLiteLLM)
    assert model.model == "anthropic/assess-model"


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


def litellm_completion(monkeypatch, content: str, usage: dict[str, int]) -> list[dict[str, Any]]:
    """Stand litellm's completion call in for one blocking answer, returning the calls made."""
    calls: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "choices": [
                {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": usage,
        }

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    return calls


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
    assert (step.input_tokens, step.output_tokens) == (120, 20)


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
    monkeypatch.setattr("app.chat.nodes.retrieve.expand_sections", fake_expand)
    install_chat_model(monkeypatch, lambda *_: fake_chat_model())

    state = await chat_graph.ainvoke(ChatState(question=QUESTION))

    assert state["sources"] == widened
    assert state["hits"] == (search_result(),)


async def test_retrieve_leaves_search_alone_when_expansion_is_off(one_result, monkeypatch):
    """The off switch skips the widening query, not just its result."""

    async def refuse(session, chunks, *, limit):
        raise AssertionError("expansion ran with EXPAND_SECTIONS off")

    monkeypatch.setattr("app.chat.nodes.retrieve.expand_sections", refuse)
    install_chat_model(monkeypatch, lambda *_: fake_chat_model())

    state = await chat_graph.ainvoke(ChatState(question=QUESTION))

    assert state["sources"] == (search_result(),)


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


@pytest.fixture
def answer_model(monkeypatch):
    model = fake_chat_model("Answered [1].")
    install_chat_model(monkeypatch, lambda *_: model)
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


def hits_for(**per_query: tuple) -> tuple[Callable, list[SearchRequest]]:
    """A search answering each query with its own hits, recording the requests made."""
    requests: list[SearchRequest] = []

    async def fake_search(session, request):
        requests.append(request)
        return per_query[request.query]

    return fake_search, requests


class TestRetrieveOverQueries:
    async def test_each_query_is_searched_and_the_hits_interleaved(self, monkeypatch):
        fake_search, requests = hits_for(
            a=(search_result(id=1), search_result(id=2)), b=(search_result(id=3),)
        )
        monkeypatch.setattr("app.chat.nodes.retrieve.search", fake_search)

        update = await retrieve(ChatState(question="A and B?", queries=("a", "b")))

        assert {r.query for r in requests} == {"a", "b"}
        assert all(r.limit == config.CHAT_SOURCES for r in requests)
        assert tuple(chunk.id for chunk in update["hits"]) == (1, 3, 2)
        assert tuple(chunk.id for chunk in update["sources"]) == (1, 3, 2)
        assert update["retrieved_sources"] == 3

    async def test_a_query_below_the_bar_keeps_its_hits_but_adds_no_sources(self, monkeypatch):
        """The out-of-corpus part cannot admit sub-bar hits to the context, yet what search
        found for it stays on the state so the split can be read against it."""
        junk = search_result(id=9, cosine_similarity=0.2, reranker_relevance=0.3)
        fake_search, _ = hits_for(a=(search_result(id=1),), b=(junk,))
        monkeypatch.setattr("app.chat.nodes.retrieve.search", fake_search)

        update = await retrieve(ChatState(question="A and B?", queries=("a", "b")))

        assert tuple(chunk.id for chunk in update["hits"]) == (1, 9)
        assert tuple(chunk.id for chunk in update["sources"]) == (1,)

    async def test_no_query_clearing_the_bar_leaves_the_context_empty(self, monkeypatch):
        junk = search_result(cosine_similarity=0.2, reranker_relevance=0.3)
        fake_search, _ = hits_for(a=(junk,), b=(junk,))
        monkeypatch.setattr("app.chat.nodes.retrieve.search", fake_search)

        update = await retrieve(ChatState(question="A and B?", queries=("a", "b")))

        assert update["sources"] == ()
        assert update["retrieved_sources"] == 0
        assert update["hits"] == (junk,)

    async def test_no_queries_searches_the_question_as_asked(self, one_result):
        update = await retrieve(ChatState(question=QUESTION))

        assert one_result == [SearchRequest(query=QUESTION, limit=config.CHAT_SOURCES)]
        assert update["sources"] == (search_result(),)

    async def test_expansion_widens_the_interleaved_survivors(self, monkeypatch):
        fake_search, _ = hits_for(a=(search_result(id=1),), b=(search_result(id=2),))
        widened: list[tuple[int, ...]] = []

        async def fake_expand(session, chunks, *, limit):
            widened.append(tuple(chunk.id for chunk in chunks))
            return (*chunks, search_result(id=3))

        monkeypatch.setattr(config, "EXPAND_SECTIONS", True)
        monkeypatch.setattr("app.chat.nodes.retrieve.search", fake_search)
        monkeypatch.setattr("app.chat.nodes.retrieve.expand_sections", fake_expand)

        update = await retrieve(ChatState(question="A and B?", queries=("a", "b")))

        assert widened == [(1, 2)]
        assert tuple(chunk.id for chunk in update["sources"]) == (1, 2, 3)


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
        monkeypatch.setattr("app.chat.nodes.retrieve.search", fake_search)

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
        monkeypatch.setattr("app.chat.nodes.retrieve.search", fake_search)
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
            "app.chat.nodes.decompose.decompose_model",
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
        assert (step.input_tokens, step.output_tokens) == (
            USAGE["input_tokens"],
            USAGE["output_tokens"],
        )


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


def test_rewrite_is_built_on_its_own_model_with_the_output_format_bound(monkeypatch):
    """The restatement is asked for in the StandaloneQuestion shape, on a model set apart
    from the answer's and assess's, as one setting per role requires."""
    monkeypatch.setattr(config, "CHAT_MODEL", "anthropic/answer-model")
    monkeypatch.setattr(config, "REWRITE_MODEL", "anthropic/rewrite-model")

    binding = rewrite_model()
    assert isinstance(binding, RunnableBinding)
    model = binding.bound
    assert isinstance(model, ChatLiteLLM)
    assert model.model == "anthropic/rewrite-model"
    assert model.streaming is False
    assert binding.kwargs["response_format"] is StandaloneQuestion


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
        assert str(prompt[0].content).startswith(ASSESS_SYSTEM_PROMPT)
        assert "[1] (32023R1805" in prompt[1].content
        assert str(prompt[1].content).endswith(f"Question: {QUESTION}")

    async def test_a_gated_question_still_refuses_without_any_model_call(
        self, loop_on, monkeypatch
    ):
        async def empty_search(session, request):
            return ()

        monkeypatch.setattr("app.chat.nodes.retrieve.search", empty_search)

        state = await run_graph()

        assert state.answer == REFUSAL_ANSWER
        assert [r.step for r in state.steps] == [ChatNode.RETRIEVE, ChatNode.REFUSE]

    async def test_a_persistently_failing_assess_call_still_synthesizes_from_the_context(
        self, loop_on, one_result, answer_model, monkeypatch
    ):
        """An assess round is best-effort: it must never destroy a request that already
        has answerable context, even when the model keeps failing."""
        assess = FailingModel(messages=iter([]), failures=10, usage=USAGE)
        monkeypatch.setattr("app.chat.nodes.assess.assess_model", lambda: assess)

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


FOLLOW_UP = "What penalties does it impose?"
RESTATED = "What penalties does FuelEU Maritime impose?"
HISTORY = (ChatTurn(question="What is FuelEU Maritime?", answer="A regulation on fuel."),)


class TestRewriteInTheGraph:
    async def test_a_first_question_runs_no_rewrite_step_and_sends_the_prompts_as_before(
        self, one_result, answer_model
    ):
        state = await run_graph()

        assert [r.step for r in state.steps] == [ChatNode.RETRIEVE, ChatNode.SYNTHESIZE]
        assert state.standalone_question == ""
        [messages] = answer_model.received
        assert [type(m) for m in messages] == [SystemMessage, HumanMessage]
        assert messages[0].content == SYSTEM_PROMPT

    async def test_a_follow_up_is_restated_searched_and_answered_with_the_thread_in_view(
        self, answer_model, rewrite_turns, monkeypatch
    ):
        rewrite = rewrite_turns(restated_message(RESTATED))
        fake_search, requests = hits_for(**{RESTATED: (search_result(),)})
        monkeypatch.setattr("app.chat.nodes.retrieve.search", fake_search)

        state = ChatState(question=FOLLOW_UP, history=HISTORY)
        state.sync_from_snapshot(await chat_graph.ainvoke(state))

        assert [r.step for r in state.steps] == [
            ChatNode.REWRITE,
            ChatNode.RETRIEVE,
            ChatNode.SYNTHESIZE,
        ]
        assert state.standalone_question == RESTATED
        assert [r.query for r in requests] == [RESTATED]
        [rewrite_prompt] = rewrite.received
        assert rewrite_prompt[0].content == REWRITE_SYSTEM_PROMPT
        assert "What is FuelEU Maritime?" in rewrite_prompt[1].content
        assert rewrite_prompt[1].content.endswith(f"Latest question: {FOLLOW_UP}")
        [messages] = answer_model.received
        assert [type(m) for m in messages] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
        assert messages[0].content == SYSTEM_PROMPT + THREAD_NOTE
        assert messages[1].content == "What is FuelEU Maritime?"
        assert messages[2].content == "A regulation on fuel."
        assert messages[3].content.endswith(f"Question: {FOLLOW_UP}")

    async def test_the_rewrite_step_records_its_usage(self, rewrite_turns):
        rewrite_turns(restated_message(RESTATED))

        update = await rewrite(ChatState(question=FOLLOW_UP, history=HISTORY))

        assert update["standalone_question"] == RESTATED
        [step] = update["steps"]
        assert (step.step, step.input_tokens, step.output_tokens) == (ChatNode.REWRITE, 1500, 40)

    async def test_an_answer_off_the_schema_searches_the_question_as_asked(
        self, rewrite_turns, caplog
    ):
        rewrite_turns(AIMessage(content="It refers to FuelEU."))

        update = await rewrite(ChatState(question=FOLLOW_UP, history=HISTORY))

        assert update["standalone_question"] == ""
        assert "rewrite answered off its schema" in caplog.text

    async def test_a_failing_call_searches_the_question_as_asked(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "app.chat.nodes.rewrite.rewrite_model",
            lambda: FailingModel(messages=iter([]), failures=9),
        )

        update = await rewrite(ChatState(question=FOLLOW_UP, history=HISTORY))

        assert update["standalone_question"] == ""
        assert "rewrite call failed" in caplog.text

    async def test_with_decompose_on_the_restated_question_is_what_gets_split(
        self, decompose_on, one_result, answer_model, rewrite_turns, decompose_turns
    ):
        rewrite_turns(restated_message(RESTATED))
        decompose = decompose_turns(split_message(RESTATED))

        state = ChatState(question=FOLLOW_UP, history=HISTORY)
        state.sync_from_snapshot(await chat_graph.ainvoke(state))

        assert [r.step for r in state.steps][:3] == [
            ChatNode.REWRITE,
            ChatNode.DECOMPOSE,
            ChatNode.RETRIEVE,
        ]
        [prompt] = decompose.received
        assert prompt[1].content == RESTATED
        assert one_result == [SearchRequest(query=RESTATED, limit=config.CHAT_SOURCES)]

    async def test_assess_sees_the_thread_before_the_context(
        self, loop_on, one_result, answer_model, rewrite_turns, assess_turns
    ):
        rewrite_turns(restated_message(RESTATED))
        assess = assess_turns(AIMessage(content=""))

        state = ChatState(question=FOLLOW_UP, history=HISTORY)
        state.sync_from_snapshot(await chat_graph.ainvoke(state))

        [messages] = assess.received
        assert [type(m) for m in messages] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
        base = build_assess_system_prompt(may_refuse=config.ASSESS_MAY_REFUSE)
        assert messages[0].content == base + THREAD_NOTE
        assert messages[3].content.endswith(f"Question: {FOLLOW_UP}")

    def test_a_first_question_sends_the_base_prompt_and_a_follow_up_adds_the_thread_note(self):
        assert system_prompt(SYSTEM_PROMPT, ()) == SYSTEM_PROMPT
        assert system_prompt(SYSTEM_PROMPT, HISTORY) == SYSTEM_PROMPT + THREAD_NOTE


def test_the_compiled_graph_has_the_edges_the_readme_draws():
    """The README's diagram is hand-drawn, so the edge list it was drawn from is asserted
    here: an edge added to the graph fails this until the drawing catches up."""
    edges = {(edge.source, edge.target) for edge in chat_graph.get_graph().edges}

    assert edges == {(source, target) for source, target in GRAPH_EDGES}


class TestFollowsOfBlocksAlreadyShown:
    """A follow_reference of a paragraph the context already shows in full fetches nothing
    new, so it is dropped before the cap is counted, leaving the budget for what is missing."""

    SHOWN_ARGS = {"celex": "32023R1805", "article": "4", "paragraph": "1"}

    async def test_a_follow_of_a_block_already_shown_is_dropped_before_the_cap(
        self, loop_on, one_result, answer_model, assess_turns, tool_results, monkeypatch
    ):
        monkeypatch.setattr(config, "ASSESS_MAX_CALLS", 1)
        assess_turns(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "follow_reference",
                        "args": self.SHOWN_ARGS,
                        "id": "c1",
                        "type": "tool_call",
                    },
                    {"name": "search", "args": {"query": "a"}, "id": "c2", "type": "tool_call"},
                ],
            ),
            AIMessage(content=""),
        )
        run_calls = tool_results()

        await run_graph()

        assert run_calls == [ToolCall(name="search", args={"query": "a"})]

    async def test_a_follow_of_a_paragraph_shown_only_in_part_still_runs(
        self, loop_on, answer_model, assess_turns, tool_results, monkeypatch
    ):
        async def fake_search(session, request):
            return (search_result(part=1, parts=2),)

        monkeypatch.setattr("app.chat.nodes.retrieve.search", fake_search)
        assess_turns(tool_call_message("follow_reference", self.SHOWN_ARGS), AIMessage(content=""))
        run_calls = tool_results()

        await run_graph()

        assert run_calls == [ToolCall(name="follow_reference", args=self.SHOWN_ARGS)]

    async def test_a_follow_of_a_whole_article_still_runs_when_only_its_chapeau_is_shown(
        self, loop_on, answer_model, assess_turns, tool_results, monkeypatch
    ):
        """The chapeau's parts say nothing about the paragraphs under it, so only a
        paragraph can be known to be shown in full."""

        async def fake_search(session, request):
            return (search_result(citation="Article 4", article="4"),)

        monkeypatch.setattr("app.chat.nodes.retrieve.search", fake_search)
        whole = {"celex": "32023R1805", "article": "4"}
        assess_turns(tool_call_message("follow_reference", whole), AIMessage(content=""))
        run_calls = tool_results()

        await run_graph()

        assert run_calls == [ToolCall(name="follow_reference", args=whole)]


class TestRefuseTool:
    """Assess may answer that nothing in the context bears on the question and no fetch would
    change that: that call alone runs as a tool step and routes to the fixed refusal, with
    no answer written."""

    REFUSED = {"explanation": "no block concerns airline luggage"}

    async def test_the_call_alone_ends_in_the_fixed_refusal_without_an_answer_call(
        self, loop_on, one_result, answer_model, assess_turns
    ):
        assess_turns(tool_call_message("refuse", self.REFUSED))

        state = await run_graph()

        assert state.answer == REFUSAL_ANSWER
        assert answer_model.received == []
        assert [r.step for r in state.steps] == [
            ChatNode.RETRIEVE,
            ChatNode.ASSESS,
            ToolStep.REFUSE,
            ChatNode.REFUSE,
        ]

    async def test_the_refusal_keeps_the_context_it_was_read_against_and_the_explanation(
        self, loop_on, one_result, answer_model, assess_turns
    ):
        assess_turns(tool_call_message("refuse", self.REFUSED))

        state = await run_graph()

        assert tuple(chunk.id for chunk in state.sources) == (1,)
        assert state.refusal == Refusal(
            reason=RefusalReason.INSUFFICIENT_CONTEXT,
            explanation="no block concerns airline luggage",
        )
        assert state.steps[2].subject == "no block concerns airline luggage"

    async def test_the_call_beside_a_fetch_is_dropped_and_the_fetch_runs(
        self, loop_on, one_result, answer_model, assess_turns, tool_results
    ):
        """A hedged turn is read as a fetch: the bias is toward answering."""
        assess_turns(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "refuse",
                        "args": self.REFUSED,
                        "id": "call_1",
                        "type": "tool_call",
                    },
                    {"name": "search", "args": {"query": "a"}, "id": "call_2", "type": "tool_call"},
                ],
            ),
            AIMessage(content=""),
        )
        run_calls = tool_results(search_result(id=2))

        state = await run_graph()

        assert run_calls == [ToolCall(name="search", args={"query": "a"})]
        assert state.answer == "Answered [1]."
        assert state.refusal is None
        assert ChatNode.REFUSE not in {r.step for r in state.steps}

    async def test_the_call_without_an_explanation_still_refuses(
        self, loop_on, one_result, answer_model, assess_turns
    ):
        assess_turns(tool_call_message("refuse", {}))

        state = await run_graph()

        assert state.answer == REFUSAL_ANSWER
        assert state.refusal == Refusal(reason=RefusalReason.INSUFFICIENT_CONTEXT)

    async def test_the_refusal_still_comes_with_rounds_left_in_the_budget(
        self, loop_on, one_result, answer_model, assess_turns, monkeypatch
    ):
        """Nothing bearing on the question is final: a second round would only read the
        same context again."""
        monkeypatch.setattr(config, "ASSESS_MAX_ROUNDS", 3)
        assess_turns(tool_call_message("refuse", self.REFUSED))

        state = await run_graph()

        assert state.answer == REFUSAL_ANSWER
        assert state.assess_rounds() == 1

    def test_the_tool_is_offered_only_while_the_switch_is_on(self, monkeypatch):
        def offered() -> list[str]:
            binding = assess_model()
            assert isinstance(binding, RunnableBinding)
            return [tool["function"]["name"] for tool in binding.kwargs["tools"]]

        monkeypatch.setattr(config, "ASSESS_MAY_REFUSE", True)
        assert offered() == ["search", "follow_reference", "refuse"]

        monkeypatch.setattr(config, "ASSESS_MAY_REFUSE", False)
        assert offered() == ["search", "follow_reference"]

    async def test_the_prompt_tells_assess_when_to_call_it_while_the_switch_is_on(
        self, loop_on, one_result, answer_model, assess_turns, monkeypatch
    ):
        monkeypatch.setattr(config, "ASSESS_MAY_REFUSE", True)
        assess = assess_turns(AIMessage(content=""))

        await run_graph()

        (prompt,) = assess.received
        assert prompt[0].content == build_assess_system_prompt(may_refuse=True)
        assert "call refuse" in prompt[0].content

    async def test_the_prompt_says_nothing_of_it_while_the_switch_is_off(
        self, loop_on, one_result, answer_model, assess_turns, monkeypatch
    ):
        monkeypatch.setattr(config, "ASSESS_MAY_REFUSE", False)
        assess = assess_turns(AIMessage(content=""))

        await run_graph()

        (prompt,) = assess.received
        assert prompt[0].content == ASSESS_SYSTEM_PROMPT
        assert "call refuse" not in prompt[0].content
