"""Chat graph: split a multi-part question, retrieve corpus context, run the assess ⇄
tools loop, then synthesize a cited answer — or refuse, before any model call, a question
the corpus does not cover."""

import asyncio
import functools
import logging
import time
from collections.abc import Awaitable, Sequence
from itertools import zip_longest
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from app.chat.enums import ChatNode
from app.chat.models import ChatState, ChatStepResult, DecomposedQuestion, ToolCall
from app.chat.prompts import (
    DECOMPOSE_SYSTEM_PROMPT,
    REFUSAL_ANSWER,
    SYSTEM_PROMPT,
    build_assess_message,
    build_assess_system_prompt,
    build_user_message,
)
from app.chat.tools import (
    already_in_context,
    build_call_step,
    is_insufficient_context,
    run_tool_call,
    tool_definitions,
)
from app.core.clock import elapsed_ms
from app.core.config import config
from app.core.db.session import get_session
from app.core.llm import LLMError, llm_retry, wrap_provider_errors
from app.retrieval.expand import expand_sections
from app.retrieval.models import RetrievedChunk, SearchRequest, SearchResult
from app.retrieval.search import search
from app.retrieval.thresholds import meets_thresholds

logger = logging.getLogger(__name__)


class NodeFn(Protocol):
    """A node: the state so far in, the fields it sets out — plus `usage`, if it called a
    model, which its step carries rather than the state. Named for the ChatNode it is."""

    __name__: str

    def __call__(self, state: ChatState) -> Awaitable[dict[str, Any]]: ...


def traced(run: NodeFn) -> NodeFn:
    """The node as the graph runs it, appending one step — how long it took, and the usage
    it reported — to the path. Outermost on a node, so a retried call is traced as a whole."""
    node = ChatNode(run.__name__)

    @functools.wraps(run)
    async def traced_run(state: ChatState) -> dict[str, Any]:
        start = time.perf_counter()
        update = await run(state)
        usage = update.pop("usage", None)
        step = ChatStepResult.from_usage(node, elapsed_ms(start), usage)
        return update | {"steps": (step,)}

    return traced_run


def chat_model(model: str, *, streaming: bool = True) -> ChatLiteLLM:
    """A chat client built per call, so config is read at call time like embed's. The model
    is the caller's, so assess and the answer can be pointed at different ones.

    Streaming is set for the answer, or litellm answers in one blocking call — even under
    the graph's messages stream — and the SSE stream carries the whole answer in a single
    token event; assess wants that one blocking call, so it turns streaming off.
    Usage is asked for, or litellm strips it from every streamed chunk and the run's
    tokens are never reported for a non-OpenAI model; it is read on the streamed path only.
    """
    return ChatLiteLLM(
        model=model,
        api_key=config.ANTHROPIC_API_KEY.get_secret_value(),
        max_tokens=config.CHAT_MAX_TOKENS,
        temperature=config.CHAT_TEMPERATURE,
        request_timeout=config.CHAT_TIMEOUT,
        streaming=streaming,
        stream_options={"include_usage": True},
    )


async def search_query(query: str) -> tuple[SearchResult, ...]:
    """One query's hits from its own session, so the queries a question split into can
    search at once rather than in turn."""
    async with get_session(auto_commit=False) as session:
        return await search(session, SearchRequest(query=query, limit=config.CHAT_SOURCES))


def interleave_by_rank(
    per_query: Sequence[Sequence[SearchResult]],
) -> tuple[SearchResult, ...]:
    """Every query's hits as one list, each query's first before any query's second, so no
    part's best hit is pushed out by another part's depth; a chunk two queries both found
    is kept once, at its earliest place."""
    merged: list[SearchResult] = []
    seen: set[int] = set()
    for rank in zip_longest(*per_query):
        for hit in rank:
            if hit is None or hit.id in seen:
                continue
            seen.add(hit.id)
            merged.append(hit)
    return tuple(merged)


@traced
async def retrieve(state: ChatState) -> dict[str, Any]:
    """The corpus's best answers to each query the question split into — or to the question
    as asked — gated query by query, so an out-of-corpus part admits nothing, and widened to
    their sections. hits keeps every query's hits, gated or not, so a refusal and a split
    can be read against what search found."""
    queries = state.queries or (state.question,)
    per_query = await asyncio.gather(*(search_query(query) for query in queries))
    hits = interleave_by_rank(per_query)
    cleared = [found for found in per_query if meets_thresholds(found)]
    if not cleared:
        return {"hits": hits, "sources": (), "retrieved_sources": 0}
    sources: tuple[RetrievedChunk, ...] = interleave_by_rank(cleared)
    if config.EXPAND_SECTIONS:
        async with get_session(auto_commit=False) as session:
            sources = await expand_sections(session, sources, limit=config.CHAT_CONTEXT_CHUNKS)
    return {"hits": hits, "sources": sources, "retrieved_sources": len(sources)}


@traced
@llm_retry
@wrap_provider_errors("chat call")
async def synthesize(state: ChatState) -> dict[str, Any]:
    """One streamed model call answering from the context with [n] citations.

    A transient provider failure is retried like embed and rerank; one that strikes
    mid-stream restarts the answer, so its tokens reach the client twice.
    """
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(build_user_message(state.question, state.sources)),
    ]
    response = await chat_model(config.CHAT_MODEL).ainvoke(messages)
    return {"answer": response.text, "usage": response.usage_metadata}


@traced
async def refuse(state: ChatState) -> dict[str, Any]:
    """The fixed refusal, in place of an answer, for a question without context."""
    return {"answer": REFUSAL_ANSWER}


def decompose_model() -> Runnable:
    """The decompose model as decompose calls it: one blocking turn, answering in the
    DecomposedQuestion shape."""
    return chat_model(config.DECOMPOSE_MODEL, streaming=False).bind(
        response_format=DecomposedQuestion
    )


@llm_retry
@wrap_provider_errors("decompose call")
async def call_decompose_model(state: ChatState) -> dict[str, Any]:
    """One model turn splitting the question into the searches it needs, capped to the
    parts allowed. One part means the question asked one thing, and queries stays empty
    so retrieve searches the question as asked; an answer off the schema is a failed call."""
    messages = [SystemMessage(DECOMPOSE_SYSTEM_PROMPT), HumanMessage(state.question)]
    response = await decompose_model().ainvoke(messages)
    try:
        split = DecomposedQuestion.model_validate_json(response.text)
    except ValidationError as exc:
        logger.warning("decompose answered off its schema: %s", exc)
        raise LLMError("decompose answered off its schema") from exc
    queries = split.queries[: config.DECOMPOSE_MAX_PARTS]
    return {"queries": queries if len(queries) > 1 else (), "usage": response.usage_metadata}


@traced
async def decompose(state: ChatState) -> dict[str, Any]:
    """The question split into its parts, or left whole when it has one — or when the
    call fails, which costs the split rather than the request."""
    try:
        return await call_decompose_model(state)
    except LLMError as exc:
        logger.warning("decompose call failed, searching the question as asked: %s", exc)
        return {"queries": ()}


def assess_model() -> Runnable:
    """The assess model as assess calls it: one blocking turn, the tool surface bound."""
    return chat_model(config.ASSESS_MODEL, streaming=False).bind_tools(tool_definitions())


@llm_retry
@wrap_provider_errors("assess call")
async def call_assess_model(state: ChatState) -> dict[str, Any]:
    """One model turn asking what would fill the gaps in the context — or, called alone,
    saying nothing bears on the question. That call beside a fetch is dropped, the fetch
    being the model's own doubt; a fetch that would only re-fetch a division the context
    already shows is dropped too, then the rest are capped to the calls a round may run —
    none of the dropped reaches state or the ledger."""
    messages = [
        SystemMessage(build_assess_system_prompt(may_refuse=config.ASSESS_MAY_REFUSE)),
        HumanMessage(build_assess_message(state.question, state.sources)),
    ]
    response = await assess_model().ainvoke(messages)
    asked = [ToolCall(name=c["name"], args=c["args"]) for c in response.tool_calls]
    insufficient = [call for call in asked if is_insufficient_context(call)]
    fetches = [call for call in asked if not is_insufficient_context(call)]
    if insufficient and not fetches:
        return {"pending_calls": (insufficient[0],), "usage": response.usage_metadata}
    if insufficient:
        logger.info("assess hedged insufficient context with a fetch, so the fetch runs")
    useful = [call for call in fetches if not already_in_context(call, state.sources)]
    calls = tuple(useful[: config.ASSESS_MAX_CALLS])
    return {"pending_calls": calls, "usage": response.usage_metadata}


@traced
async def assess(state: ChatState) -> dict[str, Any]:
    """One review of the context: the calls that would fill what is missing, or none when
    it suffices. A failing call settles for the context so far rather than failing the run."""
    try:
        return await call_assess_model(state)
    except LLMError as exc:
        logger.warning("assess call failed, settling for the context gathered so far: %s", exc)
        return {"pending_calls": ()}


def merge_sources(
    sources: tuple[RetrievedChunk, ...], additions: Sequence[RetrievedChunk], *, cap: int
) -> tuple[RetrievedChunk, ...]:
    """The context grown by a tool round: new chunks appended in arrival order, a chunk
    already present kept as it was, and nothing appended once the cap is reached. The cap
    counts the whole context, so it is read against what retrieve produced, not this round."""
    merged = list(sources)
    seen = {chunk.id for chunk in merged}
    for chunk in additions:
        if len(merged) >= cap:
            break
        if chunk.id in seen:
            continue
        seen.add(chunk.id)
        merged.append(chunk)
    return tuple(merged)


async def tools(state: ChatState) -> dict[str, Any]:
    """The round's calls run and folded into the context: dedup by chunk id, earlier context
    kept, growth capped. Each call is timed as its own step, so the path says what it cost.
    An insufficient_context call fetches nothing and leaves its reason on the state, which
    is what routes the round to the refusal."""
    fetched: list[RetrievedChunk] = []
    steps: list[ChatStepResult] = []
    insufficiency = state.insufficiency
    for call in state.pending_calls:
        start = time.perf_counter()
        fetched.extend(await run_tool_call(call))
        steps.append(build_call_step(call, ms=elapsed_ms(start)))
        if is_insufficient_context(call):
            insufficiency = str(call.args.get("reason", ""))
            logger.info("assess found the context insufficient: %s", insufficiency)

    cap = state.retrieved_sources + config.ASSESS_EXTRA_CHUNKS
    return {
        "sources": merge_sources(state.sources, fetched, cap=cap),
        "pending_calls": (),
        "insufficiency": insufficiency,
        "steps": tuple(steps),
    }


def assess_or_synthesize(state: ChatState) -> ChatNode:
    """Review again while budget remains, else answer with what there is."""
    return ChatNode.SYNTHESIZE if state.context_settled else ChatNode.ASSESS


def tools_or_synthesize(state: ChatState) -> ChatNode:
    """After assess: run what it asked for, or answer when it asked for nothing."""
    return ChatNode.SYNTHESIZE if state.context_settled else ChatNode.TOOLS


def assess_or_synthesize_or_refuse(state: ChatState) -> ChatNode:
    """After retrieve or tools: refuse for want of context — none cleared the gate, or
    assess found what there is bears on nothing — else review or answer."""
    if not state.sources or state.insufficiency is not None:
        return ChatNode.REFUSE
    return assess_or_synthesize(state)


def decompose_or_retrieve(state: ChatState) -> ChatNode:
    """At the start: split the question when the node is on, else search it as asked. An
    edge rather than a check inside the node, so a run with it off records no step."""
    return ChatNode.DECOMPOSE if config.DECOMPOSE_ENABLED else ChatNode.RETRIEVE


GRAPH_EDGES = (
    (START, ChatNode.DECOMPOSE),
    (START, ChatNode.RETRIEVE),
    (ChatNode.DECOMPOSE, ChatNode.RETRIEVE),
    (ChatNode.RETRIEVE, ChatNode.ASSESS),
    (ChatNode.RETRIEVE, ChatNode.SYNTHESIZE),
    (ChatNode.RETRIEVE, ChatNode.REFUSE),
    (ChatNode.ASSESS, ChatNode.TOOLS),
    (ChatNode.ASSESS, ChatNode.SYNTHESIZE),
    (ChatNode.TOOLS, ChatNode.ASSESS),
    (ChatNode.TOOLS, ChatNode.SYNTHESIZE),
    (ChatNode.TOOLS, ChatNode.REFUSE),
    (ChatNode.SYNTHESIZE, END),
    (ChatNode.REFUSE, END),
)
"""Every edge the graph has, as the README draws them. A test holds the compiled graph to
this, so an edge added here without redrawing the README fails before it is merged."""


def build_graph() -> CompiledStateGraph[ChatState]:
    """The compiled (decompose →) retrieve → (assess ⇄ tools) → (synthesize | refuse) graph."""
    graph = StateGraph(ChatState)
    graph.add_node(ChatNode.DECOMPOSE, decompose)
    graph.add_node(ChatNode.RETRIEVE, retrieve)
    graph.add_node(ChatNode.ASSESS, assess)
    graph.add_node(ChatNode.TOOLS, tools)
    graph.add_node(ChatNode.SYNTHESIZE, synthesize)
    graph.add_node(ChatNode.REFUSE, refuse)
    graph.add_conditional_edges(
        START, decompose_or_retrieve, [ChatNode.DECOMPOSE, ChatNode.RETRIEVE]
    )
    graph.add_edge(ChatNode.DECOMPOSE, ChatNode.RETRIEVE)
    graph.add_conditional_edges(
        ChatNode.RETRIEVE,
        assess_or_synthesize_or_refuse,
        [ChatNode.ASSESS, ChatNode.SYNTHESIZE, ChatNode.REFUSE],
    )
    graph.add_conditional_edges(
        ChatNode.ASSESS, tools_or_synthesize, [ChatNode.TOOLS, ChatNode.SYNTHESIZE]
    )
    graph.add_conditional_edges(
        ChatNode.TOOLS,
        assess_or_synthesize_or_refuse,
        [ChatNode.ASSESS, ChatNode.SYNTHESIZE, ChatNode.REFUSE],
    )
    graph.add_edge(ChatNode.SYNTHESIZE, END)
    graph.add_edge(ChatNode.REFUSE, END)
    return graph.compile()


chat_graph = build_graph()
