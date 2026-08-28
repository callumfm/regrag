"""Chat graph: retrieve corpus context, run the assess ⇄ tools loop, then synthesize a
cited answer — or refuse, before any model call, a question the corpus does not cover."""

import functools
import logging
import time
from collections.abc import Awaitable, Sequence
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.chat.enums import ChatNode
from app.chat.models import ChatState, ChatStepResult, ToolCall
from app.chat.prompts import (
    ASSESS_SYSTEM_PROMPT,
    REFUSAL_ANSWER,
    SYSTEM_PROMPT,
    build_assess_message,
    build_user_message,
)
from app.chat.tools import TOOL_DEFINITIONS, run_tool_call, tool_step
from app.core.clock import elapsed_ms
from app.core.config import config
from app.core.db.session import get_session
from app.core.llm import LLMError, llm_retry, wrap_provider_errors
from app.retrieval.expand import expand_sections
from app.retrieval.models import RetrievedChunk, SearchRequest
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


def chat_model(*, streaming: bool = True) -> ChatLiteLLM:
    """A chat client built per call, so config is read at call time like embed's.

    Streaming is set for the answer, or litellm answers in one blocking call — even under
    the graph's messages stream — and the SSE stream carries the whole answer in a single
    token event; assess wants that one blocking call, so it turns streaming off.
    Usage is asked for, or litellm strips it from every streamed chunk and the run's
    tokens are never reported for a non-OpenAI model; it is read on the streamed path only.
    """
    return ChatLiteLLM(
        model=config.CHAT_MODEL,
        api_key=config.ANTHROPIC_API_KEY.get_secret_value(),
        max_tokens=config.CHAT_MAX_TOKENS,
        request_timeout=config.CHAT_TIMEOUT,
        streaming=streaming,
        stream_options={"include_usage": True},
    )


@traced
async def retrieve(state: ChatState) -> dict[str, Any]:
    """The corpus's best answers, widened to their sections, from a node-scoped session
    so no connection is held while the model streams. Nothing, when the corpus does not
    cover the question: that empties the context, and the graph refuses instead — but what
    search found stays on the state, so the refusal can be read against it."""
    async with get_session(auto_commit=False) as session:
        hits = await search(session, SearchRequest(query=state.question, limit=config.CHAT_SOURCES))
        if not meets_thresholds(hits):
            return {"hits": hits, "sources": (), "retrieved_sources": 0}
        sources: tuple[RetrievedChunk, ...] = hits
        if config.EXPAND_SECTIONS:
            sources = await expand_sections(session, hits, limit=config.CHAT_CONTEXT_CHUNKS)
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
    response = await chat_model().ainvoke(messages)
    return {"answer": response.text, "usage": response.usage_metadata}


@traced
async def refuse(state: ChatState) -> dict[str, Any]:
    """The fixed refusal, in place of an answer, for a question without context."""
    return {"answer": REFUSAL_ANSWER}


def assess_model() -> Runnable:
    """The chat model as assess calls it: one blocking turn, the tool surface bound."""
    return chat_model(streaming=False).bind_tools(TOOL_DEFINITIONS)


@llm_retry
@wrap_provider_errors("assess call")
async def call_assess_model(state: ChatState) -> dict[str, Any]:
    """One model turn asking what would fill the gaps in the context, capped to the
    calls a round may run — the rest dropped before state or the ledger sees them."""
    messages = [
        SystemMessage(ASSESS_SYSTEM_PROMPT),
        HumanMessage(build_assess_message(state.question, state.sources)),
    ]
    response = await assess_model().ainvoke(messages)
    calls = tuple(
        ToolCall(name=c["name"], args=c["args"])
        for c in response.tool_calls[: config.ASSESS_MAX_CALLS]
    )
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
    kept, growth capped. Each call is timed as its own step, so the path says what it cost."""
    fetched: list[RetrievedChunk] = []
    steps: list[ChatStepResult] = []
    for call in state.pending_calls:
        start = time.perf_counter()
        fetched.extend(await run_tool_call(call))
        steps.append(ChatStepResult(step=tool_step(call.name), ms=elapsed_ms(start)))

    cap = state.retrieved_sources + config.ASSESS_EXTRA_CHUNKS
    return {
        "sources": merge_sources(state.sources, fetched, cap=cap),
        "pending_calls": (),
        "steps": tuple(steps),
    }


def assess_or_synthesize(state: ChatState) -> ChatNode:
    """After tools: review again while budget remains, else answer with what there is."""
    return ChatNode.SYNTHESIZE if state.context_settled else ChatNode.ASSESS


def tools_or_synthesize(state: ChatState) -> ChatNode:
    """After assess: run what it asked for, or answer when it asked for nothing."""
    return ChatNode.SYNTHESIZE if state.context_settled else ChatNode.TOOLS


def assess_or_synthesize_or_refuse(state: ChatState) -> ChatNode:
    """After retrieve: refuse for want of context, else pick up the loop as tools does."""
    return ChatNode.REFUSE if not state.sources else assess_or_synthesize(state)


GRAPH_EDGES = (
    (START, ChatNode.RETRIEVE),
    (ChatNode.RETRIEVE, ChatNode.ASSESS),
    (ChatNode.RETRIEVE, ChatNode.SYNTHESIZE),
    (ChatNode.RETRIEVE, ChatNode.REFUSE),
    (ChatNode.ASSESS, ChatNode.TOOLS),
    (ChatNode.ASSESS, ChatNode.SYNTHESIZE),
    (ChatNode.TOOLS, ChatNode.ASSESS),
    (ChatNode.TOOLS, ChatNode.SYNTHESIZE),
    (ChatNode.SYNTHESIZE, END),
    (ChatNode.REFUSE, END),
)
"""Every edge the graph has, as the README draws them. A test holds the compiled graph to
this, so an edge added here without redrawing the README fails before it is merged."""


def build_graph() -> CompiledStateGraph[ChatState]:
    """The compiled retrieve → (assess ⇄ tools) → (synthesize | refuse) graph."""
    graph = StateGraph(ChatState)
    graph.add_node(ChatNode.RETRIEVE, retrieve)
    graph.add_node(ChatNode.ASSESS, assess)
    graph.add_node(ChatNode.TOOLS, tools)
    graph.add_node(ChatNode.SYNTHESIZE, synthesize)
    graph.add_node(ChatNode.REFUSE, refuse)
    graph.add_edge(START, ChatNode.RETRIEVE)
    graph.add_conditional_edges(
        ChatNode.RETRIEVE,
        assess_or_synthesize_or_refuse,
        [ChatNode.ASSESS, ChatNode.SYNTHESIZE, ChatNode.REFUSE],
    )
    graph.add_conditional_edges(
        ChatNode.ASSESS, tools_or_synthesize, [ChatNode.TOOLS, ChatNode.SYNTHESIZE]
    )
    graph.add_conditional_edges(
        ChatNode.TOOLS, assess_or_synthesize, [ChatNode.ASSESS, ChatNode.SYNTHESIZE]
    )
    graph.add_edge(ChatNode.SYNTHESIZE, END)
    graph.add_edge(ChatNode.REFUSE, END)
    return graph.compile()


chat_graph = build_graph()
