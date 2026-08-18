"""Two-node chat graph: retrieve corpus context, synthesize a cited answer."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.chat.prompts import SYSTEM_PROMPT, build_user_message
from app.core.config import config
from app.core.db.session import get_session
from app.core.llm import llm_retry, wrap_provider_errors
from app.core.models import AppModel
from app.retrieval.expand import expand_sections
from app.retrieval.models import RetrievedChunk, SearchRequest
from app.retrieval.search import search


class ChatState(AppModel):
    """What flows through the graph for one question."""

    question: str
    sources: tuple[RetrievedChunk, ...] = ()
    answer: str = ""
    usage: UsageMetadata | None = None


def chat_model() -> ChatLiteLLM:
    """A chat client built per call, so config is read at call time like embed's.

    Streaming is set, or litellm answers in one blocking call — even under the graph's
    messages stream — and the SSE stream carries the whole answer in a single token event.
    """
    return ChatLiteLLM(
        model=config.CHAT_MODEL,
        api_key=config.ANTHROPIC_API_KEY,
        max_tokens=config.CHAT_MAX_TOKENS,
        request_timeout=config.CHAT_TIMEOUT,
        streaming=True,
    )


async def retrieve(state: ChatState) -> dict[str, tuple[RetrievedChunk, ...]]:
    """The corpus's best answers, widened to their sections, from a node-scoped session
    so no connection is held while the model streams."""
    async with get_session(auto_commit=False) as session:
        hits = await search(session, SearchRequest(query=state.question, limit=config.CHAT_SOURCES))
        sources: tuple[RetrievedChunk, ...] = hits
        if config.EXPAND_SECTIONS:
            sources = await expand_sections(session, hits, limit=config.CHAT_CONTEXT_CHUNKS)
    return {"sources": sources}


@llm_retry
@wrap_provider_errors("chat call")
async def synthesize(state: ChatState) -> dict[str, Any]:
    """One streamed model call answering from the context with [n] citations, and
    the tokens it cost.

    A transient provider failure is retried like embed and rerank; one that strikes
    mid-stream restarts the answer, so its tokens reach the client twice.
    """
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(build_user_message(state.question, state.sources)),
    ]
    response = await chat_model().ainvoke(messages)
    return {"answer": response.text, "usage": response.usage_metadata}


def build_graph() -> CompiledStateGraph[ChatState]:
    """The compiled retrieve → synthesize graph."""
    graph = StateGraph(ChatState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


chat_graph = build_graph()
