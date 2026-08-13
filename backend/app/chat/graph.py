"""Two-node chat graph: retrieve corpus context, synthesize a cited answer."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_litellm import ChatLiteLLM
from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession
from typing_extensions import TypedDict

from app.chat.prompts import SYSTEM_PROMPT, build_user_message
from app.core.config import config
from app.core.llm import TRANSIENT_PROVIDER_ERRORS, LLMError, ProviderError
from app.retrieval.models import SearchResult
from app.retrieval.pipeline import search

logger = logging.getLogger(__name__)


class ChatState(TypedDict):
    """What flows through the graph for one question."""

    question: str
    sources: tuple[SearchResult, ...]
    answer: str


chat_model = ChatLiteLLM(
    model=config.CHAT_MODEL,
    api_key=config.ANTHROPIC_API_KEY,
    max_tokens=config.CHAT_MAX_TOKENS,
    request_timeout=config.CHAT_TIMEOUT,
)


def _session() -> AsyncSession:
    """The request's DB session, read from LangGraph's runtime config."""
    return get_config()["configurable"]["session"]


async def retrieve(state: ChatState) -> dict:
    """The corpus's best answers to the question, via the full search pipeline."""
    results = await search(_session(), state["question"], limit=config.CHAT_CONTEXT_CHUNKS)
    return {"sources": results}


async def synthesize(state: ChatState) -> dict:
    """One streamed model call answering from the context with [n] citations."""
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(build_user_message(state["question"], state["sources"])),
    ]
    try:
        response = await chat_model.ainvoke(messages, get_config())
    except ProviderError as exc:
        logger.warning("chat call failed: %s", exc)
        raise LLMError(
            "chat call failed", transient=isinstance(exc, TRANSIENT_PROVIDER_ERRORS)
        ) from exc
    return {"answer": str(response.content)}


def build_graph() -> CompiledStateGraph:
    """The compiled retrieve → synthesize graph: ty doesn't match a TypedDict
    against langgraph's StateLike protocol, hence the two suppressions below."""
    graph = StateGraph(ChatState)  # ty: ignore[invalid-argument-type]
    graph.add_node("retrieve", retrieve)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()  # ty: ignore[invalid-return-type]


chat_graph = build_graph()
