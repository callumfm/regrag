"""Two-node chat graph: retrieve corpus context, synthesize a cited answer."""

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.chat.prompts import SYSTEM_PROMPT, build_user_message
from app.core.config import config
from app.core.db.session import get_session
from app.core.llm import wrap_provider_errors
from app.retrieval.expand import expand_sections
from app.retrieval.models import RetrievedChunk, SearchRequest
from app.retrieval.search import search


class ChatState(TypedDict):
    """What flows through the graph for one question."""

    question: str
    sources: tuple[RetrievedChunk, ...]
    answer: str


def chat_model() -> ChatLiteLLM:
    """A chat client built per call, so config is read at call time like embed's.

    streaming, or litellm answers in one blocking call and the SSE stream carries
    the whole answer in a single token event.
    """
    return ChatLiteLLM(
        model=config.CHAT_MODEL,
        api_key=config.ANTHROPIC_API_KEY,
        max_tokens=config.CHAT_MAX_TOKENS,
        request_timeout=config.CHAT_TIMEOUT,
        streaming=True,
    )


async def retrieve(state: ChatState) -> dict:
    """The corpus's best answers, widened to whole articles, from a node-scoped session
    so no connection is held while the model streams."""
    async with get_session() as session:
        results = await search(session, SearchRequest(query=state["question"]))
        sources: tuple[RetrievedChunk, ...] = results
        if config.EXPAND_SECTIONS:
            sources = await expand_sections(session, results)
    return {"sources": sources}


@wrap_provider_errors
async def synthesize(state: ChatState) -> dict:
    """One streamed model call answering from the context with [n] citations."""
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(build_user_message(state["question"], state["sources"])),
    ]
    response = await chat_model().ainvoke(messages)
    return {"answer": str(response.content)}


def build_graph() -> CompiledStateGraph:
    """The compiled retrieve → synthesize graph."""
    graph = StateGraph(ChatState)  # ty: ignore[invalid-argument-type]
    graph.add_node("retrieve", retrieve)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()  # ty: ignore[invalid-return-type]


chat_graph = build_graph()
