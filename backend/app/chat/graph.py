"""Two-node chat graph: retrieve corpus context, synthesize a cited answer."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from app.chat.prompts import SYSTEM_PROMPT, build_user_message
from app.core.config import config
from app.core.db.session import get_session
from app.core.llm import wrap_provider_errors
from app.retrieval.models import SearchRequest, SearchResult
from app.retrieval.search import search


class ChatState(TypedDict):
    """What flows through the graph for one question."""

    question: str
    sources: tuple[SearchResult, ...]
    answer: str


def chat_model() -> ChatLiteLLM:
    """A chat client built per call, so config is read at call time like embed's."""
    return ChatLiteLLM(
        model=config.CHAT_MODEL,
        api_key=config.ANTHROPIC_API_KEY,
        max_tokens=config.CHAT_MAX_TOKENS,
        request_timeout=config.CHAT_TIMEOUT,
    )


async def retrieve(state: ChatState) -> dict:
    """The corpus's best answers, from a node-scoped session so no connection is
    held while the model streams."""
    async with get_session() as session:
        request = SearchRequest(query=state["question"], limit=config.CHAT_CONTEXT_CHUNKS)
        results = await search(session, request)
    return {"sources": results}


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
