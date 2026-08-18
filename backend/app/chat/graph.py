"""Two-node chat graph: retrieve corpus context, synthesize a cited answer."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.chat.enums import ChatNode
from app.chat.models import ChatState
from app.chat.prompts import SYSTEM_PROMPT, build_user_message
from app.core.config import config
from app.core.db.session import get_session
from app.core.llm import llm_retry, wrap_provider_errors
from app.retrieval.expand import expand_sections
from app.retrieval.models import RetrievedChunk, SearchRequest
from app.retrieval.search import search


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
async def synthesize(state: ChatState) -> dict[str, str]:
    """One streamed model call answering from the context with [n] citations.

    A transient provider failure is retried like embed and rerank; one that strikes
    mid-stream restarts the answer, so its tokens reach the client twice.
    """
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(build_user_message(state.question, state.sources)),
    ]
    response = await chat_model().ainvoke(messages)
    return {"answer": response.text}


def build_graph() -> CompiledStateGraph[ChatState]:
    """The compiled retrieve → synthesize graph."""
    graph = StateGraph(ChatState)
    graph.add_node(ChatNode.RETRIEVE, retrieve)
    graph.add_node(ChatNode.SYNTHESIZE, synthesize)
    graph.add_edge(START, ChatNode.RETRIEVE)
    graph.add_edge(ChatNode.RETRIEVE, ChatNode.SYNTHESIZE)
    graph.add_edge(ChatNode.SYNTHESIZE, END)
    return graph.compile()


chat_graph = build_graph()
