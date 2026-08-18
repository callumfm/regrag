"""Chat graph: retrieve corpus context, then synthesize a cited answer — or refuse,
before any model call, a question the corpus does not cover."""

from collections.abc import Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.chat.enums import ChatNode
from app.chat.models import ChatState
from app.chat.prompts import REFUSAL_ANSWER, SYSTEM_PROMPT, build_user_message
from app.core.config import config
from app.core.db.session import get_session
from app.core.llm import llm_retry, wrap_provider_errors
from app.retrieval.expand import expand_sections
from app.retrieval.models import RetrievedChunk, SearchRequest, SearchResult
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


def _clears(values: Sequence[float | None], bar: float) -> bool:
    """Whether the best present value clears the bar; a signal nothing carries is not judged."""
    present = [value for value in values if value is not None]
    return not present or max(present) >= bar


def covers(hits: Sequence[SearchResult]) -> bool:
    """Whether the corpus has something for this question: the nearest hit by cosine and
    the cross-encoder's best relevance each clear their bar. Both must, because each is
    fooled where the other is not — cosine by shared vocabulary, the reranker now and
    then by a generic chunk. Nothing found is nothing covered."""
    return bool(hits) and (
        _clears([hit.cosine_similarity for hit in hits], config.CHAT_MIN_COSINE_SIMILARITY)
        and _clears([hit.reranker_relevance for hit in hits], config.CHAT_MIN_RERANKER_RELEVANCE)
    )


async def retrieve(state: ChatState) -> dict[str, tuple[RetrievedChunk, ...]]:
    """The corpus's best answers, widened to their sections, from a node-scoped session
    so no connection is held while the model streams. Nothing, when the corpus does not
    cover the question: that empties the context, and the graph refuses instead."""
    async with get_session(auto_commit=False) as session:
        hits = await search(session, SearchRequest(query=state.question, limit=config.CHAT_SOURCES))
        if not covers(hits):
            return {"sources": ()}
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


def refuse(state: ChatState) -> dict[str, str]:
    """The fixed refusal, in place of an answer, for a question without context."""
    return {"answer": REFUSAL_ANSWER}


def after_retrieve(state: ChatState) -> ChatNode:
    """Synthesize from context, or refuse for want of it — the cheap path, ahead of the
    first model call, that later stages of the graph slot in behind."""
    return ChatNode.SYNTHESIZE if state.sources else ChatNode.REFUSE


def build_graph() -> CompiledStateGraph[ChatState]:
    """The compiled retrieve → (synthesize | refuse) graph."""
    graph = StateGraph(ChatState)
    graph.add_node(ChatNode.RETRIEVE, retrieve)
    graph.add_node(ChatNode.SYNTHESIZE, synthesize)
    graph.add_node(ChatNode.REFUSE, refuse)
    graph.add_edge(START, ChatNode.RETRIEVE)
    graph.add_conditional_edges(
        ChatNode.RETRIEVE, after_retrieve, [ChatNode.SYNTHESIZE, ChatNode.REFUSE]
    )
    graph.add_edge(ChatNode.SYNTHESIZE, END)
    graph.add_edge(ChatNode.REFUSE, END)
    return graph.compile()


chat_graph = build_graph()
