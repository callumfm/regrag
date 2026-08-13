"""SSE chat endpoint streaming a cited answer from the chat graph."""

import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter
from langchain_core.runnables import RunnableConfig
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from app.chat.graph import chat_graph
from app.chat.models import ChatRequest, ChatSource
from app.core.db.session import SessionDep
from app.core.exceptions import DomainError
from app.retrieval.models import SearchResult

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _sources_event(sources: tuple[SearchResult, ...]) -> ServerSentEvent:
    """The sources event binding [n] markers to the retrieved chunks."""
    payload = [
        ChatSource.from_result(marker, result).model_dump()
        for marker, result in enumerate(sources, start=1)
    ]
    return ServerSentEvent(event="sources", data=json.dumps(payload))


async def _chat_events(
    question: str, session: AsyncSession
) -> AsyncGenerator[ServerSentEvent, None]:
    """Sources once, then tokens, then done; an error event ends a failed stream.

    astream's list-mode yields are untyped by langgraph, hence the ty ignores below.
    """
    state = {"question": question, "sources": (), "answer": ""}
    run_config: RunnableConfig = {"configurable": {"session": session}}
    try:
        async for mode, data in chat_graph.astream(
            state, config=run_config, stream_mode=["updates", "messages"]
        ):
            if mode == "updates" and "retrieve" in data:
                sources = data["retrieve"]["sources"]  # ty: ignore[invalid-argument-type]
                yield _sources_event(sources)
            elif mode == "messages":
                chunk, metadata = data
                text = chunk.content  # ty: ignore[unresolved-attribute]
                node = metadata.get("langgraph_node")  # ty: ignore[unresolved-attribute]
                if isinstance(text, str) and text and node == "synthesize":
                    yield ServerSentEvent(event="token", data=json.dumps({"text": text}))
    except DomainError as exc:
        logger.warning("chat stream failed: %s", exc.message)
        yield ServerSentEvent(event="error", data=json.dumps({"message": exc.message}))
        return
    except Exception:
        logger.exception("chat stream failed unexpectedly")
        yield ServerSentEvent(
            event="error", data=json.dumps({"message": "An unexpected error occurred"})
        )
        return
    yield ServerSentEvent(event="done", data="{}")


@router.post("/chat")
async def chat(request: ChatRequest, db: SessionDep) -> EventSourceResponse:
    """Stream a cited answer to the question over SSE."""
    return EventSourceResponse(_chat_events(request.question, db))
