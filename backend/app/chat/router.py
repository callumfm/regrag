"""SSE chat endpoint streaming a cited answer from the chat graph."""

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.chat.models import ChatEvent, ChatQuery
from app.chat.service import chat_events

router = APIRouter(tags=["chat"])

CHAT_RESPONSES: dict[int | str, dict[str, Any]] = {200: {"model": ChatEvent}}
"""What a frame's data holds, for the OpenAPI schema: FastAPI cannot read it from the
yielded ServerSentEvent, and the response class files it under text/event-stream."""


@router.post("/chat", response_class=EventSourceResponse, responses=CHAT_RESPONSES)
async def chat(query: ChatQuery) -> AsyncIterator[ServerSentEvent]:
    """Stream a cited answer to the question over SSE: sources, tokens, done; or error."""
    async for event in chat_events(query.question):
        yield ServerSentEvent(event=event.event, data=event.data)
