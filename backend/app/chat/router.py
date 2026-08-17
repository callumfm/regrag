"""SSE chat endpoint streaming a cited answer from the chat graph."""

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.chat.models import ChatEvent, ChatRequest
from app.chat.service import chat_events

router = APIRouter(tags=["chat"])

CHAT_RESPONSES: dict[int | str, dict[str, Any]] = {200: {"model": ChatEvent}}
"""The frames carry names, so their payloads are documented here rather than read from
the yield type; the response class files them under text/event-stream."""


@router.post("/chat", response_class=EventSourceResponse, responses=CHAT_RESPONSES)
async def chat(request: ChatRequest) -> AsyncIterator[ServerSentEvent]:
    """Stream a cited answer to the question over SSE: sources, tokens, done; or error."""
    async for event in chat_events(request.question):
        yield event
