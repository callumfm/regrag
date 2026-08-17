"""SSE chat endpoint streaming a cited answer from the chat graph."""

from typing import Any

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.chat.models import ChatEvent, ChatRequest
from app.chat.service import chat_events

router = APIRouter(tags=["chat"])

SSE_RESPONSE: dict[int | str, dict[str, Any]] = {
    200: {"model": ChatEvent, "content": {"text/event-stream": {}}}
}
"""Returning a Response leaves FastAPI nothing to document, so name the frames here."""


@router.post("/chat", responses=SSE_RESPONSE)
async def chat(request: ChatRequest) -> EventSourceResponse:
    """Stream a cited answer to the question over SSE."""
    return EventSourceResponse(chat_events(request.question))
