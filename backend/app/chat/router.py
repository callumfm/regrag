"""SSE chat endpoint streaming a cited answer from the chat graph."""

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.chat.models import ChatRequest
from app.chat.service import chat_events

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat(request: ChatRequest) -> EventSourceResponse:
    """Stream a cited answer to the question over SSE."""
    return EventSourceResponse(chat_events(request.question))
