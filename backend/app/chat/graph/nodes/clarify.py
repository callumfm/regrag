"""clarify: the question back to the reader, in place of an answer, when a name in the
question could mean several companies or ships."""

from typing import Any

from app.chat.graph.node import traced
from app.chat.models import ChatState


@traced
async def clarify(state: ChatState) -> dict[str, Any]:
    """The question assess asked, as the turn's answer, so the thread's next turn reads it."""
    return {"answer": state.clarification.question if state.clarification else ""}
