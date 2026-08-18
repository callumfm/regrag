"""Chat stream orchestration: the graph run, translated into SSE events and measured."""

from collections.abc import AsyncGenerator

from fastapi.sse import ServerSentEvent

from app.chat.events import error_event, event_for, sse_event
from app.chat.graph import ChatState, chat_graph
from app.chat.observability.enums import ChatOutcome
from app.chat.observability.service import record_run
from app.chat.observability.stats import StreamStats


async def chat_events(question: str) -> AsyncGenerator[ServerSentEvent, None]:
    """Sources once, then tokens, then done; an error event ends a failed stream.
    Whichever way the stream ends — done, error, or the client leaving — one chat
    run is recorded for it."""
    stats = StreamStats()
    outcome = ChatOutcome.ABORTED
    try:
        async for mode, data in chat_graph.astream(
            ChatState(question=question), stream_mode=["updates", "messages"]
        ):
            stats.observe(mode, data)
            if event := event_for(mode, data):
                yield event
        yield sse_event("done", {})
        outcome = ChatOutcome.DONE
    except Exception as exc:
        outcome = ChatOutcome.ERROR
        yield error_event(exc)
    finally:
        await record_run(stats, outcome)
