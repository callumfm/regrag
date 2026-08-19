"""Chat run state: what a graph snapshot refreshes, and what it leaves alone."""

from app.chat.enums import ChatNode
from app.chat.models import ChatNodeResult, ChatState


def test_refresh_folds_the_snapshot_on_and_leaves_the_consumer_fields_alone():
    """A values snapshot carries only the graph's channels; total_ms and error, set by the
    stream's consumer, must survive it."""
    state = ChatState(question="q", total_ms=5, error="boom")
    retrieved = ChatNodeResult(node=ChatNode.RETRIEVE, ms=12)

    state.refresh({"question": "q", "nodes": (retrieved,), "sources": (), "answer": ""})

    assert state.nodes == (retrieved,)
    assert (state.total_ms, state.error) == (5, "boom")
