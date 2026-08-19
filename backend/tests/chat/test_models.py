"""Chat run state: what a graph snapshot refreshes, and what it leaves alone."""

from app.chat.enums import ChatNode
from app.chat.models import ChatNodeResult, ChatState
from app.core.exceptions import DomainError
from tests.conftest import search_result


def test_refresh_folds_the_snapshot_on_and_leaves_the_consumer_fields_alone():
    """A values snapshot carries only the graph's channels; total_ms and error, set by the
    stream's consumer, must survive it."""
    state = ChatState(question="q", total_ms=5, error="boom")
    retrieved = ChatNodeResult(node=ChatNode.RETRIEVE, ms=12)

    state.refresh({"question": "q", "nodes": (retrieved,), "sources": (), "answer": ""})

    assert state.nodes == (retrieved,)
    assert (state.total_ms, state.error) == (5, "boom")


def test_a_domain_error_is_recorded_by_its_message():
    state = ChatState(question="q")

    state.record_error(DomainError("embedding call failed"))

    assert state.error == "embedding call failed"


def test_an_unexpected_error_is_recorded_by_its_type():
    """The message of an unexpected error is not the ledger's to keep; its type names the
    failure well enough to triage, and the same name is what the stream sends."""
    state = ChatState(question="q")

    state.record_error(RuntimeError("pool exhausted"))

    assert state.error == "RuntimeError"


def test_log_fields_count_hits_and_sources_rather_than_dumping_them():
    state = ChatState(
        question="q", hits=(search_result(), search_result(id=2)), sources=(search_result(),)
    )

    fields = state.log_fields()

    assert (fields["hits"], fields["sources"]) == (2, 1)
    assert "question" not in fields
