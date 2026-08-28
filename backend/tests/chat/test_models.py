"""Chat run state: what a graph snapshot refreshes, and what it leaves alone."""

from app.chat.enums import ChatNode
from app.chat.models import ChatNodeResult, ChatState, ToolCall, merge_sources
from app.core.config import config
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


def visited(*nodes: ChatNode) -> tuple[ChatNodeResult, ...]:
    return tuple(ChatNodeResult(node=node, ms=1) for node in nodes)


class TestMergeSources:
    def test_appends_new_chunks_after_existing_in_arrival_order(self):
        existing = (search_result(id=1),)
        additions = [search_result(id=2), search_result(id=3)]

        merged = merge_sources(existing, additions, cap=10)

        assert tuple(chunk.id for chunk in merged) == (1, 2, 3)

    def test_deduplicates_by_chunk_id_keeping_the_earlier_chunk(self):
        existing = (search_result(id=1, text="first form"),)
        additions = [search_result(id=1, text="refetched form"), search_result(id=2)]

        merged = merge_sources(existing, additions, cap=10)

        assert tuple(chunk.id for chunk in merged) == (1, 2)
        assert merged[0].text == "first form"

    def test_stops_appending_at_the_cap_so_earlier_context_wins(self):
        existing = (search_result(id=1), search_result(id=2))
        additions = [search_result(id=3), search_result(id=4)]

        merged = merge_sources(existing, additions, cap=3)

        assert tuple(chunk.id for chunk in merged) == (1, 2, 3)

    def test_existing_beyond_the_cap_is_kept_but_nothing_is_added(self):
        existing = (search_result(id=1), search_result(id=2))

        merged = merge_sources(existing, [search_result(id=3)], cap=2)

        assert tuple(chunk.id for chunk in merged) == (1, 2)


class TestToolRounds:
    def test_counts_only_tools_visits(self):
        state = ChatState(
            question="q",
            nodes=visited(ChatNode.RETRIEVE, ChatNode.ASSESS, ChatNode.TOOLS, ChatNode.ASSESS),
        )
        assert state.tool_rounds() == 1


class TestContextSettled:
    def test_not_settled_before_any_node(self):
        assert ChatState(question="q").context_settled is False

    def test_after_retrieve_with_loop_enabled_the_loop_still_runs(self, monkeypatch):
        monkeypatch.setattr(config, "ASSESS_ENABLED", True)
        state = ChatState(
            question="q", nodes=visited(ChatNode.RETRIEVE), sources=(search_result(),)
        )
        assert state.context_settled is False

    def test_after_retrieve_with_loop_disabled_context_is_settled(self, monkeypatch):
        monkeypatch.setattr(config, "ASSESS_ENABLED", False)
        state = ChatState(
            question="q", nodes=visited(ChatNode.RETRIEVE), sources=(search_result(),)
        )
        assert state.context_settled is True

    def test_after_a_gated_retrieve_context_is_settled_for_the_refusal(self, monkeypatch):
        monkeypatch.setattr(config, "ASSESS_ENABLED", True)
        state = ChatState(question="q", nodes=visited(ChatNode.RETRIEVE), sources=())
        assert state.context_settled is True

    def test_assess_asking_for_tools_is_not_settled(self):
        state = ChatState(
            question="q",
            nodes=visited(ChatNode.RETRIEVE, ChatNode.ASSESS),
            sources=(search_result(),),
            pending_calls=(ToolCall(name="search", args={"query": "penalties"}),),
        )
        assert state.context_settled is False

    def test_assess_asking_for_nothing_is_settled(self):
        state = ChatState(
            question="q",
            nodes=visited(ChatNode.RETRIEVE, ChatNode.ASSESS),
            sources=(search_result(),),
        )
        assert state.context_settled is True

    def test_tools_visit_below_the_round_cap_is_not_settled(self, monkeypatch):
        monkeypatch.setattr(config, "ASSESS_MAX_ROUNDS", 2)
        state = ChatState(
            question="q",
            nodes=visited(ChatNode.RETRIEVE, ChatNode.ASSESS, ChatNode.TOOLS),
            sources=(search_result(),),
        )
        assert state.context_settled is False

    def test_tools_visit_consuming_the_round_cap_is_settled(self, monkeypatch):
        monkeypatch.setattr(config, "ASSESS_MAX_ROUNDS", 1)
        state = ChatState(
            question="q",
            nodes=visited(ChatNode.RETRIEVE, ChatNode.ASSESS, ChatNode.TOOLS),
            sources=(search_result(),),
        )
        assert state.context_settled is True
