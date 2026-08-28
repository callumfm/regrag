"""Chat run state: what a graph snapshot refreshes, and what it leaves alone."""

from app.chat.enums import ChatNode, ToolStep
from app.chat.models import ChatState, ChatStepResult, ToolCall
from app.core.config import config
from app.core.exceptions import DomainError
from tests.conftest import search_result


def test_refresh_folds_the_snapshot_on_and_leaves_the_consumer_fields_alone():
    """A values snapshot carries only the graph's channels; total_ms and error, set by the
    stream's consumer, must survive it."""
    state = ChatState(question="q", total_ms=5, error="boom")
    retrieved = ChatStepResult(step=ChatNode.RETRIEVE, ms=12)

    state.refresh({"question": "q", "steps": (retrieved,), "sources": (), "answer": ""})

    assert state.steps == (retrieved,)
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


def visited(*steps: ChatNode | ToolStep) -> tuple[ChatStepResult, ...]:
    return tuple(ChatStepResult(step=step, ms=1) for step in steps)


class TestAssessRounds:
    def test_counts_assess_visits_not_the_calls_they_asked_for(self):
        """A round is one assess visit however many tools it ran, so the budget is spent
        by asking, not by fanning out."""
        state = ChatState(
            question="q",
            steps=visited(ChatNode.RETRIEVE, ChatNode.ASSESS, ToolStep.SEARCH, ToolStep.SEARCH),
        )
        assert state.assess_rounds() == 1


class TestContextSettled:
    def test_not_settled_before_any_node(self):
        assert ChatState(question="q").context_settled is False

    def test_after_retrieve_with_loop_enabled_the_loop_still_runs(self, monkeypatch):
        monkeypatch.setattr(config, "ASSESS_ENABLED", True)
        state = ChatState(
            question="q", steps=visited(ChatNode.RETRIEVE), sources=(search_result(),)
        )
        assert state.context_settled is False

    def test_after_retrieve_with_loop_disabled_context_is_settled(self, monkeypatch):
        monkeypatch.setattr(config, "ASSESS_ENABLED", False)
        state = ChatState(
            question="q", steps=visited(ChatNode.RETRIEVE), sources=(search_result(),)
        )
        assert state.context_settled is True

    def test_after_a_gated_retrieve_context_is_settled_for_the_refusal(self, monkeypatch):
        monkeypatch.setattr(config, "ASSESS_ENABLED", True)
        state = ChatState(question="q", steps=visited(ChatNode.RETRIEVE), sources=())
        assert state.context_settled is True

    def test_assess_asking_for_tools_is_not_settled(self):
        state = ChatState(
            question="q",
            steps=visited(ChatNode.RETRIEVE, ChatNode.ASSESS),
            sources=(search_result(),),
            pending_calls=(ToolCall(name="search", args={"query": "penalties"}),),
        )
        assert state.context_settled is False

    def test_assess_asking_for_nothing_is_settled(self):
        state = ChatState(
            question="q",
            steps=visited(ChatNode.RETRIEVE, ChatNode.ASSESS),
            sources=(search_result(),),
        )
        assert state.context_settled is True

    def test_tool_step_below_the_round_cap_is_not_settled(self, monkeypatch):
        monkeypatch.setattr(config, "ASSESS_MAX_ROUNDS", 2)
        state = ChatState(
            question="q",
            steps=visited(ChatNode.RETRIEVE, ChatNode.ASSESS, ToolStep.SEARCH),
            sources=(search_result(),),
        )
        assert state.context_settled is False

    def test_tool_step_consuming_the_round_cap_is_settled(self, monkeypatch):
        monkeypatch.setattr(config, "ASSESS_MAX_ROUNDS", 1)
        state = ChatState(
            question="q",
            steps=visited(ChatNode.RETRIEVE, ChatNode.ASSESS, ToolStep.SEARCH),
            sources=(search_result(),),
        )
        assert state.context_settled is True
