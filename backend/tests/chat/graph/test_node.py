"""What every node is built from: the shape a bound answer is read back in."""

import pytest
from langchain_core.messages import AIMessage

from app.chat.enums import ChatNode
from app.chat.graph.node import LLMResponse
from app.core.llm import LLMError


class Verdict(LLMResponse):
    node = ChatNode.ASSESS

    ok: bool


def test_a_bound_answer_is_read_back_in_its_shape():
    assert Verdict.from_response(AIMessage(content='{"ok": true}')) == Verdict(ok=True)


def test_an_answer_off_the_schema_is_the_node_s_failed_call(caplog):
    with pytest.raises(LLMError) as raised:
        Verdict.from_response(AIMessage(content='{"ok": "maybe"}'))

    assert not raised.value.transient
    assert "assess answered off its schema" in caplog.text


def test_the_schema_the_model_is_shown_carries_no_node_field():
    assert "node" not in Verdict.model_json_schema()["properties"]
