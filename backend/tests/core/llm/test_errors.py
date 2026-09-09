"""The provider error contract: what an LLMError carries, and how the wrap point names a call."""

import httpx
import openai
import pytest
from pydantic import BaseModel

from app.core.llm.errors import (
    TRANSIENT_PROVIDER_ERRORS,
    LLMError,
    parse_model_answer,
    wrap_provider_errors,
)

pytestmark = pytest.mark.anyio


def test_llm_error_status_code():
    assert LLMError("x").status_code == 502


async def test_wrap_provider_errors_names_the_call_by_its_label_not_its_function():
    @wrap_provider_errors("frobnicate call")
    async def _internal_helper_name() -> None:
        raise openai.APIConnectionError(
            message="upstream", request=httpx.Request("POST", "http://provider.example")
        )

    with pytest.raises(LLMError, match="^frobnicate call failed$"):
        await _internal_helper_name()


def test_llm_error_is_not_transient_by_default():
    assert LLMError("nope").transient is False


def test_llm_error_records_transience_when_told():
    assert LLMError("nope", transient=True).transient is True


def test_transient_errors_exclude_the_permanent_ones():
    assert openai.RateLimitError in TRANSIENT_PROVIDER_ERRORS
    assert openai.AuthenticationError not in TRANSIENT_PROVIDER_ERRORS
    assert openai.BadRequestError not in TRANSIENT_PROVIDER_ERRORS


class Verdict(BaseModel):
    ok: bool


def test_an_answer_is_read_back_in_the_shape_the_call_bound() -> None:
    assert parse_model_answer(Verdict, '{"ok": true}', label="judge") == Verdict(ok=True)


def test_an_answer_off_the_schema_is_a_failed_call_named_for_its_label(caplog) -> None:
    with pytest.raises(LLMError, match="^judge answered off its schema$") as raised:
        parse_model_answer(Verdict, '{"ok": "maybe"}', label="judge")

    assert not raised.value.transient
    assert "judge answered off its schema: " in caplog.text


def test_why_the_model_stopped_is_logged_when_the_caller_knows(caplog) -> None:
    with pytest.raises(LLMError):
        parse_model_answer(Verdict, '{"ok"', label="judge", stopped_on="length")

    assert "judge answered off its schema, stopped on length: " in caplog.text
