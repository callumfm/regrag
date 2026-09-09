"""The provider error contract: what an LLMError carries, and how the wrap point names a call."""

import httpx
import openai
import pytest

from app.core.llm.errors import TRANSIENT_PROVIDER_ERRORS, LLMError, wrap_provider_errors

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
