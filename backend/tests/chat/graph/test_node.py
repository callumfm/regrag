"""The client every chat node calls through."""

from pydantic import SecretStr

from app.chat.graph.node import chat_model
from app.core.config import config


def test_the_client_carries_the_chat_settings(monkeypatch):
    monkeypatch.setattr(config, "CHAT_MODEL", "openai/gpt-5")
    monkeypatch.setattr(config, "CHAT_MAX_TOKENS", 321)
    monkeypatch.setattr(config, "CHAT_TIMEOUT", 21)

    client = chat_model(streaming=False)

    assert client.model == "openai/gpt-5"
    assert client.max_tokens == 321
    assert client.request_timeout == 21
    assert client.temperature == config.CHAT_TEMPERATURE
    assert client.streaming is False


def test_the_client_carries_the_key_of_the_provider_the_model_names(monkeypatch):
    """A model at another provider must work as a setting, so the key is looked up from the
    model rather than pinned to one provider's variable."""
    monkeypatch.setattr(config, "CHAT_MODEL", "openai/gpt-5")
    monkeypatch.setattr(config, "OPENAI_API_KEY", SecretStr("sk-openai"))

    assert chat_model().api_key == "sk-openai"


def test_the_answer_streams_and_asks_for_its_usage():
    """litellm strips usage from every streamed chunk unless asked, and the run's tokens
    then go unreported for any model but an OpenAI one."""
    client = chat_model()

    assert client.streaming is True
    assert client.stream_options == {"include_usage": True}
