"""The client every chat node calls through."""

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


def test_the_client_passes_no_key_so_the_provider_reads_its_own(monkeypatch):
    """A model at another provider must work as a setting; a key from config would pin
    every node to the one provider whose variable config happened to name."""
    monkeypatch.setattr(config, "CHAT_MODEL", "gemini/gemini-2.5-pro")

    assert chat_model().api_key is None


def test_the_answer_streams_and_asks_for_its_usage():
    """litellm strips usage from every streamed chunk unless asked, and the run's tokens
    then go unreported for any model but an OpenAI one."""
    client = chat_model()

    assert client.streaming is True
    assert client.stream_options == {"include_usage": True}
