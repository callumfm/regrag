"""The client every chat node calls through, and the models its roles name."""

from app.chat.graph.node import chat_model, graph_models
from app.core.config import config


def test_the_client_carries_the_chat_limits_on_the_callers_model(monkeypatch):
    monkeypatch.setattr(config, "CHAT_MAX_TOKENS", 321)
    monkeypatch.setattr(config, "CHAT_TIMEOUT", 21)
    monkeypatch.setattr(config, "CHAT_TEMPERATURE", 0.0)

    client = chat_model("openai/gpt-5", streaming=False)

    assert client.model == "openai/gpt-5"
    assert client.max_tokens == 321
    assert client.request_timeout == 21
    assert client.temperature == 0.0
    assert client.streaming is False


def test_the_client_passes_no_key_so_the_provider_reads_its_own(monkeypatch):
    """A model at another provider must work as a setting; a key from config would pin
    every role to the one provider whose variable config happened to name."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-the-environment")

    assert chat_model("gemini/gemini-2.5-pro").api_key is None


def test_the_answer_streams_and_asks_for_its_usage():
    """litellm strips usage from every streamed chunk unless asked, and the run's tokens
    then go unreported for any model but an OpenAI one."""
    client = chat_model(config.CHAT_MODEL)

    assert client.streaming is True
    assert client.stream_options == {"include_usage": True}


def test_the_graph_names_every_role_it_may_call(monkeypatch):
    monkeypatch.setattr(config, "DECOMPOSE_MODEL", "openai/gpt-5")

    models = graph_models()

    assert set(models) == {
        config.CHAT_MODEL,
        config.ASSESS_MODEL,
        "openai/gpt-5",
        config.REWRITE_MODEL,
    }


def test_a_switched_off_node_still_names_its_model(monkeypatch):
    """A switch is flipped without a restart, so a key checked only when the node was on
    would leave the first flipped-on request to fail."""
    monkeypatch.setattr(config, "DECOMPOSE_ENABLED", False)
    monkeypatch.setattr(config, "ASSESS_ENABLED", False)

    assert config.DECOMPOSE_MODEL in graph_models()
    assert config.ASSESS_MODEL in graph_models()
