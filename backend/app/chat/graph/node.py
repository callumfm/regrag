"""What every chat node is built from: the contract a node fulfils, the wrapper that times
it, and the model client its call is made with."""

import functools
import time
from collections.abc import Awaitable
from typing import Any, Protocol

from langchain_litellm import ChatLiteLLM

from app.chat.enums import ChatNode
from app.chat.models import ChatState, ChatStepResult
from app.core.clock import elapsed_ms
from app.core.config import config
from app.core.llm.settings import ModelSettings


class NodeFn(Protocol):
    """A node: the state so far in, the fields it sets out — plus `usage`, if it called a
    model, which its step carries rather than the state. Named for the ChatNode it is."""

    __name__: str

    def __call__(self, state: ChatState) -> Awaitable[dict[str, Any]]: ...


def traced(run: NodeFn) -> NodeFn:
    """The node as the graph runs it, appending one step — how long it took, and the usage
    it reported — to the path. Outermost on a node, so a retried call is traced as a whole."""
    node = ChatNode(run.__name__)

    @functools.wraps(run)
    async def traced_run(state: ChatState) -> dict[str, Any]:
        start = time.perf_counter()
        update = await run(state)
        usage = update.pop("usage", None)
        step = ChatStepResult.from_usage(node, elapsed_ms(start), usage)
        return update | {"steps": (step,)}

    return traced_run


def graph_models() -> tuple[str, ...]:
    """The models the graph's four roles call, for the key check its callers run at startup.
    Named whether or not their node is switched on: a switch is flipped without a restart."""
    return (config.CHAT_MODEL, config.ASSESS_MODEL, config.DECOMPOSE_MODEL, config.REWRITE_MODEL)


def chat_model(model: str, *, streaming: bool = True) -> ChatLiteLLM:
    """A chat client built per call, so config is read at call time like embed's. The model
    is the caller's, so assess and the answer can be pointed at different ones — at different
    providers too: no key is passed, so litellm reads whichever variable the provider wants.
    The four graph roles differ in which model answers, not in the limits it answers under.

    Streaming is set for the answer, or litellm answers in one blocking call — even under
    the graph's messages stream — and the SSE stream carries the whole answer in a single
    token event; assess wants that one blocking call, so it turns streaming off.
    Usage is asked for, or litellm strips it from every streamed chunk and the run's
    tokens are never reported for a non-OpenAI model; it is read on the streamed path only.
    """
    settings = ModelSettings(
        model=model,
        timeout=config.CHAT_TIMEOUT,
        max_tokens=config.CHAT_MAX_TOKENS,
        temperature=config.CHAT_TEMPERATURE,
    )
    return ChatLiteLLM(
        model=settings.model,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        request_timeout=settings.timeout,
        streaming=streaming,
        stream_options={"include_usage": True},
    )
