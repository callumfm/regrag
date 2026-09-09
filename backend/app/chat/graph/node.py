"""What every chat node is built from: the contract a node fulfils, the wrapper that times
it, the model client its call is made with, and the shape it reads the answer back in."""

import functools
import logging
import time
from collections.abc import Awaitable
from typing import Any, ClassVar, Protocol, Self

from langchain_core.messages import BaseMessage
from langchain_litellm import ChatLiteLLM
from pydantic import ValidationError

from app.chat.enums import ChatNode
from app.chat.models import ChatState, ChatStepResult
from app.core.clock import elapsed_ms
from app.core.config import config
from app.core.llm import LLMError
from app.core.models import FrozenModel

logger = logging.getLogger(__name__)


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


def chat_model(model: str, *, streaming: bool = True) -> ChatLiteLLM:
    """A chat client built per call, so config is read at call time like embed's. The model
    is the caller's, so assess and the answer can be pointed at different ones.

    Streaming is set for the answer, or litellm answers in one blocking call — even under
    the graph's messages stream — and the SSE stream carries the whole answer in a single
    token event; assess wants that one blocking call, so it turns streaming off.
    Usage is asked for, or litellm strips it from every streamed chunk and the run's
    tokens are never reported for a non-OpenAI model; it is read on the streamed path only.
    """
    return ChatLiteLLM(
        model=model,
        api_key=config.ANTHROPIC_API_KEY.get_secret_value(),
        max_tokens=config.CHAT_MAX_TOKENS,
        temperature=config.CHAT_TEMPERATURE,
        request_timeout=config.CHAT_TIMEOUT,
        streaming=streaming,
        stream_options={"include_usage": True},
    )


class LLMResponse(FrozenModel):
    """A shape a node binds its model's answer to, and reads that answer back in. The node
    is named on the subclass, so a bad answer is logged and raised as that node's."""

    node: ClassVar[ChatNode]

    @classmethod
    def from_response(cls, response: BaseMessage) -> Self:
        """The answer in this shape; one off the schema is a failed call, which the node
        settles rather than failing the run."""
        try:
            return cls.model_validate_json(response.text)
        except ValidationError as exc:
            logger.warning("%s answered off its schema: %s", cls.node, exc)
            raise LLMError(f"{cls.node} answered off its schema") from exc
