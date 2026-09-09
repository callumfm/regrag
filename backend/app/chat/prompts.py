"""The wording more than one node shares: the thread note, and the numbered-context
formatting a citation marker refers to."""

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.chat.models import ChatTurn
from app.retrieval.models import RetrievedChunk

THREAD_NOTE = (
    " Earlier turns of the conversation come before the context; read them only to "
    "understand what the question refers to. They are not context to answer from: cite "
    "only this turn's numbered blocks."
)


def system_prompt(base: str, history: Sequence[ChatTurn]) -> str:
    """The system prompt as one turn sends it: the base alone on a first question, and
    with the thread note on a follow-up, so a first question's prompt is unchanged."""
    return f"{base}{THREAD_NOTE}" if history else base


def format_context_block(marker: int, source: RetrievedChunk) -> str:
    """One chunk as the numbered block a citation marker refers to."""
    return f"[{marker}] ({source.celex}, {source.citation})\n{source.text}"


def format_context(sources: Sequence[RetrievedChunk]) -> str:
    """The retrieved chunks as the numbered blocks the citation markers refer to."""
    return "\n\n".join(
        format_context_block(marker, source) for marker, source in enumerate(sources, start=1)
    )


def thread_messages(history: Sequence[ChatTurn]) -> list[BaseMessage]:
    """The thread's earlier turns as the message pairs a model reads them as, oldest first,
    to go between the system prompt and this turn's user message."""
    messages: list[BaseMessage] = []
    for turn in history:
        messages.append(HumanMessage(turn.question))
        messages.append(AIMessage(turn.answer))
    return messages
