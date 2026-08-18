"""Chat enumerations."""

from enum import StrEnum


class ChatNode(StrEnum):
    """The graph's nodes, as astream keys their updates."""

    RETRIEVE = "retrieve"
    SYNTHESIZE = "synthesize"


class ChatEventName(StrEnum):
    """The SSE event names a chat stream carries."""

    SOURCES = "sources"
    TOKEN = "token"
    DONE = "done"
    ERROR = "error"


class ChatOutcome(StrEnum):
    """How a chat stream ended: done, an error event, or the client leaving first.
    An ERROR run's timings cover what ran before the error."""

    DONE = "done"
    ERROR = "error"
    ABORTED = "aborted"
