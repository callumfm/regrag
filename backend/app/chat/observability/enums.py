"""Chat run enumerations."""

from enum import StrEnum


class ChatOutcome(StrEnum):
    """How a chat stream ended.

    DONE: the client received the whole answer and the done event.
    ERROR: an error event ended the stream; the timings cover what ran before it.
    ABORTED: the client left before the stream finished.
    """

    DONE = "done"
    ERROR = "error"
    ABORTED = "aborted"
