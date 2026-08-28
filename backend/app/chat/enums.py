"""Chat enumerations."""

from enum import StrEnum


class ChatNode(StrEnum):
    """The graph's nodes, as astream keys their updates."""

    RETRIEVE = "retrieve"
    ASSESS = "assess"
    TOOLS = "tools"
    SYNTHESIZE = "synthesize"
    REFUSE = "refuse"


class ToolStep(StrEnum):
    """The tool calls a path records, prefixed so one column holds both these and the
    graph's nodes without either being read for the other."""

    SEARCH = "tool_search"
    FOLLOW_REFERENCE = "tool_follow_reference"
    UNKNOWN = "tool_unknown"
    """A call to a tool the surface does not have, kept in the path because a model asking
    for one is worth seeing, and because a round that ran must leave a step behind."""


class ChatEventName(StrEnum):
    """The events a chat stream carries, as the SSE frames name them."""

    SOURCES = "sources"
    TEXT = "text"
    DONE = "done"
    ERROR = "error"


class ChatOutcome(StrEnum):
    """How a chat stream ended: done, refused before any model call, an error event,
    or the client leaving first. An ERROR run's timings cover what ran before the error."""

    DONE = "done"
    REFUSED = "refused"
    ERROR = "error"
    ABORTED = "aborted"
