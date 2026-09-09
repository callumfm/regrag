"""What a model call spends."""

import functools
import operator
from collections.abc import Iterable

from langchain_core.messages.ai import UsageMetadata

from app.core.models import FrozenModel


class TokenUsage(FrozenModel):
    """The tokens one model call spent, or several calls spent between them."""

    input_tokens: int
    output_tokens: int

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    @classmethod
    def from_metadata(cls, usage: UsageMetadata) -> "TokenUsage":
        """What one call spent, as langchain reports it on the message."""
        return cls(input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"])

    @classmethod
    def sum_reported(cls, usages: Iterable["TokenUsage | None"]) -> "TokenUsage | None":
        """The usages summed over the calls that reported one, or None when none did —
        unmeasured rather than free."""
        reported = [usage for usage in usages if usage is not None]
        return functools.reduce(operator.add, reported) if reported else None
