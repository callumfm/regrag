"""What one chat stream cost: per-stage timings and model usage, logged once it ends."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages.ai import UsageMetadata

logger = logging.getLogger(__name__)


@dataclass
class StreamStats:
    """What one stream cost, read off the graph's stream items and logged once at the end."""

    start: float = field(default_factory=time.perf_counter)
    retrieve_ms: int | None = None
    ttft_ms: int | None = None
    sources: int = 0
    usage: UsageMetadata | None = None

    def _elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.start) * 1000)

    def observe(self, mode: str, data: Any) -> None:
        """Retrieval ends at the retrieve update, the answer starts at the first text
        chunk, and the synthesize update carries the model's usage."""
        if mode == "updates" and "retrieve" in data:
            self.retrieve_ms = self._elapsed_ms()
            self.sources = len(data["retrieve"]["sources"])
        elif mode == "updates" and "synthesize" in data:
            self.usage = data["synthesize"]["usage"]
        elif mode == "messages" and self.ttft_ms is None:
            chunk, _ = data
            if chunk.text:
                self.ttft_ms = self._elapsed_ms()

    def log(self, outcome: str) -> None:
        """The one stats line per stream; the request ID rides in on the log context."""
        total_ms = self._elapsed_ms()
        input_tokens = self.usage["input_tokens"] if self.usage else None
        output_tokens = self.usage["output_tokens"] if self.usage else None
        logger.info(
            "chat %s - retrieve %sms, first token %sms, total %sms, %s sources, %s/%s tokens",
            outcome,
            self.retrieve_ms,
            self.ttft_ms,
            total_ms,
            self.sources,
            input_tokens,
            output_tokens,
            extra={
                "outcome": outcome,
                "retrieve_ms": self.retrieve_ms,
                "ttft_ms": self.ttft_ms,
                "total_ms": total_ms,
                "sources": self.sources,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )
