"""What one chat stream cost so far, read off the graph's stream items as they pass."""

import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages.ai import UsageMetadata


@dataclass
class StreamStats:
    """Per-stage timings, source count and model usage for one stream, in flight."""

    start: float = field(default_factory=time.perf_counter)
    retrieve_ms: int | None = None
    ttft_ms: int | None = None
    sources: int = 0
    usage: UsageMetadata | None = None

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.start) * 1000)

    def observe(self, mode: str, data: Any) -> None:
        """Retrieval ends at the retrieve update, the answer starts at the first text
        chunk, and the synthesize update carries the model's usage."""
        if mode == "updates" and "retrieve" in data:
            self.retrieve_ms = self.elapsed_ms()
            self.sources = len(data["retrieve"]["sources"])
        elif mode == "updates" and "synthesize" in data:
            self.usage = data["synthesize"]["usage"]
        elif mode == "messages" and self.ttft_ms is None:
            chunk, _ = data
            if chunk.text:
                self.ttft_ms = self.elapsed_ms()
