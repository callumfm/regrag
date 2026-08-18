"""What one chat request cost so far, filled in as the graph's stream items pass."""

import time

from langchain_core.messages.ai import UsageMetadata
from pydantic import Field

from app.core.clock import elapsed_ms
from app.core.models import AppModel


class RequestStats(AppModel):
    """Per-stage timings, source count and model usage for one request, in flight."""

    start: float = Field(default_factory=time.perf_counter)
    retrieve_ms: int | None = None
    ttft_ms: int | None = None
    sources: int = 0
    usage: UsageMetadata | None = None

    def retrieved(self, sources: int) -> None:
        self.retrieve_ms, self.sources = elapsed_ms(self.start), sources

    def token(self) -> None:
        if self.ttft_ms is None:
            self.ttft_ms = elapsed_ms(self.start)
