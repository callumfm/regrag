"""Shared retry policy for transient failures, parameterised by what counts as transient."""

import logging
from collections.abc import Callable

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3


def transient_retry(is_transient: Callable[[BaseException], bool]) -> Callable:
    """Retry what the predicate accepts: three attempts, exponential backoff, then reraise."""
    return retry(
        retry=retry_if_exception(is_transient),
        stop=stop_after_attempt(MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
