"""Chat failures that end a request before the graph runs."""

from fastapi import status

from app.core.exceptions import DomainError


class ThreadFullError(DomainError):
    """The thread already holds as many answered turns as a thread may; the caller starts
    a new one."""

    status_code = status.HTTP_409_CONFLICT

    def __init__(self, turns: int) -> None:
        super().__init__(
            f"This thread has reached its {turns} turns; start a new thread to keep asking"
        )
