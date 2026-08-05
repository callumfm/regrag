"""Every ingestion failure is catchable as one family."""

import pytest

from app.ingestion.exceptions import (
    DiscoveryError,
    IngestionError,
    ParseError,
    ResolutionError,
)


@pytest.mark.parametrize("error", [DiscoveryError, ResolutionError, ParseError])
def test_ingestion_failures_share_a_base(error: type[IngestionError]) -> None:
    with pytest.raises(IngestionError):
        raise error("boom")


def test_ingestion_errors_are_not_domain_errors() -> None:
    from app.core.exceptions import DomainError

    assert not issubclass(IngestionError, DomainError)
