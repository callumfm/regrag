"""Ingestion failures: raised in CLI runs, recorded per document on the run report."""


class IngestionError(Exception):
    """Base for failures during a corpus ingest run."""


class DiscoveryError(IngestionError):
    """Discovery returned an implausible result (e.g. seed act missing)."""


class ResolutionError(IngestionError):
    """No fetchable HTML could be found for a discovered document."""


class ParseError(IngestionError):
    """A document could not be parsed into a section tree."""
