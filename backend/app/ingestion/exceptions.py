"""Ingestion failures: raised in CLI runs, recorded per document on the run report."""


class IngestionError(Exception):
    """Base for failures during a corpus ingest run."""


class DiscoveryError(IngestionError):
    """Discovery returned an implausible result (e.g. seed act missing)."""


class EmptyCorpusError(IngestionError):
    """A reparse was asked for before anything had been fetched, so there is nothing to reparse."""


class ResolutionError(IngestionError):
    """No fetchable HTML could be found for a discovered document."""


class DocumentNotReadyError(IngestionError):
    """EUR-Lex has the document but is still rendering it; it is worth asking again later."""


class EmptyDocumentError(IngestionError):
    """A download returned no bytes, which is never a valid source document."""


class ParseError(IngestionError):
    """A document could not be parsed into a section tree."""
