"""Ingestion failures: raised in CLI runs, recorded per document on the run report."""


class IngestionError(Exception):
    """Base for failures during a corpus ingest run."""


class DiscoveryError(IngestionError):
    """Base for failures that make a topic's discovery result unusable."""


class MalformedDiscoveryError(DiscoveryError):
    """The SPARQL response was unreadable, or came back without its seed act."""


class CorpusShrankError(DiscoveryError):
    """Discovery lost an implausible share of the previous run, so it is not trusted as a repeal."""


class NoFetchableVersionError(IngestionError):
    """None of a document's candidate celexes served HTML."""


class DocumentStillRenderingError(IngestionError):
    """EUR-Lex is generating the document on demand; it is worth asking again later."""


class EmptyDownloadError(IngestionError):
    """A download returned no bytes, which is never a valid source document."""


class ParseError(IngestionError):
    """A document could not be parsed into a section tree."""
