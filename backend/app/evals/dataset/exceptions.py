"""Dataset failures: raised in CLI runs, printed as the reason the command did nothing."""


class DatasetError(Exception):
    """Base for a dataset the CLI cannot go on with."""


class EmptyDatasetError(DatasetError):
    """A dataset load that selected no cases."""


class UnresolvedReferenceError(DatasetError):
    """A stamp run over a reference no stored chunk answers to, which it cannot record."""
