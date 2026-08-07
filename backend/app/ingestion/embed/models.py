"""Embed-stage values: what one run's embedding pass changed."""

from app.ingestion.models import StageRunResult


class EmbedRunResult(StageRunResult):
    """How many chunk vectors a run filled in, and how many were already there."""

    embedded: int = 0
    unchanged: int = 0
