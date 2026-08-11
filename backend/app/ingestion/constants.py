"""Ingestion tunables: the values a run config will override once one exists."""

SEEDS: dict[str, str] = {"fueleu": "32023R1805", "mrv": "32015R0757"}
MAX_CHARS = 2000
PACE_SECONDS = 1.0
MAX_DROP_RATIO = 0.2
MIN_SUSPICIOUS_DROPS = 3
EMBED_PAGE_SIZE = 500
