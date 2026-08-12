"""Retrieval tunables: the values a search config will override once one exists."""

RRF_K = 60
CANDIDATES = 50
DEFAULT_LIMIT = 10

RERANK_POOL = 30
"""How many fused results the cross-encoder sees before the cut to the caller's limit."""
