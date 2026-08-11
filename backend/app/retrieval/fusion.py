"""Reciprocal Rank Fusion: merge two rank lists without comparing their scores."""

from collections.abc import Sequence

from app.retrieval.constants import RRF_K
from app.retrieval.models import FusedRank


def _ranks(chunk_ids: Sequence[int]) -> dict[int, int]:
    """Each chunk's 1-based position in a leg's result list."""
    return {chunk_id: rank for rank, chunk_id in enumerate(chunk_ids, start=1)}


def reciprocal_rank_fusion(
    vector_ids: Sequence[int], text_ids: Sequence[int], *, k: int = RRF_K
) -> list[FusedRank]:
    """Fuse two rank lists by 1/(k + rank), best first, chunk id breaking ties."""
    vector_ranks, text_ranks = _ranks(vector_ids), _ranks(text_ids)
    fused = [
        FusedRank(
            chunk_id=chunk_id,
            score=sum(
                1 / (k + rank)
                for rank in (vector_ranks.get(chunk_id), text_ranks.get(chunk_id))
                if rank is not None
            ),
            vector_rank=vector_ranks.get(chunk_id),
            text_rank=text_ranks.get(chunk_id),
        )
        for chunk_id in vector_ranks | text_ranks
    ]
    return sorted(fused, key=lambda rank: (-rank.score, rank.chunk_id))
