import pytest

from app.retrieval.constants import RRF_K
from app.retrieval.fusion import reciprocal_rank_fusion


def test_a_chunk_found_by_both_legs_scores_the_sum_of_its_two_ranks() -> None:
    (fused,) = reciprocal_rank_fusion([7], [7])

    assert fused.chunk_id == 7
    assert fused.score == 2 / (RRF_K + 1)
    assert fused.vector_rank == 1
    assert fused.text_rank == 1


def test_a_chunk_found_by_one_leg_records_no_rank_for_the_other() -> None:
    (fused,) = reciprocal_rank_fusion([7], [])

    assert fused.score == 1 / (RRF_K + 1)
    assert fused.vector_rank == 1
    assert fused.text_rank is None


def test_agreement_between_the_legs_outranks_a_better_position_in_one() -> None:
    fused = reciprocal_rank_fusion([1, 2], [2, 3])

    assert [rank.chunk_id for rank in fused] == [2, 1, 3]


def test_disjoint_legs_interleave_by_rank() -> None:
    fused = reciprocal_rank_fusion([1, 2], [3, 4])

    assert [rank.chunk_id for rank in fused] == [1, 3, 2, 4]


def test_ties_break_on_chunk_id_so_the_order_is_total() -> None:
    fused = reciprocal_rank_fusion([9, 4], [])

    assert [rank.chunk_id for rank in fused] == [9, 4]
    assert [rank.chunk_id for rank in reciprocal_rank_fusion([], [4, 9])] == [4, 9]


def test_two_empty_legs_fuse_to_nothing() -> None:
    assert reciprocal_rank_fusion([], []) == []


def test_the_fusion_constant_can_be_overridden_per_call() -> None:
    (fused,) = reciprocal_rank_fusion([7], [], k=0)

    assert fused.score == 1.0


def test_a_smaller_constant_sharpens_the_gap_between_ranks() -> None:
    spread = reciprocal_rank_fusion([1, 2], [], k=1)

    assert spread[0].score / spread[1].score == pytest.approx(1.5)
