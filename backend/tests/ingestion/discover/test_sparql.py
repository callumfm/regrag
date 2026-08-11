"""The CELLAR topic query, and folding the bindings it answers with."""

from app.ingestion.discover.sparql import collect_candidate_acts, topic_query
from tests.conftest import binding, payload


def test_topic_query_embeds_seed():
    q = topic_query("32023R1805")
    assert "resource/celex/32023R1805" in q
    assert "resource_legal_based_on_resource_legal" in q


def test_collect_candidate_acts_folds_every_binding_for_one_celex():
    p = payload(
        binding("32015R0757", force="1", cons="02015R0757-20240101"),
        binding("32015R0757", force="1", cons="02015R0757-20250101"),
    )
    acts = collect_candidate_acts(p)
    assert len(acts) == 1
    assert acts[0].consolidations == frozenset({"02015R0757-20240101", "02015R0757-20250101"})
