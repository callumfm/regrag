"""CELLAR discovery: SPARQL response parsing, filters, seed guard."""

import httpx
import pytest

from app.ingestion.discover import (
    SEEDS,
    DiscoveryError,
    DocumentSpec,
    discover,
    parse_topic_response,
    topic_query,
)


def binding(celex, force=None, cons=None):
    b = {"c": {"value": celex}}
    if force is not None:
        b["force"] = {"value": force}
    if cons is not None:
        b["cons"] = {"value": cons}
    return b


def payload(*bindings):
    return {"results": {"bindings": list(bindings)}}


def test_non_legislation_sectors_filtered():
    p = payload(
        binding("32015R0757", force="1"),
        binding("52024XC07469"),
        binding("52024IP0025"),
        binding("E2021X0415(01)"),
    )
    assert [s.ref for s in parse_topic_response("mrv", p)] == ["32015R0757"]


def test_not_in_force_filtered():
    p = payload(binding("32016R1927", force="0"), binding("32016R1928", force="1"))
    assert [s.ref for s in parse_topic_response("mrv", p)] == ["32016R1928"]


def test_folded_amendment_filtered():
    p = payload(
        binding("32023R2776", force="1", cons="02015R0757-20240101"),
        binding("32015R0757", force="1", cons="02015R0757-20240101"),
    )
    specs = parse_topic_response("mrv", p)
    assert [s.ref for s in specs] == ["32015R0757"]
    assert specs[0].candidate_ref == "02015R0757-20240101"


def test_candidate_is_max_own_stem_consolidation():
    p = payload(
        binding("32015R0757", force="1", cons="02015R0757-20240101"),
        binding("32015R0757", force="1", cons="02015R0757-20250101"),
        binding("32015R0757", force="1", cons="02015R0757-20161216"),
    )
    assert parse_topic_response("mrv", p)[0].candidate_ref == "02015R0757-20250101"


def test_no_consolidations_gives_none_candidate():
    p = payload(binding("32023R2449", force="1"))
    assert parse_topic_response("mrv", p)[0].candidate_ref is None


def test_specs_carry_topic_and_source():
    p = payload(binding("32023R1805", force="1"))
    spec = parse_topic_response("fueleu", p)[0]
    assert spec == DocumentSpec(
        topic="fueleu", source="eurlex", ref="32023R1805", candidate_ref=None
    )


def test_topic_query_embeds_seed():
    q = topic_query("32023R1805")
    assert "resource/celex/32023R1805" in q
    assert "resource_legal_based_on_resource_legal" in q


def test_discover_raises_when_seed_missing_from_results():
    def handler(request):
        return httpx.Response(200, json=payload(binding("32023R2449", force="1")))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DiscoveryError, match="32015R0757"):
            discover(client, "mrv", "32015R0757")


def test_discover_returns_parsed_specs():
    def handler(request):
        assert request.url.params["format"] == "application/sparql-results+json"
        return httpx.Response(
            200,
            json=payload(
                binding("32015R0757", force="1", cons="02015R0757-20250101"),
                binding("32023R2449", force="1"),
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        specs = discover(client, "mrv", "32015R0757")
    assert [s.ref for s in specs] == ["32015R0757", "32023R2449"]


def test_seeds_are_fueleu_and_mrv():
    assert SEEDS == {"fueleu": "32023R1805", "mrv": "32015R0757"}
