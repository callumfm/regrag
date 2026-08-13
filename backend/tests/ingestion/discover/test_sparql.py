"""The CELLAR topic query, and folding the rows it answers with into acts."""

import httpx
import pytest

from app.ingestion.discover.sparql import extract_acts, run_topic_query
from tests.conftest import binding, payload

pytestmark = pytest.mark.anyio


async def test_topic_query_asks_for_the_seed_and_json_results():
    def handler(request):
        query = request.url.params["query"]
        assert "resource/celex/32023R1805" in query
        assert "resource_legal_based_on_resource_legal" in query
        assert request.url.params["format"] == "application/sparql-results+json"
        return httpx.Response(200, json=payload(binding("32023R1805", force="1")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rows = await run_topic_query(client, "32023R1805")

    assert rows == [binding("32023R1805", force="1")]


def test_extract_acts_folds_every_row_for_one_celex():
    rows = [
        binding("32015R0757", force="1", cons="02015R0757-20240101"),
        binding("32015R0757", force="1", cons="02015R0757-20250101"),
    ]
    acts = extract_acts(rows)
    assert len(acts) == 1
    assert acts[0].consolidations == frozenset({"02015R0757-20240101", "02015R0757-20250101"})


def test_extract_acts_takes_in_force_from_whichever_row_carries_it():
    rows = [
        binding("32015R0757", cons="02015R0757-20240101"),
        binding("32015R0757", force="1", cons="02015R0757-20250101"),
    ]
    assert extract_acts(rows)[0].in_force == "1"


def test_extract_acts_groups_interleaved_rows_by_celex():
    rows = [
        binding("32023R2449", force="1"),
        binding("32015R0757", force="1", cons="02015R0757-20240101"),
        binding("32023R2449", force="1", cons="02023R2449-20250101"),
    ]
    acts = extract_acts(rows)
    assert [act.celex for act in acts] == ["32015R0757", "32023R2449"]
    assert acts[1].consolidations == frozenset({"02023R2449-20250101"})


def test_extract_acts_keeps_an_act_that_has_no_consolidations():
    assert extract_acts([binding("32023R2449", force="1")])[0].consolidations == frozenset()
