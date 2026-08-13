"""Talking to CELLAR: build the topic query, send it, and read the rows it answers with."""

from itertools import groupby

import httpx

from app.core.http import http_retry
from app.ingestion import celex
from app.ingestion.discover.models import CandidateAct

_SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"

ResultRow = dict[str, dict[str, str]]
"""One row of a result set, each variable name pointing at its value; a "binding" in SPARQL."""


def _topic_query(seed_celex: str) -> str:
    """Acts citing the seed as legal basis (plus the seed), with in-force + consolidations."""
    seed_uri = f"http://publications.europa.eu/resource/celex/{seed_celex}"
    return f"""PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
SELECT DISTINCT ?c ?force ?cons WHERE {{
  {{ ?act cdm:resource_legal_based_on_resource_legal ?seed .
    ?seed owl:sameAs <{seed_uri}> . }}
  UNION
  {{ ?act owl:sameAs <{seed_uri}> . }}
  ?act cdm:resource_legal_id_celex ?c .
  OPTIONAL {{ ?act cdm:resource_legal_in-force ?force }}
  OPTIONAL {{ ?consact cdm:act_consolidated_consolidates_resource_legal ?act .
    ?consact cdm:resource_legal_id_celex ?cons }}
}}"""


@http_retry
async def run_topic_query(client: httpx.AsyncClient, seed_celex: str) -> list[ResultRow]:
    """Ask CELLAR for one topic's acts, and hand back the rows it answers with."""
    response = await client.get(
        _SPARQL_ENDPOINT,
        params={"query": _topic_query(seed_celex), "format": "application/sparql-results+json"},
    )
    response.raise_for_status()
    return response.json()["results"]["bindings"]


def _row_celex(row: ResultRow) -> str:
    return row["c"]["value"]


def _act_from_rows(celex_id: str, rows: list[ResultRow]) -> CandidateAct:
    """CELLAR repeats an act's celex and in-force flag on every row, one row per consolidation."""
    return CandidateAct(
        celex=celex_id,
        in_force=next((row["force"]["value"] for row in rows if "force" in row), None),
        consolidations=frozenset(row["cons"]["value"] for row in rows if "cons" in row),
    )


def extract_acts(rows: list[ResultRow]) -> list[CandidateAct]:
    """One act per celex, its rows folded together; rows for non-legislation are dropped."""
    legislation = [row for row in rows if celex.is_legislation(_row_celex(row))]
    legislation.sort(key=_row_celex)
    return [
        _act_from_rows(celex_id, list(act_rows))
        for celex_id, act_rows in groupby(legislation, key=_row_celex)
    ]
