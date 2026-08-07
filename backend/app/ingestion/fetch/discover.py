"""Corpus discovery from the CELLAR graph: one SPARQL query per topic seed."""

import httpx

from app.core.http import transient_retry
from app.ingestion import celex
from app.ingestion.exceptions import DiscoveryError
from app.ingestion.fetch.models import DiscoveredDocument

SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"


def topic_query(seed_celex: str) -> str:
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


def parse_topic_response(topic: str, payload: dict) -> list[DiscoveredDocument]:
    """Apply the mechanical filters: legislation-only, in-force, not-folded."""
    in_force: dict[str, str] = {}
    consolidations: dict[str, set[str]] = {}
    for b in payload["results"]["bindings"]:
        celex_id = b["c"]["value"]
        if not celex.is_legislation(celex_id):
            continue
        consolidations.setdefault(celex_id, set())
        if "force" in b:
            in_force[celex_id] = b["force"]["value"]
        if "cons" in b:
            consolidations[celex_id].add(b["cons"]["value"])
    specs = []
    for celex_id, cons in sorted(consolidations.items()):
        if in_force.get(celex_id) != "1":
            continue
        own_stem = {c for c in cons if c.startswith(celex.consolidated_stem(celex_id))}
        if cons and not own_stem:
            continue
        specs.append(
            DiscoveredDocument(
                topic=topic,
                source="eurlex",
                celex=celex_id,
                candidate_celex=max(own_stem) if own_stem else None,
            )
        )
    return specs


@transient_retry
def discover(client: httpx.Client, topic: str, seed_celex: str) -> list[DiscoveredDocument]:
    """Run the topic query and parse it; a result set without the seed act is an error."""
    response = client.get(
        SPARQL_ENDPOINT,
        params={"query": topic_query(seed_celex), "format": "application/sparql-results+json"},
    )
    response.raise_for_status()
    specs = parse_topic_response(topic, response.json())
    if not any(spec.celex == seed_celex for spec in specs):
        raise DiscoveryError(f"{topic}: seed {seed_celex} missing from discovery results")
    return specs
