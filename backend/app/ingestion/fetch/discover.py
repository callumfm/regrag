"""Corpus discovery from the CELLAR graph: one SPARQL query per topic seed."""

import httpx

from app.core.http import transient_retry
from app.core.models import FrozenModel
from app.ingestion.exceptions import DiscoveryError

SEEDS: dict[str, str] = {"fueleu": "32023R1805", "mrv": "32015R0757"}
SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"


class DocumentSpec(FrozenModel):
    """One discovered corpus document; transient wire data between discover and fetch."""

    topic: str
    source: str
    ref: str
    candidate_ref: str | None


def topic_query(seed_ref: str) -> str:
    """Acts citing the seed as legal basis (plus the seed), with in-force + consolidations."""
    seed_uri = f"http://publications.europa.eu/resource/celex/{seed_ref}"
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


def _is_legislation(celex: str) -> bool:
    return celex.startswith("3") and len(celex) > 5 and celex[5] in "RLD"


def parse_topic_response(topic: str, payload: dict) -> list[DocumentSpec]:
    """Apply the mechanical filters: legislation-only, in-force, not-folded."""
    in_force: dict[str, str] = {}
    consolidations: dict[str, set[str]] = {}
    for b in payload["results"]["bindings"]:
        ref = b["c"]["value"]
        if not _is_legislation(ref):
            continue
        consolidations.setdefault(ref, set())
        if "force" in b:
            in_force[ref] = b["force"]["value"]
        if "cons" in b:
            consolidations[ref].add(b["cons"]["value"])
    specs = []
    for ref, cons in sorted(consolidations.items()):
        if in_force.get(ref) != "1":
            continue
        own_stem = {c for c in cons if c.startswith(f"0{ref[1:]}-")}
        if cons and not own_stem:
            continue
        specs.append(
            DocumentSpec(
                topic=topic,
                source="eurlex",
                ref=ref,
                candidate_ref=max(own_stem) if own_stem else None,
            )
        )
    return specs


@transient_retry
def discover(client: httpx.Client, topic: str, seed_ref: str) -> list[DocumentSpec]:
    """Run the topic query and parse it; a result set without the seed act is an error."""
    response = client.get(
        SPARQL_ENDPOINT,
        params={"query": topic_query(seed_ref), "format": "application/sparql-results+json"},
    )
    response.raise_for_status()
    specs = parse_topic_response(topic, response.json())
    if not any(spec.ref == seed_ref for spec in specs):
        raise DiscoveryError(f"{topic}: seed {seed_ref} missing from discovery results")
    return specs
