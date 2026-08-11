"""The CELLAR SPARQL endpoint: the topic query, and the bindings it answers with."""

from app.ingestion import celex
from app.ingestion.discover.models import CandidateAct

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


def collect_candidate_acts(payload: dict) -> list[CandidateAct]:
    """One act per celex, its bindings folded together; non-legislation sectors dropped."""
    in_force: dict[str, str] = {}
    consolidations: dict[str, set[str]] = {}
    for binding in payload["results"]["bindings"]:
        celex_id = binding["c"]["value"]
        if not celex.is_legislation(celex_id):
            continue
        consolidations.setdefault(celex_id, set())
        if "force" in binding:
            in_force[celex_id] = binding["force"]["value"]
        if "cons" in binding:
            consolidations[celex_id].add(binding["cons"]["value"])
    return [
        CandidateAct(
            celex=celex_id, in_force=in_force.get(celex_id), consolidations=frozenset(cons)
        )
        for celex_id, cons in sorted(consolidations.items())
    ]
