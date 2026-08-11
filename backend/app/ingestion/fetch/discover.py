"""CELLAR discovery: one SPARQL query per topic seed, and the policy over what it returns."""

import json
from collections.abc import Iterable, Sequence

import httpx

from app.core.http import http_retry
from app.ingestion import celex
from app.ingestion.constants import MAX_DROP_RATIO, MIN_SUSPICIOUS_DROPS, SEEDS
from app.ingestion.exceptions import CorpusShrankError, MalformedDiscoveryError
from app.ingestion.fetch.models import CandidateAct, DiscoveredDocument

SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"
IN_FORCE = "1"


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


def own_consolidations(act: CandidateAct) -> set[str]:
    """The consolidated versions that are of this act itself, not of one that absorbed it."""
    stem = celex.consolidated_stem(act.celex)
    return {version for version in act.consolidations if version.startswith(stem)}


def is_in_force(act: CandidateAct) -> bool:
    """CELLAR flags a live act with '1'; repealed and unstated acts are not fetched."""
    return act.in_force == IN_FORCE


def is_folded_into_another_act(act: CandidateAct) -> bool:
    """Every consolidation of this act belongs to another act, which now supersedes it."""
    return bool(act.consolidations) and not own_consolidations(act)


def latest_own_consolidation(act: CandidateAct) -> str | None:
    """The newest consolidated version of this act, or None if it has never been consolidated."""
    own = own_consolidations(act)
    return max(own) if own else None


def version_candidates(spec: DiscoveredDocument) -> list[str]:
    """The versions to try in order: the consolidation discovery found, then the original act."""
    return [spec.candidate_celex, spec.celex] if spec.candidate_celex else [spec.celex]


def select_topic_documents(topic: str, acts: Iterable[CandidateAct]) -> list[DiscoveredDocument]:
    """The acts worth fetching, each pointed at the version to try first."""
    return [
        DiscoveredDocument(
            topic=topic,
            source="eurlex",
            celex=act.celex,
            candidate_celex=latest_own_consolidation(act),
        )
        for act in acts
        if is_in_force(act) and not is_folded_into_another_act(act)
    ]


@http_retry
def discover_topic(client: httpx.Client, topic: str, seed_celex: str) -> list[DiscoveredDocument]:
    """Run the topic query and select from it; a result set without the seed act is an error."""
    response = client.get(
        SPARQL_ENDPOINT,
        params={"query": topic_query(seed_celex), "format": "application/sparql-results+json"},
    )
    response.raise_for_status()
    documents = select_topic_documents(topic, collect_candidate_acts(response.json()))
    if not any(document.celex == seed_celex for document in documents):
        raise MalformedDiscoveryError(f"{topic}: seed {seed_celex} missing from discovery results")
    return documents


def discover_topics(client: httpx.Client, topics: Sequence[str]) -> list[DiscoveredDocument]:
    """Discover all topics, deduped by celex (first topic wins), wrapping parse errors."""
    by_celex: dict[str, DiscoveredDocument] = {}
    for topic in topics:
        try:
            documents = discover_topic(client, topic, SEEDS[topic])
        except (KeyError, json.JSONDecodeError) as exc:
            raise MalformedDiscoveryError(f"{topic}: malformed SPARQL response: {exc!r}") from exc
        for document in documents:
            by_celex.setdefault(document.celex, document)
    return list(by_celex.values())


def find_dropped_celexes(
    documents: Sequence[DiscoveredDocument], baseline_celexes: Iterable[str]
) -> list[str]:
    """Baseline celexes discovery no longer returns; losing an implausible share is an error.

    A truncated result set is indistinguishable from a mass repeal, so refuse to call it one.
    """
    discovered = {document.celex for document in documents}
    baseline = set(baseline_celexes)
    dropped = sorted(baseline - discovered)
    if len(dropped) >= MIN_SUSPICIOUS_DROPS and len(dropped) > MAX_DROP_RATIO * len(baseline):
        raise CorpusShrankError(
            f"discovery lost {len(dropped)} of {len(baseline)} documents: {', '.join(dropped)}"
        )
    return dropped
