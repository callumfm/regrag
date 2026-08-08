"""Dev-time fixture recorder; the only code here that touches the network.

Run from backend/: PYTHONPATH=. uv run python tests/ingestion/fetch/fixtures/record.py
"""

import json
import time
from pathlib import Path

from app.core.http import http_client
from app.ingestion.constants import SEEDS
from app.ingestion.fetch.discover import (
    SPARQL_ENDPOINT,
    collect_candidate_acts,
    select_topic_documents,
    topic_query,
)
from app.ingestion.fetch.resolve import resolve_version

FIXTURES = Path(__file__).parent


def main() -> None:
    expected: dict[str, str] = {}
    missing: set[str] = set()
    with http_client(timeout=120) as client:
        for topic, seed in SEEDS.items():
            response = client.get(
                SPARQL_ENDPOINT,
                params={"query": topic_query(seed), "format": "application/sparql-results+json"},
            )
            response.raise_for_status()
            payload = response.json()
            (FIXTURES / f"sparql-{topic}.json").write_text(json.dumps(payload, indent=2) + "\n")
            for spec in select_topic_documents(topic, collect_candidate_acts(payload)):
                time.sleep(1)
                resolution = resolve_version(client, spec)
                if spec.candidate_celex and resolution.resolved_celex == spec.celex:
                    missing.add(spec.candidate_celex)
                expected[f"{topic}:{spec.celex}"] = resolution.resolved_celex
    for key, value in sorted(expected.items()):
        print(f'    "{key}": "{value}",')
    print(f"MISSING_HTML = {sorted(missing)!r}")


if __name__ == "__main__":
    main()
