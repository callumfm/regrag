"""Dev-time fixture recorder; the only code here that touches the network.

Run from backend/: PYTHONPATH=. uv run python tests/ingestion/fetch/fixtures/record.py
"""

import asyncio
import json
from pathlib import Path

from app.core.config import config
from app.core.http import http_client
from app.ingestion.discover.select import select_topic_documents
from app.ingestion.discover.sparql import extract_acts, run_topic_query
from app.ingestion.fetch.download import download_fetchable_version

FIXTURES = Path(__file__).parent


async def record() -> None:
    expected: dict[str, str] = {}
    missing: set[str] = set()
    async with http_client(timeout=120, delays=config.CRAWL_DELAYS) as client:
        for topic, seed in config.SEEDS.items():
            rows = await run_topic_query(client, seed)
            payload = {"results": {"bindings": rows}}
            (FIXTURES / f"sparql-{topic}.json").write_text(json.dumps(payload, indent=2) + "\n")
            for spec in select_topic_documents(topic, extract_acts(rows)):
                resolution, _ = await download_fetchable_version(client, spec)
                if spec.candidate_celex and resolution.resolved_celex == spec.celex:
                    missing.add(spec.candidate_celex)
                expected[f"{topic}:{spec.celex}"] = resolution.resolved_celex
    for key, value in sorted(expected.items()):
        print(f'    "{key}": "{value}",')
    print(f"MISSING_HTML = {sorted(missing)!r}")


def main() -> None:
    asyncio.run(record())


if __name__ == "__main__":
    main()
