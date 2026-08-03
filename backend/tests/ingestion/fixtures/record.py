"""Dev-time fixture recorder; the only code here that touches the network.

Run from backend/: uv run python tests/ingestion/fixtures/record.py
"""

import re
import time
from pathlib import Path

import httpx

from app.ingestion.eurlex import (
    ALL_URL,
    HEADERS,
    HTML_URL,
    is_missing_document,
    latest_consolidated_ref,
)
from app.ingestion.registry import CORPUS

FIXTURES = Path(__file__).parent


def fetch(client: httpx.Client, url: str) -> str:
    time.sleep(1)
    response = client.get(url, headers=HEADERS, follow_redirects=True)
    if response.status_code != 404:
        response.raise_for_status()
    return response.text


def trim_all_page(html: str, ref: str) -> str:
    pattern = re.compile(rf".*0{re.escape(ref[1:])}-\d{{8}}.*")
    kept = [line for line in html.splitlines() if pattern.match(line)]
    return "<html><body>\n" + "\n".join(kept) + "\n</body></html>\n"


def trim_page(html: str) -> str:
    return "\n".join(html.splitlines()[:200]) + "\n"


def main() -> None:
    missing_refs: list[str] = []
    with httpx.Client(timeout=30) as client:
        for spec in CORPUS:
            all_html = fetch(client, ALL_URL.format(ref=spec.ref))
            (FIXTURES / f"{spec.ref}-all.html").write_text(trim_all_page(all_html, spec.ref))
            consolidated = latest_consolidated_ref(all_html, spec.ref)
            resolved = spec.ref
            if consolidated:
                doc_html = fetch(client, HTML_URL.format(ref=consolidated))
                if is_missing_document(doc_html):
                    missing_refs.append(consolidated)
                    if not (FIXTURES / "missing.html").exists():
                        (FIXTURES / "missing.html").write_text(trim_page(doc_html))
                else:
                    resolved = consolidated
            if not (FIXTURES / "doc.html").exists() and resolved == spec.ref:
                doc_html = fetch(client, HTML_URL.format(ref=spec.ref))
                (FIXTURES / "doc.html").write_text(trim_page(doc_html))
            print(f'    "{spec.name}": "{resolved}",')
    print(f"MISSING_HTML = {set(missing_refs)!r}")


if __name__ == "__main__":
    main()
