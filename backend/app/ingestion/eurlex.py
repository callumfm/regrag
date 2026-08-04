"""EUR-Lex resolver: permanent base CELEX -> currently fetchable consolidated HTML URL."""

import re

import httpx

from app.ingestion.registry import DocumentSpec, Resolution

ALL_URL = "https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:{ref}"
HTML_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{ref}"
MISSING_MARKER = "The requested document does not exist."


class ResolutionError(Exception):
    """No fetchable HTML could be found for a registry entry."""


def latest_consolidated_ref(all_page_html: str, ref: str) -> str | None:
    """Consolidated CELEX ids look like '0' + ref[1:] + '-YYYYMMDD'; pick the newest."""
    dates = re.findall(rf"0{re.escape(ref[1:])}-(\d{{8}})", all_page_html)
    if not dates:
        return None
    return f"0{ref[1:]}-{max(dates)}"


def is_missing_document(html: str) -> bool:
    return MISSING_MARKER in html


def resolve(client: httpx.Client, spec: DocumentSpec) -> Resolution:
    """Return a verified HTML URL for the latest consolidated version, else the original act."""
    all_page = client.get(ALL_URL.format(ref=spec.ref))
    all_page.raise_for_status()
    consolidated = latest_consolidated_ref(all_page.text, spec.ref)
    candidates = [consolidated, spec.ref] if consolidated else [spec.ref]
    for candidate in candidates:
        url = HTML_URL.format(ref=candidate)
        response = client.get(url)
        if response.status_code == httpx.codes.NOT_FOUND or (
            response.is_success and is_missing_document(response.text)
        ):
            continue
        response.raise_for_status()
        return Resolution(resolved_ref=candidate, url=url)
    raise ResolutionError(f"{spec.name}: no fetchable HTML for {spec.ref}, tried {candidates}")
