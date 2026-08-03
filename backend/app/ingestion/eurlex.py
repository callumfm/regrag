"""EUR-Lex resolver: permanent base CELEX -> currently fetchable consolidated HTML URL."""

import re

ALL_URL = "https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:{ref}"
HTML_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{ref}"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}
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
