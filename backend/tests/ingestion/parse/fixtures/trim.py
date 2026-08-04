"""Dev-time fixture trimmer; regenerates the parse fixtures from data/raw.

Run from backend/: PYTHONPATH=. uv run python tests/ingestion/parse/fixtures/trim.py
"""

from pathlib import Path

from selectolax.parser import HTMLParser

from app.core.config import config

FIXTURES = Path(__file__).parent
KEEP: dict[str, tuple[str, ...]] = {
    "32023R1805": ("art_4", "art_5", "anx_II"),
    "32015R0757": ("art_3", "art_4", "art_11a", "anx_I"),
}
STUB = "data:image/jpg;base64,STUB"


def trim(ref: str, ids: tuple[str, ...]) -> str:
    """Keep only the named subdivisions, with base64 image payloads stubbed out."""
    tree = HTMLParser((config.RAW_DATA_DIR / f"{ref}.html").read_text(encoding="utf-8"))
    for image in tree.css("img"):
        if (image.attributes.get("src") or "").startswith("data:"):
            image.attrs["src"] = STUB
    kept = []
    for node_id in ids:
        node = tree.css_first(f'div[id="{node_id}"]')
        if node is None:
            raise SystemExit(f"{ref}: no div with id {node_id}")
        kept.append(node.html or "")
    return "<html><body>\n" + "\n".join(kept) + "\n</body></html>\n"


def main() -> None:
    for ref, ids in KEEP.items():
        html = trim(ref, ids)
        (FIXTURES / f"{ref}.html").write_text(html, encoding="utf-8")
        print(f"{ref}: {len(html) // 1024} KB")


if __name__ == "__main__":
    main()
