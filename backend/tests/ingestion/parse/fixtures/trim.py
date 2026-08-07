"""Dev-time fixture trimmer; regenerates the parse fixtures from data/raw.

Run from backend/: PYTHONPATH=. uv run python tests/ingestion/parse/fixtures/trim.py
"""

from pathlib import Path

from selectolax.parser import HTMLParser, Node

from app.core.config import config
from app.ingestion.fetch.schemas import RawDocument

FIXTURES = Path(__file__).parent
KEEP: dict[str, tuple[str, ...]] = {
    "32023R1805": ("art_4", "art_5", "anx_II"),
    "32015R0757": ("art_3", "art_4", "art_11a", "anx_I"),
}
STUB = "data:image/jpg;base64,STUB"
SECTION = "p.title-gr-seq-level-2"
MAX_ROWS = 5


def shrink(node: Node) -> None:
    """Drop the bulk the parser cannot tell apart: long table bodies, repeat sections."""
    for table in node.css("table"):
        for row in table.css("tr")[MAX_ROWS:]:
            row.decompose()
    children = list(node.iter())
    sections = [i for i, child in enumerate(children) if child.css_matches(SECTION)]
    if len(sections) > 1:
        for child in children[sections[1] :]:
            child.decompose()


def trim(celex: str, ids: tuple[str, ...]) -> str:
    """Keep only the named subdivisions, with base64 image payloads stubbed out."""
    tree = HTMLParser(
        (config.RAW_DATA_DIR / RawDocument.filename(celex)).read_text(encoding="utf-8")
    )
    for image in tree.css("img"):
        if (image.attributes.get("src") or "").startswith("data:"):
            image.attrs["src"] = STUB
    kept = []
    for node_id in ids:
        node = tree.css_first(f'div[id="{node_id}"]')
        if node is None:
            raise SystemExit(f"{celex}: no div with id {node_id}")
        shrink(node)
        kept.append(node.html or "")
    body = "<html><body>\n" + "\n".join(kept) + "\n</body></html>\n"
    return "".join(f"{line.rstrip()}\n" for line in body.splitlines())


def main() -> None:
    for celex, ids in KEEP.items():
        html = trim(celex, ids)
        (FIXTURES / f"{celex}.html").write_text(html, encoding="utf-8")
        print(f"{celex}: {len(html) // 1024} KB")


if __name__ == "__main__":
    main()
