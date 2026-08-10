"""Dev-time fixture trimmer; regenerates the parse fixtures from the stored corpus.

Run from backend/: PYTHONPATH=. uv run python tests/ingestion/parse/fixtures/trim.py
"""

import asyncio
from pathlib import Path

from selectolax.parser import HTMLParser, Node

from app.core.db.session import get_session
from app.core.storage import StorageError, get_object_store
from app.ingestion.fetch.service import get_corpus_docs
from app.ingestion.storage import read_document

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


async def stored_html() -> dict[str, str]:
    """The documents KEEP names, read back out of object storage."""
    async with get_session(auto_commit=False) as session:
        documents = await get_corpus_docs(session)
    store = get_object_store()
    wanted = [doc for doc in documents if doc.celex in KEEP]
    try:
        return {doc.celex: read_document(store, doc).decode("utf-8") for doc in wanted}
    except StorageError as exc:
        raise SystemExit(f"{exc}; run `uv run ingest` first") from exc


def trim(celex: str, html: str, ids: tuple[str, ...]) -> str:
    """Keep only the named subdivisions, with base64 image payloads stubbed out."""
    tree = HTMLParser(html)
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
    stored = asyncio.run(stored_html())
    for celex, ids in KEEP.items():
        if celex not in stored:
            raise SystemExit(f"{celex} is not in the stored corpus; run `uv run ingest` first")
        html = trim(celex, stored[celex], ids)
        (FIXTURES / f"{celex}.html").write_text(html, encoding="utf-8")
        print(f"{celex}: {len(html) // 1024} KB")


if __name__ == "__main__":
    main()
