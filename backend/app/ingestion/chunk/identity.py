"""Content-addressed chunk identity: stable across runs, unique within a document."""

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator

from app.ingestion.chunk.models import Chunk


def content_hash(chunk: Chunk) -> str:
    """Hash of everything that makes a chunk what it is; topic is provenance, not identity."""
    payload = json.dumps(
        [
            chunk.ref,
            chunk.kind,
            chunk.article,
            chunk.annex,
            chunk.title,
            list(chunk.heading_path),
            chunk.paragraph,
            chunk.part,
            chunk.parts,
            chunk.text,
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def keyed(chunks: Iterable[Chunk]) -> Iterator[tuple[Chunk, str, int]]:
    """Pair each chunk with its hash and the occurrence disambiguating identical siblings."""
    seen: Counter[str] = Counter()
    for chunk in chunks:
        digest = content_hash(chunk)
        yield chunk, digest, seen[digest]
        seen[digest] += 1
