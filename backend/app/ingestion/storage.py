"""Where the corpus lives on disk; one definition, used by the writer and the reader."""

from pathlib import Path


def raw_html_path(data_dir: Path, ref: str) -> Path:
    """The file the fetch stage writes a document's source HTML to."""
    return data_dir / f"{ref}.html"
