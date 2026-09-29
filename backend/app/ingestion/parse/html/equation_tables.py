"""Equations EUR-Lex lays out as a one-row table, `label =` beside its expression, read as
one formula so the label and the maths around a drawn formula stay inside its delimiters."""

import re

from selectolax.parser import Node

from app.ingestion.parse.formex.latex import text_to_latex
from app.ingestion.parse.html.text import clean_text

FORMULA_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
SUBSCRIPT_CLASSES = {"oj-sub", "subscript"}
TEXT_NODE = "-text"


def _cell_latex(node: Node) -> str:
    """A cell as LaTeX: its text set upright, its subscripts as `_{}`, its formulas as written."""
    parts: list[str] = []
    for child in node.iter(include_text=True):
        if child.tag == TEXT_NODE:
            for index, piece in enumerate(FORMULA_RE.split(child.text_content or "")):
                parts.append(piece if index % 2 else text_to_latex(piece))
        elif SUBSCRIPT_CLASSES & set((child.attributes.get("class") or "").split()):
            parts.append(f"_{{{_cell_latex(child)}}}")
        else:
            parts.append(_cell_latex(child))
    return "".join(parts)


def equation_text(table: Node) -> str | None:
    """The whole equation as one `$$latex$$` where the table is a single `label =` row beside
    an expression holding a formula, None for any other table."""
    rows = table.css("tr")
    if len(rows) != 1:
        return None
    cells = rows[0].css("td, th")
    if not (
        len(cells) == 2
        and clean_text(cells[0].text()).endswith("=")
        and FORMULA_RE.search(cells[1].text())
    ):
        return None
    return f"$${' '.join(_cell_latex(rows[0]).split())}$$"
