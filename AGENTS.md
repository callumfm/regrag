# AGENTS.md

- Do not preserve backward compatibility. Remove obsolete paths instead of
  adding compatibility layers, fallbacks, or migrations.
- Choose the simplest implementation that fully meets the current
  requirements. Avoid speculative abstractions, configuration, and
  indirection.
- Grow the system in layers. Start from the smallest version that works end
  to end, and add each new capability on top of a product that already
  works. Never trade a working product for unfinished complexity.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce overall
  complexity or improve reliability. Do not reimplement common
  functionality without a clear reason.
- Lean on the dependencies already in the project before writing your own
  implementation or adding packages. Do not assume a library lacks a
  capability without checking its documentation and types.
- Make architectural decisions for the long term. Do not accept a stopgap
  that only works for now and is meant to be replaced later.
- Do not write inline comments. Prefer self-describing code; where
  explanation is genuinely needed, use a docstring of 1-2 lines max.
- Package by capability, then by stage. Each capability (`ingestion`,
  `retrieval`, ...) owns its `enums.py`, `models.py`, `service.py`, and
  `router.py` where it has HTTP surface. A capability with pipeline stages
  gives each stage a sub-package holding its own `models.py` and a `stage.py`
  exposing `<verb>_document` and `<verb>_documents`; other modules in the
  sub-package hold how that stage does its work. A sub-package that owns a
  database table also owns its `schemas.py` and `service.py` — `fetch/` owns
  `raw_documents`, `chunk/` owns `document_chunks`. Types describing the run
  as a whole, including every stage delta and the run report, live in the
  capability's `models.py`. `core/` holds infrastructure only, including its
  own operational endpoints such as health. New ORM schemas must be added to
  `core/db/registry.py`.
