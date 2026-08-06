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
- Take the subject of a call positionally and everything else as
  keyword-only: `session`/`record` up front, then `*`, then the data. Stage
  entry points, service functions and multi-argument constructors all follow
  this, so no call site depends on the order of interchangeable arguments.
- Package by capability, then by stage. Each capability (`ingestion`,
  `retrieval`, ...) owns its `enums.py`, `models.py`, `service.py`, the
  `schemas.py` of any table it owns outright (`ingestion` owns `ingest_runs`),
  and `router.py` where it has HTTP surface. A capability with pipeline stages
  gives each stage a sub-package holding its own `models.py` and a `stage.py`
  whose entry point is `<verb>_documents` — what the pipeline calls, returning
  that stage's delta, and the only public name in the module. Add a
  `_<verb>_document` beside it only where one document's work needs isolating,
  meaning I/O or per-document error capture; `fetch/` and `parse/` have one,
  `chunk/` does not, because chunking is pure and `chunk/chunker.py` already
  owns that name. Everything else in `stage.py` is a `_`-prefixed helper, so
  the seam the pipeline depends on is visible in the module itself rather than
  re-exported through `__init__.py`, which stays empty. Other modules in the
  sub-package hold how that stage does its work. A sub-package that owns a
  database table owns its `schemas.py` and `service.py` — `fetch/` owns
  `raw_documents`, `chunk/` owns `document_chunks`. Types describing the run
  as a whole, including every stage delta and the run report, live in the
  capability's `models.py`. `core/` holds infrastructure only, including its
  own operational endpoints such as health. New ORM schemas must be added to
  `core/db/registry.py`.
