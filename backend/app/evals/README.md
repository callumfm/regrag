# Evals

This module focuses on the evaluation of the graph against a set of authored cases stored in `dataset/golden.json`.

```bash
uv run evals check    # report how far the dataset has drifted from the corpus
uv run evals stamp    # record what the cited text says now
uv run evals run      # score the dataset against the current graph
uv run evals tune     # rank retrieval settings against the dataset
```

Each command self-documents: `uv run evals <command> --help` prints what it does and every flag it takes.

## Dataset

A case is a question, what a correct answer must say, and the divisions of law that answer comes from.

```json
{
  "id": "fueleu-scope-third-country-voyage",
  "kind": "in_corpus",
  "question": "If my ship sails from Rotterdam to Singapore, how much of that voyage's energy counts under FuelEU?",
  "answer": "Half of it. For a voyage arriving at or departing from an EU port ...",
  "references": [{ "celex": "32023R1805", "article": "2", "content_hashes": ["cebdcb0a2d14", "..."] }]
}
```

A case is one of two kinds. An `in_corpus` case is scored on what retrieval found and what the answer cited, so it needs both an answer and references. An `out_of_corpus` case asks something the corpus does not cover and is scored on refusal alone.

## Drift

A problem the dataset faces is that laws get amended which may render some of our eval cases as stale. To track this drift, each reference also records a fingerprint of the text that was there when the case was written. `evals check` fingerprints it again and compares:

| Kind | What it means |
| ---- | ------------- |
| `unresolved` | The reference no longer resolves to any stored chunk. This eval case is now invalid |
| `stale` | The reference still exists but the cited text has changed since the case was stamped. This eval case needs to be updated |
| `unstamped` | Nothing recorded to compare against, so drift cannot be seen on this reference |

`evals run` reports the stale cases alongside the scores, but never fails on one: repairing a case means a human re-reading the new text. The check covers the whole dataset, so `--case` narrows what is scored, not what is checked for drift.

## Stamping

`evals stamp` records what the cited text says now. Run it on a newly authored case, or on a stale one you have just re-reviewed against the new text. The stamp asserts that the dataset has been reviewed. `--case` stamps a subset.

Stamps are excluded from `dataset_sha`, so re-stamping a case does not break comparability with runs that scored the same assertions before it.

## Running

Each case is driven through the same graph the `/chat` endpoint runs, and ends in the same `ChatState` a real request ends in, so a run is scored off what production records rather than off a parallel eval path. Cases run one at a time, so a per-case timing measures that case alone.

## Metrics

Scoring lives in `metrics.py`, each measure a plain function over the run's results.

| Metric | Scored over | What it measures |
| ------ | ----------- | ---------------- |
| `raw_hit_rate` | in-corpus | Search found at least one authored reference |
| `raw_recall` | in-corpus | Share of authored references search found |
| `expanded_hit_rate` | in-corpus | At least one authored reference reached the prompt |
| `expanded_recall` | in-corpus | Share of authored references that reached the prompt |
| `cited_references` | in-corpus | Share of authored references the answer cited |
| `markers_in_context` | answers citing anything | Share of `[n]` markers addressing a block the model was given |
| `gate_refusal_rate` | out-of-corpus | Share the pre-model gate refused |
| `false_refusals` | in-corpus | Cases the gate refused |
| `refused_a_found_reference` | in-corpus | Of those, the ones where search had already found a reference |

The raw and expanded pairs are worth reading together. Expansion widens each hit into its surrounding section, and against a fixed context budget that can push a reference *out*, so expanded recall is not guaranteed to be the higher of the two.

## Tuning

Tune measures a baseline, then re-measures once per candidate value, one factor at a time. It runs retrieval only so a sweep costs Postgres time rather than model spend. Rows are ranked by expanded recall, ties broken by the cheaper context.

The grid of parameters is stored in `tune/params.py`. Because some parameters, like `MIN_RERANKER_RELEVANCE` are dependent on the reranker being enabled, there is a `requires` field to ensure that this is applied even if the baseline doesn't have it.

## Caching

Embed and rerank calls replay from disk under `EVAL_CACHE_DIR`, keyed on each call's own request parameters. The first run over a case pays for them and every run after it does not. Deleting the directory invalidates the lot.

Synthesis is deliberately not cached as a run replaying its own answers would measure the cache rather than the model. Cached timings measure a disk read where a provider call would be, so `cached` is recorded on every run to keep the two from being compared. Use `--no-cache` for a latency baseline, or when a change alters what those calls *are*, such as a different embedding model.
