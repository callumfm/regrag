# Retrieval

This module is the read side of the corpus: given a question, find the chunks of law that answer it.

```bash
uv run retrieve "how is energy at berth reported"     # print hits with their scores
uv run retrieve "..." --celex 32023R1805 --limit 5    # one act, fewer results
```

## Hybrid search

A hybrid search that combines keyword search with vector search, using Reciprocal Rank Fusion to merge the two rankings into one. Keyword search catches exact terms, vector search catches paraphrases, and fusion means a chunk found by either leg still surfaces.

## Reranking

A cross-encoder reads the question and each chunk together and scores how relevant they actually are. It is more accurate than search, but too slow to run over the whole corpus, so it only rescores the top results search returned. If the call fails we fall back to the search order.

## The gate

To minimise hallucination and avoid the cost of answer synthesis, we require a minimum relevance score before calling the model.

## Section expansion

Search ranks chunks, but the section is the unit that answers — "the limit referred to in paragraph 1" means nothing on its own. So each hit can be widened out to the article it was cut from. Off by default, since widening eats context that authored references would otherwise fill.

## Following references

Chunks store the citations found in their text as structured fields, so a reference like "Article 5" can be looked up directly rather than searched for. Used to pull in cited law, and by the evals to fingerprint what a case was authored against.
