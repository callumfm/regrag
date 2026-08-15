import random
from collections.abc import Callable

import pytest
from litellm.types.rerank import RerankResponse
from sqlalchemy import Select, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import EMBED_DIMENSIONS, config
from app.core.llm import LLMError
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.schemas import IngestRun
from app.retrieval.models import SearchFilters, SearchRequest, SearchResult
from app.retrieval.rerank import rerank_results
from app.retrieval.search import (
    _text_candidates,
    _tune_hnsw_walk,
    _vector_candidates,
    hybrid_search,
    search,
)
from tests.retrieval.conftest import toy_embed

pytestmark = pytest.mark.anyio

NO_FILTERS = SearchFilters()
SUPERSEDED = 300
"""Dead vectors hugging the query: enough to exhaust hnsw.ef_search, whose default is 40."""
SURVIVORS = 50
"""Live rows a ring farther out, far more than the limit so the assertion is not about recall."""
WANTED = 5
HUGGING, BEHIND = 0.01, 0.1
SEED = 8
"""Fixed, so the graph this test builds and walks is the same one on every run."""


def random_query(rng: random.Random) -> list[float]:
    """A signed query vector, which the corpus's non-negative vectors all sit far from."""
    return [rng.uniform(-1.0, 1.0) for _ in range(EMBED_DIMENSIONS)]


def nudge(rng: random.Random, query: list[float], scale: float) -> list[float]:
    """The query pushed a set distance in a fresh random direction."""
    return [value + scale * (rng.random() - 0.5) for value in query]


async def leg_ids(session: AsyncSession, leg: Select) -> list[int]:
    """The ids one leg returns in its own rank order, run as the fused query would run it."""
    await _tune_hnsw_walk(session, config.SEARCH_CANDIDATES)
    return [row.id for row in await session.execute(leg)]


async def load_pgvector(session: AsyncSession) -> None:
    """Force the module to load, since its settings are unvalidated placeholders until it does."""
    await session.execute(text("SELECT '[1,2,3]'::vector <-> '[3,2,1]'::vector"))


async def citations_for(session: AsyncSession, chunk_ids: list[int]) -> list[str]:
    """The citation of each id, in the order the ids were given."""
    stmt = select(DocumentChunk.id, DocumentChunk.citation).where(DocumentChunk.id.in_(chunk_ids))
    citations = {chunk_id: citation for chunk_id, citation in (await session.execute(stmt)).all()}
    return [citations[chunk_id] for chunk_id in chunk_ids]


# The HNSW walk


async def test_the_walk_runs_in_the_order_fusion_ranks_on(db_session: AsyncSession) -> None:
    await _tune_hnsw_walk(db_session, config.SEARCH_CANDIDATES)

    assert await db_session.scalar(text("SHOW hnsw.iterative_scan")) == "strict_order"


async def test_the_walk_is_sized_from_the_candidate_pool(db_session: AsyncSession) -> None:
    await _tune_hnsw_walk(db_session, 50)

    assert await db_session.scalar(text("SHOW hnsw.ef_search")) == "200"


async def test_a_candidate_pool_past_the_ceiling_is_clamped_rather_than_refused(
    db_session: AsyncSession,
) -> None:
    """pgvector rejects an ef_search above 1000, which a tuning knob must not be able to trigger."""
    await load_pgvector(db_session)

    await _tune_hnsw_walk(db_session, 1000)

    assert await db_session.scalar(text("SHOW hnsw.ef_search")) == "1000"


async def test_hybrid_search_sizes_the_walk_from_the_candidates_it_is_given(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    """The knob and the tuning drifted apart while ef_search was frozen at import."""
    await hybrid_search(
        db_session,
        toy_embed("energy"),
        "energy",
        NO_FILTERS,
        candidates=100,
        rrf_k=config.RRF_K,
        limit=5,
    )

    assert await db_session.scalar(text("SHOW hnsw.ef_search")) == "400"


# The vector leg


async def test_the_vector_leg_ranks_the_chunk_whose_text_was_embedded_first(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    target = corpus[0]

    found = await leg_ids(db_session, _vector_candidates(toy_embed(target.text), NO_FILTERS, 5))

    assert found[0] == target.id


async def test_the_vector_leg_respects_its_limit(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await leg_ids(db_session, _vector_candidates(toy_embed("energy"), NO_FILTERS, 3))

    assert len(found) == 3


async def test_the_vector_leg_meets_its_limit_past_dead_tuples(
    db_session: AsyncSession,
    ingest_run: IngestRun,
    make_chunk_row: Callable[..., DocumentChunk],
) -> None:
    """Re-ingesting a document leaves old vectors dead in the graph, nearer than its new ones."""
    rng = random.Random(SEED)
    query = random_query(rng)
    vectors = [nudge(rng, query, HUGGING) for _ in range(SUPERSEDED)]
    vectors += [nudge(rng, query, BEHIND) for _ in range(SURVIVORS)]
    rows = [
        make_chunk_row(ingest_run, content_hash=f"{index:064d}", embedding=vector)
        for index, vector in enumerate(vectors)
    ]
    db_session.add_all(rows)
    await db_session.flush()
    for row in rows[:SUPERSEDED]:
        await db_session.delete(row)
    await db_session.flush()
    await db_session.execute(text("SET LOCAL enable_seqscan = off"))

    found = await leg_ids(db_session, _vector_candidates(query, NO_FILTERS, WANTED))

    assert len(found) == WANTED


async def test_the_vector_leg_skips_chunks_with_no_vector(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    unembedded = corpus[0]
    await db_session.execute(
        update(DocumentChunk).where(DocumentChunk.id == unembedded.id).values(embedding=None)
    )

    found = await leg_ids(db_session, _vector_candidates(toy_embed("energy"), NO_FILTERS, 50))

    assert unembedded.id not in found


async def test_a_topic_filter_narrows_the_vector_leg_to_one_act(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await leg_ids(
        db_session, _vector_candidates(toy_embed("energy"), SearchFilters(topic="mrv"), 50)
    )

    stmt = select(DocumentChunk.topic).where(DocumentChunk.id.in_(found))
    assert set(await db_session.scalars(stmt)) == {"mrv"}


# The text leg


async def test_the_text_leg_finds_an_article_by_its_citation(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await leg_ids(db_session, _text_candidates("Article 11a", NO_FILTERS, 4))

    assert found
    assert all(
        citation.startswith("Article 11a") for citation in await citations_for(db_session, found)
    )


async def test_the_text_leg_does_not_confuse_article_11_with_article_11a(
    db_session: AsyncSession,
    corpus: list[DocumentChunk],
    ingest_run: IngestRun,
    make_chunk_row: Callable[..., DocumentChunk],
) -> None:
    article_11 = make_chunk_row(
        ingest_run,
        content_hash="c" * 64,
        article="11",
        paragraph="1",
        citation="Article 11(1)",
        text="Companies shall submit their monitoring plan to the verifier before the period.",
    )
    db_session.add(article_11)
    await db_session.flush()

    found_11a = await leg_ids(db_session, _text_candidates("Article 11a", NO_FILTERS, 4))
    assert article_11.id not in found_11a
    assert all(
        citation.startswith("Article 11a")
        for citation in await citations_for(db_session, found_11a)
    )

    found_11 = await leg_ids(db_session, _text_candidates("Article 11", NO_FILTERS, 5))
    assert found_11[0] == article_11.id


async def test_a_query_of_only_stopwords_matches_nothing(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    assert await leg_ids(db_session, _text_candidates("the of and", NO_FILTERS, 50)) == []


async def test_a_celex_filter_narrows_the_text_leg_to_one_act(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await leg_ids(
        db_session, _text_candidates("energy", SearchFilters(celex="32023R1805"), 50)
    )

    stmt = select(DocumentChunk.celex).where(DocumentChunk.id.in_(found))
    assert set(await db_session.scalars(stmt)) == {"32023R1805"}


# The whole search: embed, fuse, rerank


async def test_an_article_query_returns_that_article(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await search(db_session, SearchRequest(query="Article 11a"))

    assert any(result.citation.startswith("Article 11a") for result in found)


async def test_a_topic_filter_excludes_the_other_act(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    request = SearchRequest(query="greenhouse gas emissions", filters=SearchFilters(topic="mrv"))

    found = await search(db_session, request)

    assert {result.topic for result in found} == {"mrv"}


async def test_a_paraphrase_reaches_the_article_that_defines_the_term(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    request = SearchRequest(query="monitoring plan submitted to the verifier", limit=5)

    found = await search(db_session, request)

    assert any(result.citation.startswith("Article 4") for result in found)


async def test_results_come_back_in_fused_order(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await search(db_session, SearchRequest(query="greenhouse gas emissions"))

    assert [result.score for result in found] == sorted(
        (result.score for result in found), reverse=True
    )
    assert any(result.vector_rank is not None and result.text_rank is not None for result in found)


async def test_a_result_records_which_legs_found_it(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await search(db_session, SearchRequest(query="Article 11a"))

    assert all(result.vector_rank is not None or result.text_rank is not None for result in found)
    assert any(result.text_rank is not None for result in found)


async def test_the_limit_caps_the_results(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    assert len(await search(db_session, SearchRequest(query="energy", limit=3))) == 3


async def test_a_query_matching_no_keywords_still_returns_vector_hits(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await search(db_session, SearchRequest(query="the of and"))

    assert found
    assert all(result.text_rank is None for result in found)


async def test_a_search_over_an_empty_corpus_returns_nothing(empty_session: AsyncSession) -> None:
    assert await search(empty_session, SearchRequest(query="verification period")) == ()


async def test_the_candidate_pool_is_read_from_config_per_call(
    db_session: AsyncSession, corpus: list[DocumentChunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "SEARCH_CANDIDATES", 1)

    found = await search(db_session, SearchRequest(query="greenhouse gas emissions", limit=10))

    assert len(found) == 2


async def test_a_provider_failure_surfaces_rather_than_returning_nothing(
    db_session: AsyncSession, corpus: list[DocumentChunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _fail(texts: list[str], **kwargs: object) -> list[list[float]]:
        raise LLMError("embedding call failed")

    monkeypatch.setattr("app.retrieval.search.embed", _fail)

    with pytest.raises(LLMError):
        await search(db_session, SearchRequest(query="verification period"))


async def test_rerank_receives_a_pool_wider_than_the_limit(
    db_session: AsyncSession, corpus: list[DocumentChunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    pools = []

    async def _capture(query, results, *, limit):
        pools.append(len(results))
        return results[:limit]

    monkeypatch.setattr("app.retrieval.search.rerank_results", _capture)

    found = await search(db_session, SearchRequest(query="greenhouse gas emissions", limit=3))

    assert len(found) == 3
    assert 3 < pools[0] <= config.RERANK_POOL


async def test_the_reranked_order_is_what_the_caller_receives(
    db_session: AsyncSession, corpus: list[DocumentChunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _reverse(query, results, *, limit):
        return tuple(reversed(results))[:limit]

    monkeypatch.setattr("app.retrieval.search.rerank_results", _reverse)

    found = await search(db_session, SearchRequest(query="greenhouse gas emissions", limit=5))

    assert [result.score for result in found] == sorted(result.score for result in found)


async def test_disabling_rerank_skips_the_step_and_keeps_the_narrow_pool(
    db_session: AsyncSession, corpus: list[DocumentChunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    async def _record(query, results, *, limit):
        calls.append(query)
        return results[:limit]

    monkeypatch.setattr("app.retrieval.search.rerank_results", _record)
    monkeypatch.setattr(config, "RERANK_ENABLED", False)

    found = await search(db_session, SearchRequest(query="greenhouse gas emissions", limit=3))

    assert len(found) == 3
    assert calls == []


async def test_the_real_rerank_path_reorders_the_pool(
    db_session: AsyncSession, corpus: list[DocumentChunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[tuple[SearchResult, ...]] = []

    async def spying_rerank_results(query, results, *, limit):
        captured.append(results)
        return await rerank_results(query, results, limit=limit)

    async def fake_arerank(**kwargs):
        count = len(kwargs["documents"])
        return RerankResponse(
            results=[{"index": i, "relevance_score": float(i)} for i in reversed(range(count))]
        )

    monkeypatch.setattr("app.retrieval.search.rerank_results", spying_rerank_results)
    monkeypatch.setattr("app.retrieval.rerank.litellm.arerank", fake_arerank)

    found = await search(db_session, SearchRequest(query="greenhouse gas emissions", limit=5))

    assert found == tuple(reversed(captured[0]))[:5]


async def test_a_limit_above_the_rerank_pool_is_honoured(
    db_session: AsyncSession, corpus: list[DocumentChunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    pools = []

    async def _capture(query, results, *, limit):
        pools.append(len(results))
        return results[:limit]

    monkeypatch.setattr("app.retrieval.search.rerank_results", _capture)
    request = SearchRequest(query="greenhouse gas emissions", limit=config.RERANK_POOL + 10)

    await search(db_session, request)

    assert pools[0] > config.RERANK_POOL


async def test_a_search_result_carries_the_references_its_chunk_cites(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    """The fused select projects the links too, so an answer can follow on from what it found."""
    request = SearchRequest(query="monitoring plan submitted to the verifier", limit=10)

    found = await search(db_session, request)

    cited = [reference for result in found for reference in result.references]

    assert cited
    assert all(reference.raw for reference in cited)
