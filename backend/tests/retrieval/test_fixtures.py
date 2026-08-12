import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.schemas import DocumentChunk
from tests.retrieval.conftest import toy_embed

pytestmark = pytest.mark.anyio


def test_the_toy_embedder_is_stable_across_calls() -> None:
    assert toy_embed("verification period") == toy_embed("verification period")


def test_the_toy_embedder_puts_overlapping_texts_nearer_than_unrelated_ones() -> None:
    anchor = toy_embed("the verification period for a company")
    near = toy_embed("the verification period")
    far = toy_embed("greenhouse gas intensity of energy")

    assert sum(a * b for a, b in zip(anchor, near, strict=True)) > sum(
        a * b for a, b in zip(anchor, far, strict=True)
    )


def test_an_empty_text_embeds_without_dividing_by_zero() -> None:
    assert set(toy_embed("")) == {0.0}


async def test_the_corpus_holds_both_acts_embedded_with_the_articles_the_suite_asserts_on(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    assert {row.celex for row in corpus} == {"32023R1805", "32015R0757"}
    assert all(row.embedding is not None for row in corpus)
    assert {"4", "5", "3", "11a"} <= {row.article for row in corpus if row.article}
