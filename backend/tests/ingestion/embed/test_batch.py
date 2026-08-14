"""How vectorless chunks are grouped into the units of work one provider call takes."""

from app.ingestion.embed.batch import ChunkToEmbed, _batch_by_document


def test_batches_split_at_the_provider_ceiling():
    produced = _batch_by_document([ChunkToEmbed(id=0, celex="32023R1805", text="")] * 129)

    assert [len(batch) for _, batch in produced] == [128, 1]


def test_batches_never_span_two_documents():
    chunks = [ChunkToEmbed(id=0, celex="32015R0757", text="")] * 3 + [
        ChunkToEmbed(id=0, celex="32023R1805", text="")
    ] * 2
    produced = _batch_by_document(chunks)

    assert [(celex, len(batch)) for celex, batch in produced] == [
        ("32015R0757", 3),
        ("32023R1805", 2),
    ]
