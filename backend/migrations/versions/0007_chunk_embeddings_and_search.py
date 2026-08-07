"""chunk embeddings and search vector

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-06 13:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str | Sequence[str] | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DIMENSIONS = 1024
SEARCH_VECTOR_SQL = (
    """setweight(to_tsvector('english', citation || ' ' || coalesce(title, '')), 'A')"""
    """ || setweight(to_tsvector('english', "text"), 'B')"""
)


def upgrade() -> None:
    """Nullable embedding: the embed stage fills it, and an unembedded chunk is not an error."""
    op.add_column("document_chunks", sa.Column("embedding", Vector(DIMENSIONS), nullable=True))
    op.execute(
        "ALTER TABLE document_chunks ADD COLUMN search_vector tsvector "
        f"GENERATED ALWAYS AS ({SEARCH_VECTOR_SQL}) STORED"
    )
    op.create_index(
        "ix_document_chunks_search_vector",
        "document_chunks",
        ["search_vector"],
        postgresql_using="gin",
    )
    op.create_index(
        "ix_document_chunks_embedding",
        "document_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_embedding", table_name="document_chunks")
    op.drop_index("ix_document_chunks_search_vector", table_name="document_chunks")
    op.drop_column("document_chunks", "search_vector")
    op.drop_column("document_chunks", "embedding")
