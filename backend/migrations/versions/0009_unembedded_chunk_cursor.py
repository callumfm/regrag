"""unembedded chunk cursor

Revision ID: 0009
Revises: 0008
Create Date: 2026-08-11 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0009"
down_revision: str | Sequence[str] | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "ix_document_chunks_unembedded_cursor"


def upgrade() -> None:
    """Index the embed sweep's keyset cursor, so each page is a forward scan not a fresh sort.

    Partial on the same predicate the sweep filters by: the index holds only chunks still
    awaiting a vector, so it shrinks to nothing as a run completes.
    """
    op.create_index(
        INDEX_NAME,
        "document_chunks",
        ["celex", "id"],
        postgresql_where=sa.text("embedding IS NULL"),
    )


def downgrade() -> None:
    op.drop_index(INDEX_NAME, table_name="document_chunks")
