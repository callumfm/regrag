"""raw document topic run

Revision ID: 0016
Revises: 0015
Create Date: 2026-08-14 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0016"
down_revision: str | Sequence[str] | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "ix_raw_documents_topic_run"


def upgrade() -> None:
    """Index the topic grouping behind the standing corpus, so its aggregate reads the index only.

    The corpus query takes the highest successful run id per topic; leading on topic lets that
    max come off the index rather than a scan of every row the table has ever accumulated.
    """
    op.create_index(INDEX_NAME, "raw_documents", ["topic", "ingest_run_id"])


def downgrade() -> None:
    op.drop_index(INDEX_NAME, table_name="raw_documents")
