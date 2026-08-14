"""document chunk metadata hash

Revision ID: 0015
Revises: 0014
Create Date: 2026-08-13 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0015"
down_revision: str | Sequence[str] | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Fingerprint of the metadata columns, so drift detection compares one string per row.
    No backfill: NULL reads as drifted, so the next ingest run fills it in."""
    op.add_column("document_chunks", sa.Column("metadata_hash", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("document_chunks", "metadata_hash")
