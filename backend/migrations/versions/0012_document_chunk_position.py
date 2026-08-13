"""document chunk position

Revision ID: 0012
Revises: 0011
Create Date: 2026-08-13 11:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0012"
down_revision: str | Sequence[str] | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Record a chunk's ordinal in its document, the only sort key an annex has once re-ingest
    stops issuing ids in document order; rows from before it read as 0 until the next ingest."""
    op.add_column(
        "document_chunks",
        sa.Column("position", sa.Integer(), nullable=False, server_default="0"),
    )
    op.alter_column("document_chunks", "position", server_default=None)


def downgrade() -> None:
    op.drop_column("document_chunks", "position")
