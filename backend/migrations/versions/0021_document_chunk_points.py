"""document chunk points

Revision ID: 0021
Revises: 0020
Create Date: 2026-09-10 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0021"
down_revision: str | Sequence[str] | None = "0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """The points a chunk's text opens lines with, so a citation by point is a lookup rather
    than a regex over the text. Empty until the next ingest run: the column is metadata, so
    every row's metadata hash drifts and the run fills it without re-embedding anything."""
    op.add_column(
        "document_chunks",
        sa.Column("points", postgresql.ARRAY(sa.String()), nullable=False, server_default="{}"),
    )


def downgrade() -> None:
    op.drop_column("document_chunks", "points")
