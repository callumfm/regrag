"""ingest run result

Revision ID: 0008
Revises: 0007
Create Date: 2026-08-07 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: str | Sequence[str] | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Per-stage counts and failures, so a failed run says why without its container log.

    Nullable and not backfilled: a run that aborted before producing a result stores NULL,
    which is honest, where a zeroed dict would claim every stage did nothing.
    """
    op.add_column("ingest_runs", sa.Column("result", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("ingest_runs", "result")
