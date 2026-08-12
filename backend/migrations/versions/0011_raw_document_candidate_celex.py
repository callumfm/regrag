"""raw document candidate celex

Revision ID: 0011
Revises: 0010
Create Date: 2026-08-11 17:40:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0011"
down_revision: str | Sequence[str] | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Record the version discovery asked for, so reuse can tell a new consolidation from a
    denied one; rows from before it was recorded read as null and are fetched once more."""
    op.add_column("raw_documents", sa.Column("candidate_celex", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("raw_documents", "candidate_celex")
