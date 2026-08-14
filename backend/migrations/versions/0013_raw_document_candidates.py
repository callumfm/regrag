"""raw document candidates

Revision ID: 0013
Revises: 0012
Create Date: 2026-08-13 15:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0013"
down_revision: str | Sequence[str] | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Record every consolidation discovery offered, not just the newest, so a denied one falls
    back to the next consolidation rather than to the original act; the column it replaces held
    only the newest, which the array keeps as its first element."""
    op.add_column(
        "raw_documents",
        sa.Column("candidates", sa.ARRAY(sa.String()), nullable=False, server_default="{}"),
    )
    op.execute(
        "UPDATE raw_documents SET candidates = ARRAY[candidate_celex] "
        "WHERE candidate_celex IS NOT NULL"
    )
    op.alter_column("raw_documents", "candidates", server_default=None)
    op.drop_column("raw_documents", "candidate_celex")


def downgrade() -> None:
    op.add_column("raw_documents", sa.Column("candidate_celex", sa.String(), nullable=True))
    op.execute(
        "UPDATE raw_documents SET candidate_celex = candidates[1] "
        "WHERE array_length(candidates, 1) > 0"
    )
    op.drop_column("raw_documents", "candidates")
