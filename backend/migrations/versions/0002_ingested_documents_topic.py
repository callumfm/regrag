"""ingested_documents topic

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-03 18:27:46.304253

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("ingested_documents", sa.Column("topic", sa.String(), nullable=False))


def downgrade() -> None:
    op.drop_column("ingested_documents", "topic")
