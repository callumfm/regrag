"""chat request nodes

Revision ID: 0018
Revises: 0017
Create Date: 2026-08-19 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0018"
down_revision: str | Sequence[str] | None = "0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """The graph path a request took, which outcome alone cannot carry once a tool loop
    makes the path variable. Existing rows take an empty path: it was never recorded."""
    op.add_column(
        "chat_requests",
        sa.Column("nodes", sa.ARRAY(sa.String()), nullable=False, server_default="{}"),
    )


def downgrade() -> None:
    op.drop_column("chat_requests", "nodes")
