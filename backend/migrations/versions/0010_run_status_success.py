"""run status success

Revision ID: 0010
Revises: 0009
Create Date: 2026-08-11 13:45:22.831774

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0010"
down_revision: str | Sequence[str] | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

STATUS_TYPE = sa.Enum(
    "running", "success", "failed", "aborted", name="ingestrunstatus", native_enum=False
)


def upgrade() -> None:
    """Rename the clean-run status; the column narrows because its longest value shrank."""
    op.execute("UPDATE ingest_runs SET status='success' WHERE status='completed'")
    op.alter_column(
        "ingest_runs",
        "status",
        existing_type=sa.VARCHAR(length=9),
        type_=STATUS_TYPE,
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "ingest_runs",
        "status",
        existing_type=STATUS_TYPE,
        type_=sa.VARCHAR(length=9),
        existing_nullable=False,
    )
    op.execute("UPDATE ingest_runs SET status='completed' WHERE status='success'")
