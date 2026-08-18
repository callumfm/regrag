"""chat runs

Revision ID: 0017
Revises: 0016
Create Date: 2026-08-18 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0017"
down_revision: str | Sequence[str] | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "ix_chat_runs_created_at"


def upgrade() -> None:
    """One row per streamed answer: its stage timings, source count and token usage.

    Indexed on created_at because that is what a spend cap sums over: tokens in the
    last window, read off the index rather than every run the table has accumulated.
    """
    op.create_table(
        "chat_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("request_id", sa.String(), nullable=True),
        sa.Column(
            "outcome",
            sa.Enum("done", "error", "aborted", name="chatoutcome", native_enum=False),
            nullable=False,
        ),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("retrieve_ms", sa.Integer(), nullable=True),
        sa.Column("ttft_ms", sa.Integer(), nullable=True),
        sa.Column("total_ms", sa.Integer(), nullable=False),
        sa.Column("sources", sa.Integer(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(INDEX_NAME, "chat_runs", ["created_at"])


def downgrade() -> None:
    op.drop_index(INDEX_NAME, table_name="chat_runs")
    op.drop_table("chat_runs")
