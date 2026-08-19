"""chat request nodes and error

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

INDEX_NAME = "ix_chat_request_nodes_chat_request_id"


def upgrade() -> None:
    """A row per node a request ran through, with its time and tokens, and an error column
    on the request; the per-stage timings on the request go."""
    op.create_table(
        "chat_request_nodes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("chat_request_id", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("node", sa.String(), nullable=False),
        sa.Column("ms", sa.Integer(), nullable=False),
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
        sa.ForeignKeyConstraint(["chat_request_id"], ["chat_requests.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(INDEX_NAME, "chat_request_nodes", ["chat_request_id"])
    op.add_column("chat_requests", sa.Column("error", sa.String(), nullable=True))
    op.drop_column("chat_requests", "retrieve_ms")
    op.drop_column("chat_requests", "ttft_ms")


def downgrade() -> None:
    op.add_column("chat_requests", sa.Column("ttft_ms", sa.Integer(), nullable=True))
    op.add_column("chat_requests", sa.Column("retrieve_ms", sa.Integer(), nullable=True))
    op.drop_column("chat_requests", "error")
    op.drop_index(INDEX_NAME, table_name="chat_request_nodes")
    op.drop_table("chat_request_nodes")
