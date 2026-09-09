"""chat request thread and answer

Revision ID: 0020
Revises: 0019
Create Date: 2026-09-09 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0020"
down_revision: str | Sequence[str] | None = "0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "ix_chat_requests_thread_id"


def upgrade() -> None:
    """The thread a request belongs to and the answer it gave: the row becomes the thread's
    memory, read back before a follow-up runs."""
    op.add_column("chat_requests", sa.Column("thread_id", sa.Uuid(), nullable=True))
    op.add_column("chat_requests", sa.Column("answer", sa.String(), nullable=True))
    op.create_index(INDEX_NAME, "chat_requests", ["thread_id"])


def downgrade() -> None:
    op.drop_index(INDEX_NAME, table_name="chat_requests")
    op.drop_column("chat_requests", "answer")
    op.drop_column("chat_requests", "thread_id")
