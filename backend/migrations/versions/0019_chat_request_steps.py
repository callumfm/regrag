"""chat request steps

Revision ID: 0019
Revises: 0018
Create Date: 2026-08-28 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0019"
down_revision: str | Sequence[str] | None = "0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _rename_belongings(old: str, new: str) -> None:
    """Postgres keeps the old table's name on everything a rename leaves behind — the index,
    sequence, primary key and foreign key — so each is renamed after the table itself."""
    op.execute(f"ALTER INDEX ix_{old}_chat_request_id RENAME TO ix_{new}_chat_request_id")
    op.execute(f"ALTER SEQUENCE {old}_id_seq RENAME TO {new}_id_seq")
    op.execute(f"ALTER TABLE {new} RENAME CONSTRAINT {old}_pkey TO {new}_pkey")
    op.execute(
        f"ALTER TABLE {new} RENAME CONSTRAINT "
        f"{old}_chat_request_id_fkey TO {new}_chat_request_id_fkey"
    )


def upgrade() -> None:
    """A path row is now a step: a graph node, or one tool call an assess round ran."""
    op.rename_table("chat_request_nodes", "chat_request_steps")
    op.alter_column("chat_request_steps", "node", new_column_name="step")
    _rename_belongings("chat_request_nodes", "chat_request_steps")


def downgrade() -> None:
    op.rename_table("chat_request_steps", "chat_request_nodes")
    op.alter_column("chat_request_nodes", "step", new_column_name="node")
    _rename_belongings("chat_request_steps", "chat_request_nodes")
