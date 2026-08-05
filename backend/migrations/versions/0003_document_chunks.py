"""document chunks

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-05 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("ref", sa.String(), nullable=False),
        sa.Column("topic", sa.String(), nullable=False),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("occurrence", sa.Integer(), nullable=False),
        sa.Column(
            "kind",
            sa.Enum(
                "article",
                "paragraph",
                "annex",
                "heading",
                "table",
                name="sectionkind",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("article", sa.String(), nullable=True),
        sa.Column("annex", sa.String(), nullable=True),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("paragraph", sa.String(), nullable=True),
        sa.Column("heading_path", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("part", sa.Integer(), nullable=False),
        sa.Column("parts", sa.Integer(), nullable=False),
        sa.Column("citation", sa.String(), nullable=False),
        sa.Column("text", sa.String(), nullable=False),
        sa.Column("references", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("corpus_version", sa.String(), nullable=False),
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
        sa.UniqueConstraint("ref", "content_hash", "occurrence"),
    )


def downgrade() -> None:
    op.drop_table("document_chunks")
