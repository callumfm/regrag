"""document chunks ingest run fk

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-06 09:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

FK = "document_chunks_ingest_run_id_fkey"


def upgrade() -> None:
    """Chunks point at the run that first stored them; the version comes from that run.

    A chunk whose version matches no run cannot be attributed, so it is dropped rather
    than pointed at an arbitrary run — the next ingest restores it.
    """
    op.add_column("document_chunks", sa.Column("ingest_run_id", sa.Integer(), nullable=True))
    op.execute(
        """
        UPDATE document_chunks c
           SET ingest_run_id = (
               SELECT min(r.id) FROM ingest_runs r WHERE r.corpus_version = c.corpus_version
           )
        """
    )
    op.execute("DELETE FROM document_chunks WHERE ingest_run_id IS NULL")
    op.alter_column("document_chunks", "ingest_run_id", nullable=False)
    op.create_foreign_key(
        FK, "document_chunks", "ingest_runs", ["ingest_run_id"], ["id"], ondelete="CASCADE"
    )
    op.drop_column("document_chunks", "corpus_version")


def downgrade() -> None:
    op.add_column("document_chunks", sa.Column("corpus_version", sa.String(), nullable=True))
    op.execute(
        """
        UPDATE document_chunks c
           SET corpus_version = (
               SELECT r.corpus_version FROM ingest_runs r WHERE r.id = c.ingest_run_id
           )
        """
    )
    op.drop_constraint(FK, "document_chunks", type_="foreignkey")
    op.drop_column("document_chunks", "ingest_run_id")
