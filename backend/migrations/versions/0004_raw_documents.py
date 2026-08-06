"""raw documents

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-05 11:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.rename_table("ingested_documents", "raw_documents")
    op.execute(
        "ALTER TABLE raw_documents RENAME CONSTRAINT ingested_documents_pkey TO raw_documents_pkey"
    )
    op.execute(
        "ALTER TABLE raw_documents RENAME CONSTRAINT "
        "ingested_documents_ingest_run_id_fkey TO raw_documents_ingest_run_id_fkey"
    )
    op.drop_constraint("ingested_documents_ingest_run_id_name_key", "raw_documents", type_="unique")
    op.drop_column("raw_documents", "name")
    op.create_unique_constraint(
        "raw_documents_ingest_run_id_ref_key", "raw_documents", ["ingest_run_id", "ref"]
    )


def downgrade() -> None:
    op.drop_constraint("raw_documents_ingest_run_id_ref_key", "raw_documents", type_="unique")
    op.add_column("raw_documents", sa.Column("name", sa.String(), nullable=True))
    op.execute("UPDATE raw_documents SET name = ref")
    op.alter_column("raw_documents", "name", nullable=False)
    op.create_unique_constraint(
        "ingested_documents_ingest_run_id_name_key", "raw_documents", ["ingest_run_id", "name"]
    )
    op.execute(
        "ALTER TABLE raw_documents RENAME CONSTRAINT "
        "raw_documents_ingest_run_id_fkey TO ingested_documents_ingest_run_id_fkey"
    )
    op.execute(
        "ALTER TABLE raw_documents RENAME CONSTRAINT raw_documents_pkey TO ingested_documents_pkey"
    )
    op.rename_table("raw_documents", "ingested_documents")
