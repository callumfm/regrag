"""rename ref to celex

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-06 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CONSTRAINTS = (
    (
        "raw_documents",
        "raw_documents_ingest_run_id_ref_key",
        "raw_documents_ingest_run_id_celex_key",
    ),
    (
        "document_chunks",
        "document_chunks_ref_content_hash_occurrence_key",
        "document_chunks_celex_content_hash_occurrence_key",
    ),
)


def upgrade() -> None:
    """CELEX is the EU's identity scheme; 'ref' said nothing and collided with 'references'."""
    op.alter_column("raw_documents", "ref", new_column_name="celex")
    op.alter_column("raw_documents", "resolved_ref", new_column_name="resolved_celex")
    op.alter_column("document_chunks", "ref", new_column_name="celex")
    for table, old, new in CONSTRAINTS:
        op.execute(f"ALTER TABLE {table} RENAME CONSTRAINT {old} TO {new}")


def downgrade() -> None:
    for table, old, new in CONSTRAINTS:
        op.execute(f"ALTER TABLE {table} RENAME CONSTRAINT {new} TO {old}")
    op.alter_column("document_chunks", "celex", new_column_name="ref")
    op.alter_column("raw_documents", "resolved_celex", new_column_name="resolved_ref")
    op.alter_column("raw_documents", "celex", new_column_name="ref")
