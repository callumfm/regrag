"""drop raw document url

Revision ID: 0014
Revises: 0013
Create Date: 2026-08-13 16:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0014"
down_revision: str | Sequence[str] | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

URL_EXPRESSION = (
    "'https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:' || resolved_celex"
)


def upgrade() -> None:
    """Every row's url was the EUR-Lex template applied to its own resolved_celex, so the column
    held nothing the row did not already say; FetchedVersion.url now derives it."""
    op.drop_column("raw_documents", "url")


def downgrade() -> None:
    op.add_column("raw_documents", sa.Column("url", sa.String(), nullable=True))
    op.execute(f"UPDATE raw_documents SET url = {URL_EXPRESSION}")
    op.alter_column("raw_documents", "url", nullable=False)
