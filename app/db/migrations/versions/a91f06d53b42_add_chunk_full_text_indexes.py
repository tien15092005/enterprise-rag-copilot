"""add chunk full-text indexes

Revision ID: a91f06d53b42
Revises: 740cad2326eb
Create Date: 2026-09-27 16:20:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "a91f06d53b42"
down_revision: str | None = "740cad2326eb"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            "CREATE INDEX ix_chunks_fts_english ON chunks "
            "USING gin (to_tsvector('english'::regconfig, text)) WHERE language = 'en'"
        )
        op.execute(
            "CREATE INDEX ix_chunks_fts_simple ON chunks "
            "USING gin (to_tsvector('simple'::regconfig, text)) WHERE language = 'vi'"
        )


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_chunks_fts_simple")
        op.execute("DROP INDEX IF EXISTS ix_chunks_fts_english")
