"""GIN index on document.title tsvector for hybrid retrieval title boost

Revision ID: 0004_document_title_fts
Revises: 0003_query_log_answer
Create Date: 2026-04-24 16:15:00
"""
from alembic import op

revision = "0004_document_title_fts"
down_revision = "0003_query_log_answer"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_documents_title_fts "
        "ON documents USING GIN (to_tsvector('english', coalesce(title, '')))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_documents_title_fts")
