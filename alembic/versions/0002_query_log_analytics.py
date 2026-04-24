"""query log analytics fields

Revision ID: 0002_query_log_analytics
Revises: 0001_initial
Create Date: 2026-04-23 00:00:00
"""
from alembic import op
import sqlalchemy as sa

revision = "0002_query_log_analytics"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("query_logs", sa.Column("embedding_provider", sa.String(length=40)))
    op.add_column("query_logs", sa.Column("embedding_model", sa.String(length=160)))
    op.add_column("query_logs", sa.Column("generation_provider", sa.String(length=40)))
    op.add_column("query_logs", sa.Column("generation_model", sa.String(length=160)))
    op.add_column("query_logs", sa.Column("prompt_tokens", sa.Integer()))
    op.add_column("query_logs", sa.Column("completion_tokens", sa.Integer()))
    op.add_column("query_logs", sa.Column("total_tokens", sa.Integer()))
    op.add_column("query_logs", sa.Column("cost_usd", sa.Float()))
    op.add_column("query_logs", sa.Column("confidence", sa.String(length=16)))
    op.add_column("query_logs", sa.Column("top_score", sa.Float()))
    op.create_index("ix_query_logs_created_at", "query_logs", ["created_at"])
    op.create_index("ix_query_logs_generation_provider", "query_logs", ["generation_provider"])
    op.create_index("ix_query_logs_confidence", "query_logs", ["confidence"])
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_query_logs_query_trgm "
        "ON query_logs USING GIN (query gin_trgm_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_query_logs_query_trgm")
    op.drop_index("ix_query_logs_confidence", table_name="query_logs")
    op.drop_index("ix_query_logs_generation_provider", table_name="query_logs")
    op.drop_index("ix_query_logs_created_at", table_name="query_logs")
    op.drop_column("query_logs", "top_score")
    op.drop_column("query_logs", "confidence")
    op.drop_column("query_logs", "cost_usd")
    op.drop_column("query_logs", "total_tokens")
    op.drop_column("query_logs", "completion_tokens")
    op.drop_column("query_logs", "prompt_tokens")
    op.drop_column("query_logs", "generation_model")
    op.drop_column("query_logs", "generation_provider")
    op.drop_column("query_logs", "embedding_model")
    op.drop_column("query_logs", "embedding_provider")
