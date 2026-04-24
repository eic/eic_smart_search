"""store answer text on query_logs for analytics dashboard

Revision ID: 0003_query_log_answer
Revises: 0002_query_log_analytics
Create Date: 2026-04-24 15:00:00
"""
from alembic import op
import sqlalchemy as sa

revision = "0003_query_log_answer"
down_revision = "0002_query_log_analytics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("query_logs", sa.Column("answer", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("query_logs", "answer")
