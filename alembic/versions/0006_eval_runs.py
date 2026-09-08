"""eval_runs table for the evaluation harness

Revision ID: 0006_eval_runs
Revises: 0005_persons
Create Date: 2026-08-10 12:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0006_eval_runs"
down_revision = "0005_persons"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "eval_runs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("label", sa.String(length=200)),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="running"),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("results", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("cases_total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_eval_runs_created_at", "eval_runs", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_eval_runs_created_at", table_name="eval_runs")
    op.drop_table("eval_runs")
