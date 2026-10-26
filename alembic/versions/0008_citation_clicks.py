"""Citation clicks: dedicated table for widget citation-click telemetry.

The widget used to beacon citation clicks to POST /feedback, where the
unknown fields were dropped and each click landed as an empty feedback row
(NULL rating, NULL comment). Clicks now go to POST /citations/click and
are stored here, keyed by the query they belong to.

Revision ID: 0008_citation_clicks
Revises: 0007_schema_cleanup
"""
from alembic import op
import sqlalchemy as sa

revision = "0008_citation_clicks"
down_revision = "0007_schema_cleanup"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "citation_clicks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "query_log_id",
            sa.String(36),
            sa.ForeignKey("query_logs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("chunk_id", sa.String(36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_citation_clicks_query_log_id", "citation_clicks", ["query_log_id"])


def downgrade() -> None:
    op.drop_index("ix_citation_clicks_query_log_id", table_name="citation_clicks")
    op.drop_table("citation_clicks")
