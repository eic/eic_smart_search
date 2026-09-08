"""Schema cleanup: drop write-only tables/columns and unused indexes.

Findings from the 2026-08 schema audit:
- permissions_metadata: written on every document upsert, read nowhere —
  visibility is enforced via chunks.visibility.
- ingestion_state: write-only restatement of the newest crawl_jobs.stats.
- Assorted write-only columns (never read by any query, endpoint or
  dashboard) and indexes no query pattern uses.
- crawl_jobs gains the one index its only reader (ORDER BY created_at
  DESC) actually needs.

No user-facing data is touched: query_logs keeps query/answer/tokens/
cost/confidence history intact (total_tokens was redundant with
prompt+completion; filters/embedding_* were constants).

Revision ID: 0007_schema_cleanup
Revises: 0006_eval_runs
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0007_schema_cleanup"
down_revision = "0006_eval_runs"
branch_labels = None
depends_on = None

# DROP INDEX IF EXISTS keeps the migration idempotent across both deployed
# databases (smartsearch + starsearch).
_DEAD_INDEXES = (
    "ix_sources_source_type",
    "ix_sources_visibility",
    "ix_documents_visibility",
    "ix_documents_content_hash",
    "ix_chunks_source_id",
    "ix_chunks_content_hash",
    "ix_crawl_jobs_source_id",
    "ix_crawl_jobs_connector",
    "ix_crawl_jobs_status",
    "ix_query_logs_scope",
    "ix_query_logs_generation_provider",
    "ix_feedback_query_log_id",
    "ix_persons_orcid",
    "ix_persons_github_login",
)


def upgrade() -> None:
    # Dead tables.
    op.drop_table("permissions_metadata")
    op.drop_table("ingestion_state")

    # Write-only columns.
    op.drop_column("sources", "enabled")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS qdrant_point_id")
    op.drop_column("query_logs", "filters")
    op.drop_column("query_logs", "embedding_provider")
    op.drop_column("query_logs", "embedding_model")
    op.drop_column("query_logs", "total_tokens")
    op.drop_column("feedback", "selected_citation_ids")
    op.drop_column("feedback", "feedback_metadata")
    op.drop_column("persons", "github_login")
    op.drop_column("persons", "notes")
    op.drop_column("persons", "person_metadata")

    # Indexes with no matching query pattern.
    for name in _DEAD_INDEXES:
        op.execute(f"DROP INDEX IF EXISTS {name}")

    # The dashboard's jobs list sorts the whole table by created_at DESC.
    op.create_index("ix_crawl_jobs_created_at", "crawl_jobs", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_crawl_jobs_created_at", table_name="crawl_jobs")

    op.add_column("persons", sa.Column("person_metadata", JSONB, nullable=False, server_default="{}"))
    op.add_column("persons", sa.Column("notes", sa.Text))
    op.add_column("persons", sa.Column("github_login", sa.String(120)))
    op.add_column("feedback", sa.Column("feedback_metadata", JSONB, nullable=False, server_default="{}"))
    op.add_column("feedback", sa.Column("selected_citation_ids", JSONB, nullable=False, server_default="[]"))
    op.add_column("query_logs", sa.Column("total_tokens", sa.Integer))
    op.add_column("query_logs", sa.Column("embedding_model", sa.String(160)))
    op.add_column("query_logs", sa.Column("embedding_provider", sa.String(40)))
    op.add_column("query_logs", sa.Column("filters", JSONB, nullable=False, server_default="{}"))
    op.add_column("chunks", sa.Column("qdrant_point_id", sa.String(80)))
    op.add_column("sources", sa.Column("enabled", sa.Boolean, nullable=False, server_default="true"))

    op.create_table(
        "ingestion_state",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("source_id", sa.String(36), sa.ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("state_key", sa.String(180), nullable=False),
        sa.Column("state_value", JSONB, nullable=False, server_default="{}"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("source_id", "state_key", name="uq_ingestion_state_source_key"),
    )
    op.create_table(
        "permissions_metadata",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("document_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("visibility", sa.String(24), nullable=False, index=True),
        sa.Column("allowed_groups", JSONB, nullable=False, server_default="[]"),
        sa.Column("denied_groups", JSONB, nullable=False, server_default="[]"),
        sa.Column("policy_metadata", JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
