"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-04-20 00:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.create_table(
        "sources",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("source_type", sa.String(length=40), nullable=False),
        sa.Column("base_url", sa.Text()),
        sa.Column("visibility", sa.String(length=24), nullable=False, server_default="public"),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_sources_name", "sources", ["name"], unique=True)
    op.create_index("ix_sources_source_type", "sources", ["source_type"])
    op.create_index("ix_sources_visibility", "sources", ["visibility"])

    op.create_table(
        "documents",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("source_id", sa.String(length=36), sa.ForeignKey("sources.id", ondelete="CASCADE"), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(length=40), nullable=False),
        sa.Column("source_name", sa.String(length=120), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("repo_path", sa.Text()),
        sa.Column("filetype", sa.String(length=40)),
        sa.Column("visibility", sa.String(length=24), nullable=False, server_default="public"),
        sa.Column("section_path", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("last_updated", sa.DateTime(timezone=True)),
        sa.Column("doc_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("source_id", "external_id", name="uq_documents_source_external_id"),
    )
    op.create_index("ix_documents_source_id", "documents", ["source_id"])
    op.create_index("ix_documents_source_name", "documents", ["source_name"])
    op.create_index("ix_documents_source_type", "documents", ["source_type"])
    op.create_index("ix_documents_visibility", "documents", ["visibility"])
    op.create_index("ix_documents_repo_path", "documents", ["repo_path"])
    op.create_index("ix_documents_filetype", "documents", ["filetype"])
    op.create_index("ix_documents_content_hash", "documents", ["content_hash"])

    op.create_table(
        "chunks",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("document_id", sa.String(length=36), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_id", sa.String(length=36), sa.ForeignKey("sources.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("heading_path", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("qdrant_point_id", sa.String(length=80)),
        sa.Column("visibility", sa.String(length=24), nullable=False, server_default="public"),
        sa.Column("chunk_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
        sa.UniqueConstraint("qdrant_point_id", name="uq_chunks_qdrant_point_id"),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_source_id", "chunks", ["source_id"])
    op.create_index("ix_chunks_visibility", "chunks", ["visibility"])
    op.create_index("ix_chunks_content_hash", "chunks", ["content_hash"])
    op.execute("CREATE INDEX ix_chunks_content_fts ON chunks USING GIN (to_tsvector('english', content))")

    op.create_table(
        "crawl_jobs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("source_id", sa.String(length=36), sa.ForeignKey("sources.id", ondelete="SET NULL")),
        sa.Column("connector", sa.String(length=80), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="queued"),
        sa.Column("requested_by", sa.String(length=160)),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("error", sa.Text()),
        sa.Column("stats", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_crawl_jobs_source_id", "crawl_jobs", ["source_id"])
    op.create_index("ix_crawl_jobs_connector", "crawl_jobs", ["connector"])
    op.create_index("ix_crawl_jobs_status", "crawl_jobs", ["status"])

    op.create_table(
        "ingestion_state",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("source_id", sa.String(length=36), sa.ForeignKey("sources.id", ondelete="CASCADE"), nullable=False),
        sa.Column("state_key", sa.String(length=180), nullable=False),
        sa.Column("state_value", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("source_id", "state_key", name="uq_ingestion_state_source_key"),
    )
    op.create_index("ix_ingestion_state_source_id", "ingestion_state", ["source_id"])

    op.create_table(
        "permissions_metadata",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("document_id", sa.String(length=36), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("visibility", sa.String(length=24), nullable=False),
        sa.Column("allowed_groups", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("denied_groups", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("policy_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_permissions_metadata_document_id", "permissions_metadata", ["document_id"])
    op.create_index("ix_permissions_metadata_visibility", "permissions_metadata", ["visibility"])

    op.create_table(
        "query_logs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("scope", sa.String(length=24), nullable=False, server_default="public"),
        sa.Column("filters", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("top_k", sa.Integer(), nullable=False),
        sa.Column("answer_generated", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("result_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Integer()),
        sa.Column("retrieval_debug", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_query_logs_scope", "query_logs", ["scope"])

    op.create_table(
        "feedback",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("query_log_id", sa.String(length=36), sa.ForeignKey("query_logs.id", ondelete="SET NULL")),
        sa.Column("query", sa.Text()),
        sa.Column("rating", sa.Integer()),
        sa.Column("comment", sa.Text()),
        sa.Column("selected_citation_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("feedback_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_feedback_query_log_id", "feedback", ["query_log_id"])


def downgrade() -> None:
    op.drop_table("feedback")
    op.drop_table("query_logs")
    op.drop_table("permissions_metadata")
    op.drop_table("ingestion_state")
    op.drop_table("crawl_jobs")
    op.drop_index("ix_chunks_content_fts", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.drop_table("sources")

