"""persons table for the lookup_person tool

Revision ID: 0005_persons
Revises: 0004_document_title_fts
Create Date: 2026-05-04 12:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0005_persons"
down_revision = "0004_document_title_fts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "persons",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("canonical_name", sa.Text(), nullable=False),
        sa.Column("normalized_name", sa.Text(), nullable=False),
        sa.Column("aliases", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("affiliations", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("orcid", sa.String(length=32)),
        sa.Column("github_login", sa.String(length=120)),
        sa.Column("email", sa.String(length=320)),
        sa.Column("working_groups", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("activity", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("activity_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("last_seen", sa.DateTime(timezone=True)),
        sa.Column("notes", sa.Text()),
        sa.Column("person_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_persons_normalized_name", "persons", ["normalized_name"], unique=True)
    op.create_index("ix_persons_orcid", "persons", ["orcid"])
    op.create_index("ix_persons_github_login", "persons", ["github_login"])
    # Trigram index lets `lookup_person` do fuzzy match on partial / misspelled
    # names without scanning the table. pg_trgm is already enabled by 0001_initial.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_persons_canonical_name_trgm "
        "ON persons USING GIN (canonical_name gin_trgm_ops)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_persons_normalized_name_trgm "
        "ON persons USING GIN (normalized_name gin_trgm_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_persons_normalized_name_trgm")
    op.execute("DROP INDEX IF EXISTS ix_persons_canonical_name_trgm")
    op.drop_index("ix_persons_github_login", table_name="persons")
    op.drop_index("ix_persons_orcid", table_name="persons")
    op.drop_index("ix_persons_normalized_name", table_name="persons")
    op.drop_table("persons")
