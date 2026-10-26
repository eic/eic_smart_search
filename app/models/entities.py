import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    base_url: Mapped[str | None] = mapped_column(Text)
    visibility: Mapped[str] = mapped_column(String(24), nullable=False, default="public", index=True)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )

    documents: Mapped[list["Document"]] = relationship(back_populates="source", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("source_id", "external_id", name="uq_documents_source_external_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True)
    external_id: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    source_name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    repo_path: Mapped[str | None] = mapped_column(Text, index=True)
    filetype: Mapped[str | None] = mapped_column(String(40), index=True)
    visibility: Mapped[str] = mapped_column(String(24), nullable=False, default="public", index=True)
    section_path: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    last_updated: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    doc_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )

    source: Mapped[Source] = relationship(back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    heading_path: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    visibility: Mapped[str] = mapped_column(String(24), nullable=False, default="public", index=True)
    chunk_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )

    document: Mapped[Document] = relationship(back_populates="chunks")
    source: Mapped[Source] = relationship()


class CrawlJob(Base):
    __tablename__ = "crawl_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    source_id: Mapped[str | None] = mapped_column(ForeignKey("sources.id", ondelete="SET NULL"), index=True)
    connector: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued", index=True)
    requested_by: Mapped[str | None] = mapped_column(String(160))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)
    stats: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    query_log_id: Mapped[str | None] = mapped_column(ForeignKey("query_logs.id", ondelete="SET NULL"), index=True)
    query: Mapped[str | None] = mapped_column(Text)
    rating: Mapped[int | None] = mapped_column(Integer)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class CitationClick(Base):
    __tablename__ = "citation_clicks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    query_log_id: Mapped[str | None] = mapped_column(ForeignKey("query_logs.id", ondelete="SET NULL"), index=True)
    chunk_id: Mapped[str] = mapped_column(String(36), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    scope: Mapped[str] = mapped_column(String(24), nullable=False, default="public")
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    answer_generated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    result_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    retrieval_debug: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    generation_provider: Mapped[str | None] = mapped_column(String(40))
    generation_model: Mapped[str | None] = mapped_column(String(160))
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[str | None] = mapped_column(String(16))
    top_score: Mapped[float | None] = mapped_column(Float)
    answer: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)


class EvalRun(Base):
    """One evaluation-harness run over the golden set (see app/eval)."""

    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    label: Mapped[str | None] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running")
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    results: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    cases_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Person(Base):
    """Aggregated record of a person mentioned across the indexed sources.

    Built by `app.people.builder.rebuild_people` — never written from a /query
    request. The `activity` JSONB carries the per-source breadcrumbs that the
    LLM tool surfaces ("recent talks", "recent papers", "recent commits").
    """

    __tablename__ = "persons"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    canonical_name: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_name: Mapped[str] = mapped_column(Text, nullable=False, unique=True, index=True)
    aliases: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    affiliations: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    orcid: Mapped[str | None] = mapped_column(String(32))
    email: Mapped[str | None] = mapped_column(String(320))
    working_groups: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    activity: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    activity_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )
